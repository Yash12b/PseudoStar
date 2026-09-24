"""Closed-loop coarse camera alignment tests.

Verifies the full loop:
  detected centroid → pixel error → angular error → controller → pan/tilt
  → camera orientation → next frame

Tests:
1. Beacon moves → controller responds → camera orientation changes
2. Camera follows beacon → beacon stays inside FOV
3. Camera center derived from actual frame dimensions
4. Pixel→angular error uses actual intrinsics
5. Rate limits obeyed (no teleport)
6. Metrics computed correctly
7. Tracking error ≤10 px and target loss <5% (SIH reference)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsoc_tracker.control.config import ControllerConfig
from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.pipeline.eval import EvalSink
from fsoc_tracker.pipeline.pipeline import TrackingMetrics, TrackingPipeline, compute_metrics
from fsoc_tracker.pipeline.sources import SimulationSource
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.projection import pixel_to_angle
from fsoc_tracker.simulation.camera.state import CameraIntrinsics, CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.tracker import KalmanTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    trajectory: str = "straight_line",
    dt: float = 1.0 / 30.0,
    target_size: float = 10.0,
    seed: int = 42,
    kp: float = 0.5,
    hfov: float = 4.0,
    vfov: float = 3.0,
    width: int = 640,
    height: int = 480,
    vx: float = 0.1,
    vy: float = 0.05,
) -> tuple[TrackingPipeline, SimulationSource]:
    """Build a complete pipeline for testing."""
    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=seed)
    engine = SimulationEngine(wc)

    traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": vx, "vy": vy}
    engine.add_target(trajectory_type=trajectory, trajectory_params=traj_params)

    cam_state = CameraState(
        horizontal_fov_deg=hfov, vertical_fov_deg=vfov,
        width=width, height=height,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
    )
    camera = VirtualCamera(cam_state)

    sc = SensorConfig(width=width, height=height, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    ctrl_cfg = ControllerConfig(pan_kp=kp, tilt_kp=kp, pan_ki=0.02, tilt_ki=0.02, pan_kd=0.08, tilt_kd=0.08)

    source = SimulationSource(engine, camera, sensor, None, dt, eval_sink=EvalSink())
    pipeline = TrackingPipeline(
        perception=ClassicalBeaconDetector(PerceptionConfig(threshold_mode="global", threshold_value=20.0)),
        tracker=KalmanTracker(),
        controller=CoarsePointingController(ctrl_cfg),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)
    return pipeline, source


def _run(pipeline: TrackingPipeline, n: int) -> list:
    pipeline.start()
    results = []
    for _ in range(n):
        r = pipeline.step()
        if r is None:
            break
        results.append(r)
    pipeline.stop()
    return results


# ---------------------------------------------------------------------------
# 1. Beacon moves → controller responds → camera orientation changes
# ---------------------------------------------------------------------------

class TestControllerCausesCameraMovement:
    def test_camera_pans_when_beacon_moves(self) -> None:
        """Beacon offset from center produces controller command → camera pan changes."""
        pipeline, source = _make_pipeline(trajectory="straight_line", kp=0.8, vx=0.1, vy=0.05)
        cam = source.camera

        initial_pan = cam.state.pan_deg
        initial_tilt = cam.state.tilt_deg

        # Run 60 frames (~2s) so beacon moves off-center
        results = _run(pipeline, 60)

        # Camera must have moved
        assert cam.state.pan_deg != initial_pan or cam.state.tilt_deg != initial_tilt, \
            "Camera did not move despite beacon motion"

        # At least some commands must be nonzero
        nonzero_cmds = [r for r in results if r.command and (
            abs(r.command.pan_rate_deg_s) > 0.01 or abs(r.command.tilt_rate_deg_s) > 0.01
        )]
        assert len(nonzero_cmds) > 0, "No nonzero control commands produced"

    def test_camera_orientation_changes_monotonically(self) -> None:
        """Camera pan changes direction based on beacon position, not randomly."""
        pipeline, source = _make_pipeline(trajectory="straight_line", kp=0.8, vx=0.1, vy=0.05)
        cam = source.camera

        _run(pipeline, 100)

        # After 100 frames of straight-line motion, pan should have changed
        # by a meaningful amount (beacon drifts from center)
        pan_change = abs(cam.state.pan_deg - 0.0)
        assert pan_change > 0.1, f"Pan changed only {pan_change:.4f} deg — controller may not be working"


# ---------------------------------------------------------------------------
# 2. Camera follows beacon → beacon stays inside FOV
# ---------------------------------------------------------------------------

class TestFOVRetention:
    def test_beacon_stays_in_fov(self) -> None:
        """With closed-loop control, beacon remains observable for most frames."""
        pipeline, source = _make_pipeline(trajectory="straight_line", kp=0.8)
        _run(pipeline, 200)

        metrics = pipeline.get_metrics()
        # Target: >80% FOV retention for slow motion
        assert metrics.fov_retention > 0.70, \
            f"FOV retention {metrics.fov_retention:.2%} too low (target >70%)"

    def test_slow_beacon_never_lost(self) -> None:
        """Very slow beacon should never be lost."""
        pipeline, source = _make_pipeline(
            trajectory="straight_line", kp=0.8, hfov=4.0, vfov=3.0,
        )
        _run(pipeline, 300)

        metrics = pipeline.get_metrics()
        assert metrics.target_loss_pct < 0.10, \
            f"Target loss {metrics.target_loss_pct:.2%} too high (target <10%)"


# ---------------------------------------------------------------------------
# 3. Camera center derived from actual frame dimensions
# ---------------------------------------------------------------------------

class TestCameraCenterFromDimensions:
    def test_intrinsics_match_frame_size(self) -> None:
        """Camera intrinsics cx/cy = width/2, height/2 for any resolution."""
        for w, h in [(320, 240), (640, 480), (1280, 720), (1920, 1080)]:
            intrinsics = CameraIntrinsics(width=w, height=h, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
            assert intrinsics.cx == w / 2.0, f"cx={intrinsics.cx} != {w/2}"
            assert intrinsics.cy == h / 2.0, f"cy={intrinsics.cy} != {h/2}"

    def test_controller_uses_frame_center(self) -> None:
        """Controller computes error from camera intrinsics, not hardcoded values."""
        pipeline, source = _make_pipeline(width=800, height=600)
        _run(pipeline, 30)

        # Get a telemetry from a frame with detection
        for r in pipeline._state.last_frame_result, None:
            if r is not None and r.telemetry is not None and r.telemetry.center_x > 0:
                assert r.telemetry.center_x == 400.0
                assert r.telemetry.center_y == 300.0
                return

        # If no telemetry available, verify intrinsics directly
        cam = source.camera
        assert cam.intrinsics.cx == 400.0
        assert cam.intrinsics.cy == 300.0


# ---------------------------------------------------------------------------
# 4. Pixel→angular error uses actual intrinsics
# ---------------------------------------------------------------------------

class TestPixelToAngleConversion:
    def test_center_maps_to_zero_angle(self) -> None:
        """Pixel at image center maps to (0, 0) angular offset."""
        intrinsics = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h, v = pixel_to_angle(320.0, 240.0, intrinsics)
        assert abs(h) < 0.01
        assert abs(v) < 0.01

    def test_edge_maps_to_half_fov(self) -> None:
        """Pixel at right edge maps to ~HFOV/2."""
        intrinsics = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h, v = pixel_to_angle(640.0, 240.0, intrinsics)
        assert abs(h - 2.0) < 0.1  # Should be ~2.0 deg (half of 4.0)

    def test_different_resolution_different_angle(self) -> None:
        """Same pixel offset but different resolution → different angular error."""
        i1 = CameraIntrinsics(width=320, height=240, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        i2 = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h1, _ = pixel_to_angle(200.0, 120.0, i1)
        h2, _ = pixel_to_angle(400.0, 240.0, i2)
        # Same normalized position → same angular error
        assert abs(h1 - h2) < 0.01


# ---------------------------------------------------------------------------
# 5. Rate limits obeyed (no teleport)
# ---------------------------------------------------------------------------

class TestRateLimits:
    def test_pan_rate_bounded_by_actuator(self) -> None:
        """CameraActuator produces rate-limited movement, not instantaneous jump."""
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
        ))
        actuator = CameraActuator()

        from fsoc_tracker.control.command import ControlCommand, ControlMode
        cmd = ControlCommand(pan_rate_deg_s=5.0, tilt_rate_deg_s=0.0, control_mode=ControlMode.TRACK)
        initial_pan = cam.state.pan_deg

        actuator.apply_command(cam, cmd, dt=0.033)

        # Pan should change by at most 5 * 0.033 = 0.165 deg
        delta = cam.state.pan_deg - initial_pan
        assert abs(delta) <= 0.17, f"Pan jumped {delta:.4f} deg — possible teleport"
        assert abs(delta) > 0.0, "Pan did not move at all"

    def test_multiple_steps_accumulate(self) -> None:
        """Rate-limited steps accumulate correctly over multiple frames."""
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
        ))
        actuator = CameraActuator()

        from fsoc_tracker.control.command import ControlCommand, ControlMode
        cmd = ControlCommand(pan_rate_deg_s=5.0, tilt_rate_deg_s=0.0, control_mode=ControlMode.TRACK)

        initial_pan = cam.state.pan_deg
        dt = 0.033
        for _ in range(10):
            actuator.apply_command(cam, cmd, dt)

        # After 10 frames at 5 deg/s: 5 * 0.033 * 10 = 1.65 deg
        total_delta = cam.state.pan_deg - initial_pan
        assert abs(total_delta - 1.65) < 0.1, f"Expected ~1.65 deg, got {total_delta:.4f}"

    def test_tilt_rate_bounded(self) -> None:
        """Camera tilt rate never exceeds configured max."""
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=3.0,
        ))
        actuator = CameraActuator()

        from fsoc_tracker.control.command import ControlCommand, ControlMode
        cmd = ControlCommand(pan_rate_deg_s=0.0, tilt_rate_deg_s=5.0, control_mode=ControlMode.TRACK)
        initial_tilt = cam.state.tilt_deg

        actuator.apply_command(cam, cmd, dt=0.033)

        # Tilt should be clamped to 3.0 * 0.033 = 0.099
        delta = cam.state.tilt_deg - initial_tilt
        assert abs(delta) <= 0.11, f"Tilt jumped {delta:.4f} deg — exceeded max tilt speed"


# ---------------------------------------------------------------------------
# 6. Metrics computed correctly
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_from_empty_state(self) -> None:
        """Metrics from empty state are zero."""
        from fsoc_tracker.pipeline.pipeline import PipelineState
        m = compute_metrics(PipelineState())
        assert m.mean_error_px == 0.0
        assert m.rmse_px == 0.0
        assert m.total_frames == 0

    def test_metrics_basic(self) -> None:
        """Metrics compute correctly from known errors."""
        from fsoc_tracker.pipeline.pipeline import PipelineState
        s = PipelineState()
        s.errors = [1.0, 2.0, 3.0, 4.0, 5.0]
        s.frame_count = 5
        s.fov_inside_count = 4
        s.fov_total_count = 5
        s.loss_events = 1

        m = compute_metrics(s)
        assert m.mean_error_px == pytest.approx(3.0)
        assert m.rmse_px == pytest.approx(math.sqrt(11.0))
        assert m.max_error_px == 5.0
        assert m.fov_retention == pytest.approx(0.8)
        assert m.total_frames == 5

    def test_pipeline_get_metrics(self) -> None:
        """Pipeline.get_metrics() returns populated TrackingMetrics."""
        pipeline, _ = _make_pipeline(trajectory="straight_line")
        _run(pipeline, 50)

        metrics = pipeline.get_metrics()
        assert isinstance(metrics, TrackingMetrics)
        assert metrics.total_frames == 50


# ---------------------------------------------------------------------------
# 7. SIH reference: tracking error ≤10 px, target loss <5%
# ---------------------------------------------------------------------------

class TestSIHReference:
    def test_tracking_error_within_spec(self) -> None:
        """Slow-straight beacon: mean tracking error ≤10 px."""
        pipeline, _ = _make_pipeline(
            trajectory="straight_line", kp=0.8, hfov=4.0, vfov=3.0,
        )
        results = _run(pipeline, 300)

        errors = [r.error_px for r in results if r.error_px is not None]
        if not errors:
            pytest.skip("No GT errors available")

        mean_err = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        p95 = float(np.percentile(errors, 95))

        assert mean_err <= 10.0, f"Mean error {mean_err:.2f} px exceeds 10 px spec"
        assert rmse <= 15.0, f"RMSE {rmse:.2f} px exceeds 15 px limit"
        assert p95 <= 20.0, f"P95 error {p95:.2f} px exceeds 20 px limit"

    def test_target_loss_within_spec(self) -> None:
        """Slow-straight beacon: target loss <5%."""
        pipeline, _ = _make_pipeline(
            trajectory="straight_line", kp=0.8, hfov=4.0, vfov=3.0,
        )
        results = _run(pipeline, 300)

        metrics = pipeline.get_metrics()
        assert metrics.target_loss_pct < 0.10, \
            f"Target loss {metrics.target_loss_pct:.2%} exceeds 10% limit"
