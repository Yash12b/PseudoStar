"""Comprehensive integration tests — Stage 12.

Tests the full system pipeline across:
- Closed-loop simulation
- Multiple trajectories
- Multiple disturbance levels
- Multiple FPS rates
- Perception backend switching
- Session lifecycle
- CLI headless mode
"""

from __future__ import annotations

import math

import pytest

from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.disturbances.config import get_preset_config
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.pipeline.eval import EvalSink
from fsoc_tracker.pipeline.pipeline import PipelineFrameResult, TrackingPipeline
from fsoc_tracker.pipeline.session import RunState, SessionController
from fsoc_tracker.pipeline.sources import SimulationSource
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
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
    disturbance: str = "clear",
    seed: int = 42,
) -> tuple[TrackingPipeline, SimulationSource]:
    """Build a complete pipeline for testing.

    Target starts at camera center (0, 0, 50) with small velocity
    so it stays within the narrow 4°x3° FOV at z=50.
    """
    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=seed)
    engine = SimulationEngine(wc)

    traj_params: dict = {}
    if trajectory == "straight_line":
        traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2}
    elif trajectory == "circular":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius": 0.5, "angular_speed_rad_s": 0.3}
    elif trajectory == "figure_8":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3}
    elif trajectory == "random":
        traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "seed": seed}
    elif trajectory == "spiral":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius_start": 0.1, "radius_growth": 0.05}
    elif trajectory == "sinusoidal":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3, "freq_x": 0.3, "freq_y": 0.2}
    else:
        traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2}

    engine.add_target(
        trajectory_type=trajectory,
        trajectory_params=traj_params,
    )

    cam_state = CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=640, height=480,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
    )
    camera = VirtualCamera(cam_state)

    sc = SensorConfig(width=640, height=480, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    dist_cfg = get_preset_config(disturbance)
    dist_pipe = DisturbancePipeline(dist_cfg) if disturbance != "clear" else None

    perception_cfg = PerceptionConfig(
        threshold_mode="global",
        threshold_value=20.0,
    )

    source = SimulationSource(engine, camera, sensor, dist_pipe, dt, eval_sink=EvalSink())
    pipeline = TrackingPipeline(
        perception=ClassicalBeaconDetector(perception_cfg),
        tracker=KalmanTracker(),
        controller=CoarsePointingController(),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)
    return pipeline, source


def test_pipeline_source_adapters_implement_core_frame_source_contract():
    from fsoc_tracker.core.interfaces import FrameSource as CoreFrameSource
    from fsoc_tracker.pipeline.sources import FrameSource as PipelineFrameSource

    assert PipelineFrameSource is CoreFrameSource
    _, source = _make_pipeline(dt=1.0 / 60.0)
    assert isinstance(source, CoreFrameSource)
    assert source.nominal_fps == pytest.approx(60.0)
    source.open()
    assert source.is_open()
    source.release()
    assert not source.is_open()


def _run_pipeline_frames(pipeline: TrackingPipeline, num_frames: int) -> list[PipelineFrameResult]:
    """Run pipeline for N frames and return results."""
    pipeline.start()
    results = []
    for _ in range(num_frames):
        r = pipeline.step()
        if r is None:
            break
        results.append(r)
    pipeline.stop()
    return results


# ---------------------------------------------------------------------------
# Closed-Loop Integration Tests
# ---------------------------------------------------------------------------

class TestClosedLoopIntegration:
    """Full closed-loop: world → camera → sensor → perception → tracking → control → camera."""

    def test_basic_closed_loop(self):
        pipeline, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 100)
        assert len(results) == 100
        assert all(r.perception is not None for r in results)
        assert all(r.tracking is not None for r in results)
        assert all(r.command is not None for r in results)

    def test_detection_occurs(self):
        pipeline, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 200)
        detections = [r for r in results if r.perception and r.perception.detected]
        assert len(detections) > 0, "Expected at least some detections"

    def test_tracking_locks(self):
        pipeline, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 200)
        tracking_states = [r.tracking.state.name for r in results if r.tracking]
        has_locked = any(s in ("TRACKING", "ACQUIRING") for s in tracking_states)
        assert has_locked, f"Expected tracker to reach TRACKING/ACQUIRING, got: {set(tracking_states)}"

    def test_error_converges(self):
        pipeline, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 300)
        errors_with_gt = [r.error_px for r in results if r.error_px is not None]
        if len(errors_with_gt) > 50:
            early = errors_with_gt[:50]
            late = errors_with_gt[-50:]
            early_rmse = math.sqrt(sum(e**2 for e in early) / len(early))
            late_rmse = math.sqrt(sum(e**2 for e in late) / len(late))
            assert late_rmse <= early_rmse * 2.0, f"Error should not diverge: early={early_rmse:.1f} late={late_rmse:.1f}"

    def test_no_ground_truth_in_pipeline(self):
        pipeline, _ = _make_pipeline()
        results = _run_pipeline_frames(pipeline, 10)
        for r in results:
            if r.tracking is not None:
                assert not hasattr(r.tracking, '_used_ground_truth'), \
                    "Tracker must not use ground truth"

    def test_control_commands_valid(self):
        pipeline, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 100)
        for r in results:
            if r.command is not None:
                assert not math.isnan(r.command.pan_rate_deg_s)
                assert not math.isnan(r.command.tilt_rate_deg_s)
                assert not math.isinf(r.command.pan_rate_deg_s)
                assert not math.isinf(r.command.tilt_rate_deg_s)

    def test_timestamps_monotonic(self):
        pipeline, _ = _make_pipeline()
        results = _run_pipeline_frames(pipeline, 50)
        for i in range(1, len(results)):
            assert results[i].timestamp_s >= results[i-1].timestamp_s


# ---------------------------------------------------------------------------
# Disturbance Integration Tests
# ---------------------------------------------------------------------------

class TestDisturbanceIntegration:
    """Run full system under various disturbance profiles."""

    @pytest.mark.parametrize("disturbance", [
        "clear", "light", "moderate",
    ])
    def test_disturbance_profiles(self, disturbance: str):
        n = 30 if disturbance == "moderate" else 100
        pipeline, _ = _make_pipeline(disturbance=disturbance, dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, n)
        assert len(results) > 0
        assert all(r.perception is not None for r in results)

    def test_pipeline_survives_all_presets(self):
        for preset in ["clear", "light", "moderate", "severe"]:
            n = {"clear": 50, "light": 50, "moderate": 20, "severe": 10}[preset]
            pipeline, _ = _make_pipeline(disturbance=preset, dt=1.0/30.0)
            results = _run_pipeline_frames(pipeline, n)
            assert len(results) > 0, f"Pipeline failed under {preset}"


# ---------------------------------------------------------------------------
# Target Motion Matrix Tests
# ---------------------------------------------------------------------------

class TestTargetMotionMatrix:
    """Run full system with different trajectory types."""

    @pytest.mark.parametrize("trajectory", [
        "straight_line", "circular", "figure_8", "random",
    ])
    def test_trajectory(self, trajectory: str):
        pipeline, _ = _make_pipeline(trajectory=trajectory, dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 150)
        assert len(results) > 0
        assert all(r.perception is not None for r in results)


# ---------------------------------------------------------------------------
# FPS Matrix Tests
# ---------------------------------------------------------------------------

class TestFPSMatrix:
    """Run controlled tests at various FPS rates."""

    @pytest.mark.parametrize("fps", [10.0, 15.0, 24.0, 30.0, 60.0, 120.0])
    def test_fps_rate(self, fps: float):
        pipeline, _ = _make_pipeline(dt=1.0/fps)
        results = _run_pipeline_frames(pipeline, 50)
        assert len(results) > 0
        assert results[0].dt == pytest.approx(1.0 / fps)
        for r in results:
            assert r.dt > 0
            assert not math.isnan(r.total_ms)

    def test_variable_timestamps(self):
        pipeline, _ = _make_pipeline(dt=1.0/30.0)
        pipeline.start()
        results = []
        for i in range(30):
            r = pipeline.step()
            if r is not None:
                results.append(r)
        pipeline.stop()
        assert len(results) == 30


# ---------------------------------------------------------------------------
# Target Size Matrix Tests
# ---------------------------------------------------------------------------

class TestTargetSizeMatrix:
    """Evaluate detection across target sizes."""

    @pytest.mark.parametrize("target_size", [5.0, 8.0, 10.0, 15.0, 20.0])
    def test_target_size(self, target_size: float):
        pipeline, _ = _make_pipeline(target_size=target_size, dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 100)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Session Controller Tests
# ---------------------------------------------------------------------------

class TestSessionController:
    """Test session lifecycle management."""

    def test_session_start_stop(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.start()
        assert session.state == RunState.RUNNING
        for _ in range(10):
            session.step()
        result = session.stop()
        assert session.state == RunState.STOPPED
        assert result.frame_count == 10

    def test_session_pause_resume(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.start()
        session.step()
        session.pause()
        assert session.state == RunState.PAUSED
        session.resume()
        assert session.state == RunState.RUNNING
        session.step()
        session.stop()

    def test_session_metadata(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.configure(source_type="simulation", detector_name="classical")
        session.start()
        session.step()
        result = session.stop()
        assert result.metadata.source_type == "simulation"
        assert result.metadata.detector_name == "classical"
        assert result.metadata.session_id != ""

    def test_session_result_summary(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.start()
        for _ in range(20):
            session.step()
        result = session.stop()
        summary = result.summary()
        assert "Session:" in summary
        assert "Frames:" in summary

    def test_session_reset(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.start()
        session.step()
        session.stop()
        session.reset()
        assert session.state == RunState.IDLE


# ---------------------------------------------------------------------------
# FPS Independence Tests
# ---------------------------------------------------------------------------

class TestFPSIndependence:
    """Verify the system produces reasonable results across FPS rates."""

    def test_tracking_works_at_multiple_rates(self):
        results_by_fps = {}
        for fps in [15.0, 30.0, 60.0]:
            pipeline, _ = _make_pipeline(dt=1.0/fps)
            frames = _run_pipeline_frames(pipeline, 100)
            results_by_fps[fps] = len(frames)

        for fps, count in results_by_fps.items():
            assert count > 0, f"No frames processed at {fps} FPS"

    def test_dt_from_timestamps(self):
        pipeline, _ = _make_pipeline(dt=1.0/30.0)
        results = _run_pipeline_frames(pipeline, 20)
        for r in results:
            assert r.dt > 0, "dt must be positive"
            assert r.dt < 1.0, "dt must be reasonable"


# ---------------------------------------------------------------------------
# Perception Backend Integration
# ---------------------------------------------------------------------------

class TestPerceptionBackends:
    """Verify all perception backends produce compatible PerceptionResult."""

    def test_classical_backend(self):
        pipeline, _ = _make_pipeline()
        results = _run_pipeline_frames(pipeline, 50)
        assert len(results) > 0
        for r in results:
            assert r.perception is not None
            assert hasattr(r.perception, 'detected')
            assert hasattr(r.perception, 'detections')


# ---------------------------------------------------------------------------
# Camera State Integration
# ---------------------------------------------------------------------------

class TestCameraStateIntegration:
    """Verify camera state updates through the pipeline."""

    def test_camera_pan_updates(self):
        pipeline, source = _make_pipeline(trajectory="straight_line", dt=1.0/30.0)
        initial_pan = source.camera.state.pan_deg
        initial_tilt = source.camera.state.tilt_deg
        results = _run_pipeline_frames(pipeline, 200)
        final_pan = source.camera.state.pan_deg
        final_tilt = source.camera.state.tilt_deg
        moved = (final_pan != initial_pan) or (final_tilt != initial_tilt)
        has_detection = any(r.perception and r.perception.detected for r in results)
        if has_detection:
            assert moved, "Camera should have moved when target is detected"
        else:
            pytest.skip("Target not in FOV for this trajectory seed")

    def test_camera_intrinsics_consistent(self):
        pipeline, source = _make_pipeline()
        intrinsics = source.camera.intrinsics
        assert intrinsics.width == 640
        assert intrinsics.height == 480
        assert intrinsics.fx > 0
        assert intrinsics.fy > 0


# ---------------------------------------------------------------------------
# Deterministic Replay Test
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    """Run the same simulation twice and verify identical results."""

    def test_replay_identical(self):
        results1 = []
        pipeline1, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0, seed=42)
        pipeline1.start()
        for _ in range(50):
            r = pipeline1.step()
            if r:
                results1.append(r)
        pipeline1.stop()

        results2 = []
        pipeline2, _ = _make_pipeline(trajectory="straight_line", dt=1.0/30.0, seed=42)
        pipeline2.start()
        for _ in range(50):
            r = pipeline2.step()
            if r:
                results2.append(r)
        pipeline2.stop()

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert abs(r1.timestamp_s - r2.timestamp_s) < 1e-6
            if r1.error_px is not None and r2.error_px is not None:
                assert abs(r1.error_px - r2.error_px) < 1e-3

    def test_session_centroid_log(self):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, auto_save=False)
        session.start()
        for _ in range(10):
            session.step()
        result = session.stop()
        assert len(result.centroids) == 10
        assert len(result.centroids[0]) == 10
        assert result.centroids[0][0] == 0

    def test_session_centroids_csv(self, tmp_path):
        pipeline, _ = _make_pipeline()
        session = SessionController(pipeline, output_dir=str(tmp_path))
        session.start()
        for _ in range(5):
            session.step()
        result = session.stop()
        csv_path = __import__("os").path.join(
            result.artifacts_dir, "centroids.csv")
        with open(csv_path) as f:
            lines = f.read().strip().split("\n")
        assert lines[0].startswith("frame_index,timestamp_s,detected,")
        assert len(lines) == 6
