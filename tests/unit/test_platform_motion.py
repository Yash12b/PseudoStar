"""Moving Terminal A: platform motion + strapdown mount carry.

Covers platform stepping (drift rates, static default, unknown-name
rejection), the strapdown helper math, source-level camera carry,
placement coherence (no double-move), and closed-loop tracking with
both endpoints moving.
"""

from __future__ import annotations

import pytest

from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.platform import (
    apply_platform_delta_to_camera,
    PlatformState,
    step_platform,
)
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.simulation.world_builder import build_benchmark_world


@pytest.fixture
def qapp():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _engine(profile: str = "nominal", seed: int = 42) -> SimulationEngine:
    w = build_benchmark_world(profile, seed=seed)
    eng = SimulationEngine(WorldConfig(width=2000.0, height=2000.0,
                                       random_seed=seed))
    eng.load_scenario(w)
    return eng


class TestPlatformStepping:
    def test_static_default_does_not_move(self):
        eng = _engine("nominal")
        p0 = eng.get_terminal_a_state()
        for _ in range(100):
            eng.step(1.0 / 30.0)
        p1 = eng.get_terminal_a_state()
        assert (p1.x, p1.y, p1.z) == pytest.approx((p0.x, p0.y, p0.z))
        assert p1.yaw_deg == pytest.approx(p0.yaw_deg)

    def test_drift_integrates_rates(self):
        eng = _engine("moving")
        p0 = eng.get_terminal_a_state()
        for _ in range(300):
            eng.step(1.0 / 30.0)
        p1 = eng.get_terminal_a_state()
        assert p1.x == pytest.approx(p0.x + 0.3 * 10.0)
        assert p1.yaw_deg == pytest.approx(p0.yaw_deg + 0.2 * 10.0)
        assert p1.vx == pytest.approx(0.3)
        assert p1.yaw_rate_deg_s == pytest.approx(0.2)

    def test_unknown_motion_rejected(self):
        from fsoc_tracker.simulation.world_builder import (
            create_world,
            set_terminal_motion,
        )
        cfg = create_world()
        with pytest.raises(ValueError):
            set_terminal_motion(cfg, "warp")
        cfg2 = create_world()
        set_terminal_motion(cfg2, "drift", vx=1.0)
        eng = SimulationEngine(WorldConfig())
        eng.load_scenario(cfg2)
        eng.step(1.0)
        assert eng.get_terminal_a_state().x == pytest.approx(
            cfg2.terminal.x + 1.0)

    def test_step_platform_pure(self):
        p = PlatformState(x=1.0, yaw_deg=10.0)
        step_platform(p, "static", {}, 1.0)
        assert (p.x, p.yaw_deg) == (1.0, 10.0)
        step_platform(p, "drift", {"vx": 2.0, "yaw_rate_deg_s": 5.0}, 0.5)
        assert p.x == pytest.approx(2.0)
        assert p.yaw_deg == pytest.approx(12.5)
        with pytest.raises(ValueError):
            step_platform(p, "warp", {}, 1.0)


class TestStrapdownCarry:
    def test_delta_math(self):
        from types import SimpleNamespace
        cam = SimpleNamespace(position_x=100.0, position_y=100.0,
                              position_z=50.0, pan_deg=5.0, tilt_deg=1.0,
                              roll_deg=0.0)
        prev = PlatformState(x=100.0, y=100.0, z=50.0,
                             yaw_deg=5.0, pitch_deg=1.0)
        cur = PlatformState(x=103.0, y=99.0, z=50.0,
                            yaw_deg=5.5, pitch_deg=0.8, roll_deg=0.2)
        apply_platform_delta_to_camera(cam, prev, cur)
        assert cam.position_x == pytest.approx(103.0)
        assert cam.position_y == pytest.approx(99.0)
        assert cam.pan_deg == pytest.approx(5.5)
        assert cam.tilt_deg == pytest.approx(0.8)
        assert cam.roll_deg == pytest.approx(0.2)

    def test_junk_pose_raises_instead_of_poisoning(self):
        from types import SimpleNamespace
        cam = SimpleNamespace(position_x=100.0, position_y=100.0,
                              position_z=50.0, pan_deg=5.0, tilt_deg=1.0,
                              roll_deg=0.0)
        junk = SimpleNamespace(x="nope", y=0.0, z=0.0,
                               yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
        good = PlatformState()
        with pytest.raises(TypeError):
            apply_platform_delta_to_camera(cam, good, junk)
        with pytest.raises(TypeError):
            apply_platform_delta_to_camera(cam, junk, good)
        # Camera untouched: no junk arithmetic leaked in.
        assert cam.position_x == pytest.approx(100.0)
        assert cam.pan_deg == pytest.approx(5.0)

    def test_source_carries_camera_with_platform(self):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import (
            VirtualSensorRenderer,
        )
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        eng = _engine("moving")
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            position_x=1000.0, position_y=1000.0, position_z=50.0))
        src = VirtualSimulationSource(
            eng, cam, VirtualSensorRenderer(SensorConfig()),
            None, 1.0 / 30.0)
        src.open()
        for _ in range(300):
            assert src.read() is not None
        plat = eng.get_terminal_a_state()
        assert cam.state.position_x == pytest.approx(plat.x)
        assert cam.state.position_y == pytest.approx(plat.y)
        # Platform yawed 2 deg with no gimbal commands: boresight rode along.
        assert cam.state.pan_deg == pytest.approx(plat.yaw_deg)
        src.release()


class TestPlacementCoherence:
    def test_placement_moves_platform_without_double_carry(
            self, qapp) -> None:
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "simulation",
            "trajectory": "straight_line",
            "camera_width": 64,
            "camera_height": 48,
        })
        worker._init_pipeline()
        assert worker.set_terminal_pose(500.0, 1000.0, 1500.0) is True
        pose = worker._sim_engine.get_terminal_pose()
        assert pose[:3] == pytest.approx((500.0, 1000.0, 1500.0))
        before = (worker._camera.state.position_x,
                  worker._camera.state.position_y,
                  worker._camera.state.position_z)
        worker._source.read()  # static platform: carry must be zero
        after = (worker._camera.state.position_x,
                 worker._camera.state.position_y,
                 worker._camera.state.position_z)
        assert after == pytest.approx(before)
        worker._release_resources()


class TestClosedLoopMoving:
    def test_tracking_holds_with_both_endpoints_moving(self):
        from fsoc_tracker.control.controller import (
            CameraActuator,
            CoarsePointingController,
        )
        from fsoc_tracker.perception.classical_engine import (
            ClassicalBeaconDetector,
        )
        from fsoc_tracker.perception.config import PerceptionConfig
        from fsoc_tracker.pipeline.eval import EvalSink
        from fsoc_tracker.pipeline.pipeline import TrackingPipeline
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import (
            VirtualSensorRenderer,
        )
        from fsoc_tracker.tracking.config import TrackerConfig
        from fsoc_tracker.tracking.tracker import KalmanTracker
        eng = _engine("moving")
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            position_x=1000.0, position_y=1000.0, position_z=50.0))
        sink = EvalSink()
        src = VirtualSimulationSource(
            eng, cam, VirtualSensorRenderer(SensorConfig()),
            None, 1.0 / 30.0, eval_sink=sink)
        pipe = TrackingPipeline(
            perception=ClassicalBeaconDetector(PerceptionConfig(
                threshold_mode="global", threshold_value=20.0)),
            tracker=KalmanTracker(TrackerConfig()),
            controller=CoarsePointingController(),
            actuator=CameraActuator())
        pipe.set_source(src)
        pipe.start()
        errs: list[float] = []
        for _ in range(200):
            r = pipe.step()
            gt = sink.primary_for(r.frame_index)
            if gt is not None and getattr(gt, "target_visible", False):
                errs.append(abs(r.tracking.estimated_x - gt.target_pixel_x)
                            + abs(r.tracking.estimated_y - gt.target_pixel_y))
        pipe.stop()
        import math
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        assert rmse < 10.0  # PS target holds with Terminal A moving
        assert pipe.state.frame_count == 200
