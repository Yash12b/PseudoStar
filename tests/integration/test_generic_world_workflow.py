"""Flagship acceptance: generic configurable world workflow (§19).

CREATE WORLD -> place Terminal A -> add beacons w/ independent motion ->
primary (+manual option) -> disturbances -> start -> camera POV retains
orientation (no teleport) -> beacons advance independently -> coherent
state. Uses only generic world APIs; no scene presets exist anymore.
"""
import math

import pytest

from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.simulation.world_builder import (
    add_beacon,
    create_world,
    move_terminal_a,
    place_terminal_a,
    randomize_world,
    remove_beacon,
    restart_simulation,
    set_disturbance,
    set_primary_beacon,
    set_terminal_a_orientation,
)


@pytest.fixture
def mission_world():
    cfg = create_world(seed=7, name="Acceptance")
    place_terminal_a(cfg, 1000.0, 1000.0, 50.0)
    set_terminal_a_orientation(cfg, 5.0, 0.0)  # deliberately off-target
    add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
               motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                              "vx": 0.5, "vy": 0.1},
               seed=11, is_primary=True)
    add_beacon(cfg, 900.0, 950.0, 480.0, motion="circular",
               motion_params={"cx": 900.0, "cy": 950.0, "cz": 480.0,
                              "radius": 6.0, "angular_speed_rad_s": 0.4},
               seed=12)
    add_beacon(cfg, 1100.0, 1050.0, 520.0, motion="figure_8",
               motion_params={"cx": 1100.0, "cy": 1050.0, "cz": 520.0,
                              "amplitude_x": 8.0, "amplitude_y": 5.0},
               seed=13)
    add_beacon(cfg, 950.0, 1020.0, 510.0, motion="random",
               motion_params={"x0": 950.0, "y0": 1020.0, "z0": 510.0,
                              "seed": 14},
               seed=14)
    add_beacon(cfg, 1050.0, 980.0, 490.0, motion="user_controlled",
               motion_params={"x0": 1050.0, "y0": 980.0, "z0": 490.0},
               seed=15)
    assert cfg.primary_beacon_id == 0
    set_disturbance(cfg, preset="light")
    return cfg


def _engine_for(cfg):
    engine = SimulationEngine(WorldConfig(random_seed=cfg.seed))
    engine.load_scenario(cfg)
    return engine


class TestWorldSetup:
    def test_terminal_pose_and_beacons(self, mission_world):
        engine = _engine_for(mission_world)
        assert engine.get_terminal_pose()[:3] == pytest.approx(
            (1000.0, 1000.0, 50.0))
        assert len(engine.get_state().targets) == 5
        assert engine.primary_beacon_id == 0

    def test_independent_motion(self, mission_world):
        engine = _engine_for(mission_world)
        before = {t.target_id: (t.x, t.y, t.z)
                  for t in engine.get_state().targets}
        engine.step(1.0)
        after = {t.target_id: (t.x, t.y, t.z)
                 for t in engine.get_state().targets}
        moved = [tid for tid in before if before[tid] != after[tid]]
        # The 4 procedural beacons advanced on their own models; the
        # user-controlled beacon correctly holds until commanded.
        assert sorted(moved) == [0, 1, 2, 3]
        assert before[4] == after[4]
        # Straight-line beacon moved exactly by its velocity.
        dx = after[0][0] - before[0][0]
        assert dx == pytest.approx(0.5)

    def test_primary_replacement_keeps_motions(self, mission_world):
        engine = _engine_for(mission_world)
        engine.step(0.5)
        snapshot = {t.target_id: (t.x, t.y, t.z)
                    for t in engine.get_state().targets}
        assert set_primary_beacon(mission_world, 2) is True
        assert mission_world.primary_beacon_id == 2
        assert engine.set_primary_beacon(2) is True
        engine.step(0.5)
        after = {t.target_id: (t.x, t.y, t.z)
                 for t in engine.get_state().targets}
        # Straight-line beacon advanced exactly by v*dt across the
        # redesignation (no teleport, no freeze).
        assert after[0][0] - snapshot[0][0] == pytest.approx(0.25)
        # Every procedural beacon kept advancing on its own model
        # (user-controlled id 4 holds until commanded).
        assert all(after[i] != snapshot[i] for i in (0, 1, 2, 3))
        assert after[4] == snapshot[4]

    def test_manual_beacon_ai_blind(self, mission_world):
        engine = _engine_for(mission_world)
        assert engine.nudge_target(4, 20.0, 0.0, 0.0) is True
        engine.step(0.1)
        by_id = {t.target_id: t
                 for t in engine.get_state().targets}
        assert by_id[4].x == pytest.approx(1070.0)
        # Neighbours untouched by the manual command.
        assert by_id[3].x != pytest.approx(1070.0)


class TestMissionStart:
    def test_camera_keeps_configured_orientation(self, mission_world):
        """No ground-truth teleport: camera starts as configured."""
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        engine = _engine_for(mission_world)
        t = mission_world.terminal
        camera = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            position_x=t.x, position_y=t.y, position_z=t.z,
            pan_deg=t.yaw_deg, tilt_deg=t.pitch_deg))
        assert camera.state.pan_deg == pytest.approx(5.0)
        # Beacon at bearing ~0 stays off the 5-degree boresight:
        # acquisition must search, never teleport.
        bearing = math.degrees(math.atan2(1000.0 - 1000.0, 500.0 - 50.0))
        assert abs(bearing - camera.state.pan_deg) > 2.0
        engine.step(1.0 / 30.0)
        assert camera.state.pan_deg == pytest.approx(5.0)

    def test_closed_loop_tracks_primary(self, mission_world):
        """Existing perception/tracking/control close the loop on the
        generic world (reuses proven components; implements nothing)."""
        from fsoc_tracker.control.controller import (
            CameraActuator,
            CoarsePointingController,
        )
        from fsoc_tracker.perception.classical_engine import (
            ClassicalBeaconDetector,
        )
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import (
            VirtualSensorRenderer,
        )
        from fsoc_tracker.tracking.tracker import KalmanTracker

        # Easy geometry for loop integration (the hard-start case is
        # covered by test_camera_keeps_configured_orientation).
        cfg = create_world(seed=7, name="Loop")
        place_terminal_a(cfg, 1000.0, 1000.0, 50.0)
        set_terminal_a_orientation(cfg, 0.0, 0.0)
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.3, "vy": 0.1},
                   seed=3, is_primary=True)
        engine = _engine_for(cfg)
        t = cfg.terminal
        camera = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            position_x=t.x, position_y=t.y, position_z=t.z,
            pan_deg=t.yaw_deg, tilt_deg=t.pitch_deg))
        sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))
        detector, tracker = ClassicalBeaconDetector(), KalmanTracker()
        controller, actuator = CoarsePointingController(), CameraActuator()
        sim_time, errors = 0.0, []
        for i in range(120):
            engine.step(1.0 / 30.0)
            rendered = sensor.render(
                camera, engine.get_state().get_active_targets(),
                sim_time, i)
            det = detector.detect(rendered.image, sim_time, i)
            dets = (det.detections if det.primary_detection
                    and det.primary_detection.detected else [])
            trk = tracker.update(dets, sim_time)
            cmd, _ = controller.compute(
                trk, camera.intrinsics, 1.0 / 30.0, sim_time)
            actuator.apply_command(camera, cmd, 1.0 / 30.0)
            gt = rendered.primary_ground_truth
            if gt is not None and gt.target_visible:
                errors.append(math.dist(
                    (trk.estimated_x, trk.estimated_y),
                    (gt.target_pixel_x, gt.target_pixel_y)))
            sim_time += 1.0 / 30.0
        assert errors, "no scored frames produced"
        assert sum(errors) / len(errors) < 30.0
        # Camera was driven by perception (moved off its 5-degree start).
        assert camera.state.pan_deg != pytest.approx(5.0)


class TestWorldOps:
    def test_remove_and_move(self):
        cfg = randomize_world(master_seed=3, n_beacons=3)
        assert len(cfg.beacons) == 3
        bid = cfg.beacons[0].beacon_id
        assert remove_beacon(cfg, bid) is True
        assert len(cfg.beacons) == 2
        other = cfg.beacons[0].beacon_id
        from fsoc_tracker.simulation.world_builder import move_beacon
        assert move_beacon(cfg, other, 10.0, 20.0, 30.0) is True
        b = cfg.get_beacon(other)
        assert (b.x0, b.y0, b.z0) == (10.0, 20.0, 30.0)

    def test_terminal_move_and_reset(self, mission_world):
        move_terminal_a(mission_world, 50.0, 0.0, 0.0)
        from fsoc_tracker.simulation.world_builder import (
            get_terminal_a_state,
        )
        assert get_terminal_a_state(mission_world)["x"] == pytest.approx(
            1050.0)
        engine = _engine_for(mission_world)
        restart_simulation(engine, mission_world)
        assert engine.time == pytest.approx(0.0)
        assert len(engine.get_state().targets) == 5
