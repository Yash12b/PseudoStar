"""Tests for the generic configurable world (no scene presets)."""
import math

import pytest

from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.simulation.world_builder import (
    add_beacon,
    available_motions,
    build_benchmark_world,
    clear_disturbance,
    clear_primary_beacon,
    create_world,
    get_terminal_a_state,
    place_terminal_a,
    randomize_world,
    remove_beacon,
    reset_world,
    restart_simulation,
    set_manual_control,
    set_primary_beacon,
    set_terminal_a_orientation,
)


class TestCreateWorld:
    def test_empty_world_no_beacons(self):
        cfg = create_world()
        assert cfg.beacons == []
        assert cfg.primary_beacon_id is None

    def test_terminal_defaults(self):
        cfg = create_world()
        st = get_terminal_a_state(cfg)
        assert st["x"] == pytest.approx(1000.0)
        assert st["yaw_deg"] == pytest.approx(0.0)


class TestTerminalPlacement:
    def test_place_arbitrary(self):
        cfg = create_world()
        place_terminal_a(cfg, 100.0, 200.0, 300.0)
        assert get_terminal_a_state(cfg)["x"] == pytest.approx(100.0)

    def test_place_clamps_to_bounds(self):
        cfg = create_world()
        place_terminal_a(cfg, -50.0, 9999.0, 10.0)
        st = get_terminal_a_state(cfg)
        assert st["x"] == pytest.approx(0.0)
        assert st["y"] == pytest.approx(2000.0)

    def test_orientation_independent(self):
        cfg = create_world()
        set_terminal_a_orientation(cfg, 25.0, -5.0)
        st = get_terminal_a_state(cfg)
        assert st["yaw_deg"] == pytest.approx(25.0)
        place_terminal_a(cfg, 10.0, 10.0, 10.0)
        # Placing never re-aims.
        assert get_terminal_a_state(cfg)["yaw_deg"] == pytest.approx(25.0)

    def test_engine_applies_pose(self):
        cfg = create_world()
        place_terminal_a(cfg, 111.0, 222.0, 333.0)
        set_terminal_a_orientation(cfg, 12.0, -3.0)
        engine = SimulationEngine(WorldConfig())
        engine.load_scenario(cfg)
        assert engine.get_terminal_pose() == pytest.approx(
            (111.0, 222.0, 333.0, 12.0, -3.0))


class TestBeaconCreation:
    def test_ids_auto_increment_never_reused(self):
        cfg = create_world()
        a = add_beacon(cfg, 0, 0, 100, motion="straight_line")
        b = add_beacon(cfg, 1, 1, 100, motion="circular")
        assert (a.beacon_id, b.beacon_id) == (0, 1)
        assert remove_beacon(cfg, 0) is True
        c = add_beacon(cfg, 2, 2, 100, motion="random")
        assert c.beacon_id == 2

    def test_remove_missing(self):
        assert remove_beacon(create_world(), 99) is False

    def test_engine_remove(self):
        engine = SimulationEngine(WorldConfig())
        t = engine.add_target(trajectory_type="straight_line",
                              trajectory_params={"x0": 0, "y0": 0, "z0": 100,
                                                 "vx": 1.0, "vy": 0.0})
        assert engine.remove_beacon(t.target_id) is True
        assert engine.remove_beacon(t.target_id) is False

    def test_unknown_motion_rejected(self):
        with pytest.raises(ValueError):
            add_beacon(create_world(), 0, 0, 100, motion="nope")

    def test_required_motions_available(self):
        for m in ("straight_line", "circular", "figure_8", "random"):
            assert m in available_motions()


class TestIndependentMotion:
    def _engine_two(self):
        cfg = create_world()
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 1.0, "vy": 0.0},
                   seed=1)
        add_beacon(cfg, 900.0, 900.0, 500.0, motion="circular",
                   motion_params={"cx": 900.0, "cy": 900.0, "cz": 500.0,
                                  "radius": 5.0, "angular_speed_rad_s": 0.5},
                   seed=2)
        engine = SimulationEngine(WorldConfig())
        engine.load_scenario(cfg)
        return engine

    def test_different_models_advance(self):
        engine = self._engine_two()
        engine.step(1.0)
        s = engine.get_state()
        assert s.targets[0].x == pytest.approx(1001.0)
        assert (s.targets[1].x, s.targets[1].y) != pytest.approx((900.0, 900.0))

    def test_moving_a_leaves_b(self):
        engine = SimulationEngine(WorldConfig())
        a = engine.add_target(
            trajectory_type="user_controlled",
            trajectory_params={"x0": 500.0, "y0": 500.0, "z0": 500.0})
        b = engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 0.0, "y0": 0.0, "z0": 100.0,
                               "vx": 0.0, "vy": 0.0})
        assert engine.nudge_target(a.target_id, 50.0, 0.0, 0.0) is True
        engine.step(0.1)
        s = engine.get_state()
        by_id = {t.target_id: t for t in s.targets}
        assert by_id[a.target_id].x == pytest.approx(550.0)
        assert by_id[b.target_id].x == pytest.approx(0.0)

    def test_seed_control(self):
        def run(seed):
            engine = SimulationEngine(WorldConfig())
            engine.add_target(
                trajectory_type="random",
                trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                   "seed": seed})
            for _ in range(30):
                engine.step(1.0 / 30.0)
            return engine.get_state().targets[0].x
        assert run(7) == pytest.approx(run(7))
        assert run(7) != pytest.approx(run(8))


class TestPrimaryBeacon:
    def test_set_replace_clear(self):
        cfg = create_world()
        add_beacon(cfg, 0, 0, 100, motion="straight_line")
        add_beacon(cfg, 5, 5, 100, motion="circular")
        assert set_primary_beacon(cfg, 1) is True
        assert cfg.primary_beacon_id == 1
        assert set_primary_beacon(cfg, 0) is True
        assert cfg.primary_beacon_id == 0
        assert sum(b.is_primary for b in cfg.beacons) == 1
        clear_primary_beacon(cfg)
        assert cfg.primary_beacon_id is None

    def test_set_missing(self):
        assert set_primary_beacon(create_world(), 3) is False

    def test_engine_primary_no_camera_move(self):
        engine = SimulationEngine(WorldConfig())
        a = engine.add_target(trajectory_type="straight_line",
                              trajectory_params={"x0": 0, "y0": 0, "z0": 100,
                                                 "vx": 1.0, "vy": 0.0})
        b = engine.add_target(trajectory_type="circular",
                              trajectory_params={"cx": 50, "cy": 50, "cz": 100,
                                                 "radius": 2.0,
                                                 "angular_speed_rad_s": 0.3})
        assert engine.set_primary_beacon(b.target_id) is True
        assert engine.primary_beacon_id == b.target_id
        assert engine.set_primary_beacon(a.target_id) is True
        assert engine.primary_beacon_id == a.target_id
        assert engine.clear_primary_beacon() is None
        assert engine.primary_beacon_id is None

    def test_load_scenario_applies_primary(self):
        cfg = create_world()
        add_beacon(cfg, 0, 0, 100, motion="straight_line", is_primary=True)
        add_beacon(cfg, 9, 9, 100, motion="circular")
        assert cfg.primary_beacon_id == 0
        engine = SimulationEngine(WorldConfig())
        engine.load_scenario(cfg)
        assert engine.primary_beacon_id == 0


class TestManualControl:
    def test_switch_keeps_position(self):
        cfg = create_world()
        b = add_beacon(cfg, 111.0, 222.0, 333.0, motion="straight_line",
                       motion_params={"vx": 1.0, "vy": 0.0})
        assert set_manual_control(cfg, b.beacon_id) is True
        assert cfg.get_beacon(b.beacon_id).trajectory == "user_controlled"
        p = cfg.get_beacon(b.beacon_id).trajectory_params
        assert (p["x0"], p["y0"], p["z0"]) == (111.0, 222.0, 333.0)


class TestRandomizeWorld:
    def test_deterministic(self):
        a = randomize_world(master_seed=12345, n_beacons=5)
        b = randomize_world(master_seed=12345, n_beacons=5)
        assert [(t.x0, t.y0, t.z0) for t in a.beacons] == \
               [(t.x0, t.y0, t.z0) for t in b.beacons]
        assert a.primary_beacon_id == b.primary_beacon_id

    def test_seed_varies(self):
        a = randomize_world(master_seed=1, n_beacons=5)
        b = randomize_world(master_seed=2, n_beacons=5)
        assert [(t.x0, t.y0, t.z0) for t in a.beacons] != \
               [(t.x0, t.y0, t.z0) for t in b.beacons]

    def test_separations(self):
        cfg = randomize_world(master_seed=9, n_beacons=6,
                              min_beacon_separation=30.0,
                              min_terminal_beacon_separation=100.0)
        pts = [(t.x0, t.y0, t.z0) for t in cfg.beacons]
        t = cfg.terminal
        for p in pts:
            assert math.dist(p, (t.x, t.y, t.z)) >= 100.0 - 1e-6
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert math.dist(pts[i], pts[j]) >= 30.0 - 1e-6

    def test_primary_not_centered(self):
        cfg = randomize_world(master_seed=11, n_beacons=4,
                              difficulty="hard")
        t = cfg.terminal
        p = cfg.get_beacon(cfg.primary_beacon_id)
        yaw_off = abs(math.degrees(math.atan2(p.x0 - t.x, p.z0 - t.z))
                      - t.yaw_deg)
        assert yaw_off >= 2.5


class TestResetRestart:
    def test_restart_reloads_deterministically(self):
        cfg = randomize_world(master_seed=5, n_beacons=3)
        engine = SimulationEngine(WorldConfig())
        restart_simulation(engine, cfg)
        for _ in range(10):
            engine.step(1.0 / 30.0)
        first = [(t.x, t.y, t.z) for t in engine.get_state().targets]
        restart_simulation(engine, cfg)
        for _ in range(10):
            engine.step(1.0 / 30.0)
        second = [(t.x, t.y, t.z) for t in engine.get_state().targets]
        assert first == second

    def test_reset_world_copy(self):
        cfg = create_world()
        add_beacon(cfg, 0, 0, 100, motion="straight_line")
        c2 = reset_world(cfg)
        assert c2 is not cfg
        assert len(c2.beacons) == 1


class TestBenchmarkWorlds:
    def test_profiles_deterministic(self):
        for profile in ("nominal", "multi", "distractor", "coded",
                        "moving", "loss", "noise", "fog", "jitter"):
            a = build_benchmark_world(profile, seed=42)
            b = build_benchmark_world(profile, seed=42)
            assert [(t.x0, t.y0, t.z0, t.trajectory)
                    for t in a.beacons] == \
                   [(t.x0, t.y0, t.z0, t.trajectory)
                    for t in b.beacons]

    def test_unknown_profile(self):
        import pytest as _p
        with _p.raises(ValueError):
            build_benchmark_world("nope", seed=1)

    def test_disturbance_config(self):
        from fsoc_tracker.simulation.world_builder import set_disturbance
        cfg = create_world()
        set_disturbance(cfg, preset="fog", jitter_px=5.0, noise_sigma=3.0)
        assert cfg.disturbances.enabled is True
        assert cfg.disturbances.preset == "fog"
        clear_disturbance(cfg)
        assert cfg.disturbances.enabled is False
