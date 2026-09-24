"""Tests for the simulation engine, world, and integration."""

from __future__ import annotations

import pytest

from fsoc_tracker.core.exceptions import SimulationError
from fsoc_tracker.simulation.boundaries import BoundaryMode, apply_boundary, apply_boundary_3d
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.target import WorldTargetState
from fsoc_tracker.simulation.world import WorldConfig, WorldState

# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestBoundary:
    def test_clamp_inside(self) -> None:
        pos, vel = apply_boundary(5.0, 10.0, 0.0, 10.0, BoundaryMode.CLAMP)
        assert pos == 5.0
        assert vel == 10.0

    def test_clamp_below(self) -> None:
        pos, vel = apply_boundary(-1.0, 10.0, 0.0, 10.0, BoundaryMode.CLAMP)
        assert pos == 0.0
        assert vel == 0.0

    def test_clamp_above(self) -> None:
        pos, vel = apply_boundary(11.0, 10.0, 0.0, 10.0, BoundaryMode.CLAMP)
        assert pos == 10.0
        assert vel == 0.0

    def test_reflect_below(self) -> None:
        pos, vel = apply_boundary(-1.0, 5.0, 0.0, 10.0, BoundaryMode.REFLECT)
        assert pos == 1.0
        assert vel == -5.0

    def test_reflect_above(self) -> None:
        pos, vel = apply_boundary(11.0, 5.0, 0.0, 10.0, BoundaryMode.REFLECT)
        assert pos == 9.0
        assert vel == -5.0

    def test_wrap(self) -> None:
        pos, vel = apply_boundary(11.0, 5.0, 0.0, 10.0, BoundaryMode.WRAP)
        assert pos == 1.0
        assert vel == 5.0

    def test_wrap_negative(self) -> None:
        pos, vel = apply_boundary(-1.0, 5.0, 0.0, 10.0, BoundaryMode.WRAP)
        assert pos == 9.0
        assert vel == 5.0

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            apply_boundary(5.0, 1.0, 10.0, 0.0, BoundaryMode.CLAMP)

    def test_boundary_3d(self) -> None:
        x, y, z, vx, vy, vz = apply_boundary_3d(
            -1, 5, 101, 1, 2, 3,
            0, 10, 0, 10, 0, 100,
            BoundaryMode.CLAMP,
        )
        assert x == 0.0
        assert y == 5.0
        assert z == 100.0
        assert vx == 0.0


# ---------------------------------------------------------------------------
# World / Target state tests
# ---------------------------------------------------------------------------

class TestWorldState:
    def test_defaults(self) -> None:
        ws = WorldState()
        assert ws.simulation_time_s == 0.0
        assert ws.step_count == 0
        assert len(ws.targets) == 0

    def test_get_target(self) -> None:
        ws = WorldState()
        ws.targets.append(WorldTargetState(target_id=42))
        t = ws.get_target(42)
        assert t is not None
        assert t.target_id == 42
        assert ws.get_target(99) is None

    def test_get_active_targets(self) -> None:
        ws = WorldState()
        ws.targets.append(WorldTargetState(target_id=0, active=True))
        ws.targets.append(WorldTargetState(target_id=1, active=False))
        ws.targets.append(WorldTargetState(target_id=2, active=True))
        active = ws.get_active_targets()
        assert len(active) == 2
        assert all(t.active for t in active)

    def test_serialization_roundtrip(self) -> None:
        ws = WorldState(simulation_time_s=5.0, step_count=10)
        ws.targets.append(WorldTargetState(target_id=0, x=100, y=200))
        d = ws.to_dict()
        ws2 = WorldState.from_dict(d)
        assert ws2.simulation_time_s == 5.0
        assert len(ws2.targets) == 1
        assert ws2.targets[0].x == 100.0


class TestWorldTargetState:
    def test_position_property(self) -> None:
        t = WorldTargetState(x=1, y=2, z=3)
        assert t.position == (1, 2, 3)

    def test_velocity_property(self) -> None:
        t = WorldTargetState(vx=4, vy=5, vz=6)
        assert t.velocity == (4, 5, 6)

    def test_serialization_roundtrip(self) -> None:
        t = WorldTargetState(target_id=7, x=100, y=200, z=300, shape="circle")
        d = t.to_dict()
        t2 = WorldTargetState.from_dict(d)
        assert t2.target_id == 7
        assert t2.x == 100.0
        assert t2.shape == "circle"


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    def test_add_target_default(self) -> None:
        engine = SimulationEngine()
        target = engine.add_target()
        assert target.target_id == 0
        assert target.active

    def test_step_advances_time(self) -> None:
        engine = SimulationEngine()
        engine.add_target()
        state = engine.step(0.1)
        assert state.simulation_time_s == pytest.approx(0.1)
        assert state.step_count == 1

    def test_step_negative_dt_raises(self) -> None:
        engine = SimulationEngine()
        with pytest.raises(SimulationError):
            engine.step(-0.1)

    def test_step_zero_dt_raises(self) -> None:
        engine = SimulationEngine()
        with pytest.raises(SimulationError):
            engine.step(0.0)

    def test_reset(self) -> None:
        engine = SimulationEngine()
        engine.add_target(trajectory_type="straight_line", trajectory_params={"vx": 10})
        engine.step(1.0)
        engine.reset()
        assert engine.time == 0.0
        assert engine.step_count == 0

    def test_trajectory_drives_position(self) -> None:
        engine = SimulationEngine()
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 0, "y0": 0, "vx": 100, "vy": 0},
        )
        state = engine.step(1.0)
        assert state.targets[0].x == pytest.approx(100.0)
        assert state.targets[0].y == pytest.approx(0.0)

    def test_multiple_targets(self) -> None:
        engine = SimulationEngine()
        engine.add_target(trajectory_type="straight_line", trajectory_params={"vx": 10})
        engine.add_target(trajectory_type="circular", trajectory_params={"cx": 500, "cy": 500, "radius": 100})
        state = engine.step(0.1)
        assert len(state.targets) == 2

    def test_boundary_enforcement(self) -> None:
        config = WorldConfig(width=100, height=100, depth=100, boundary_mode=BoundaryMode.CLAMP)
        engine = SimulationEngine(config)
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 95, "y0": 50, "vx": 200, "vy": 0},
        )
        state = engine.step(1.0)
        # Should be clamped at x_max = 100
        assert state.targets[0].x <= 100.0

    def test_state_serialization(self) -> None:
        engine = SimulationEngine()
        engine.add_target(trajectory_type="straight_line", trajectory_params={"vx": 10})
        engine.step(0.5)
        state = engine.get_state()
        d = state.to_dict()
        assert "targets" in d
        assert "simulation_time_s" in d
        assert d["simulation_time_s"] == pytest.approx(0.5)

    def test_update_to(self) -> None:
        engine = SimulationEngine()
        engine.add_target(trajectory_type="straight_line", trajectory_params={"x0": 0, "vx": 100})
        state = engine.update_to(2.0)
        assert state.simulation_time_s == pytest.approx(2.0)
        assert state.targets[0].x == pytest.approx(200.0)

    def test_update_to_rewind(self) -> None:
        engine = SimulationEngine()
        engine.add_target(trajectory_type="straight_line", trajectory_params={"x0": 0, "vx": 100})
        engine.step(5.0)
        state = engine.update_to(2.0)
        assert state.simulation_time_s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# FPS independence tests
# ---------------------------------------------------------------------------

class TestFPSIndependence:
    """Prove that target position after equal simulation time is
    independent of the dt used (for analytical trajectories)."""

    @pytest.mark.parametrize(
        "trajectory_type,params",
        [
            ("straight_line", {"x0": 0, "y0": 0, "vx": 100, "vy": 50}),
            ("circular", {"cx": 1000, "cy": 1000, "radius": 200, "angular_speed_rad_s": 1.0}),
            ("figure_8", {"cx": 1000, "cy": 1000, "amplitude_x": 100, "amplitude_y": 80}),
        ],
    )
    def test_fps_independence(self, trajectory_type: str, params: dict) -> None:
        sim_time = 3.0
        results: list[tuple[float, float]] = []

        for dt in [1.0 / 15, 1.0 / 30, 1.0 / 60, 1.0 / 120]:
            engine = SimulationEngine()
            engine.add_target(trajectory_type=trajectory_type, trajectory_params=params)
            steps = int(sim_time / dt)
            for _ in range(steps):
                state = engine.step(dt)
            results.append((state.targets[0].x, state.targets[0].y))

        # All results should be very close (analytical trajectories)
        ref_x, ref_y = results[0]
        for rx, ry in results[1:]:
            assert rx == pytest.approx(ref_x, abs=0.5), f"X deviates: {rx} vs {ref_x}"
            assert ry == pytest.approx(ref_y, abs=0.5), f"Y deviates: {ry} vs {ref_y}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        results1: list[float] = []
        results2: list[float] = []

        engine1 = SimulationEngine()
        engine1.add_target(
            trajectory_type="random",
            trajectory_params={"seed": 42, "num_waypoints": 20},
        )
        for _ in range(100):
            state = engine1.step(0.1)
            results1.append(state.targets[0].x)

        engine2 = SimulationEngine()
        engine2.add_target(
            trajectory_type="random",
            trajectory_params={"seed": 42, "num_waypoints": 20},
        )
        for _ in range(100):
            state = engine2.step(0.1)
            results2.append(state.targets[0].x)

        assert results1 == results2

    def test_circular_reproducibility(self) -> None:
        engine1 = SimulationEngine()
        engine1.add_target(trajectory_type="circular", trajectory_params={"radius": 200})
        engine2 = SimulationEngine()
        engine2.add_target(trajectory_type="circular", trajectory_params={"radius": 200})

        for _ in range(50):
            s1 = engine1.step(0.1)
            s2 = engine2.step(0.1)
            assert s1.targets[0].x == pytest.approx(s2.targets[0].x, rel=1e-12)
            assert s1.targets[0].y == pytest.approx(s2.targets[0].y, rel=1e-12)


# ---------------------------------------------------------------------------
# Engine reset + replay
# ---------------------------------------------------------------------------

class TestResetReplay:
    def test_reset_replay_identical(self) -> None:
        engine = SimulationEngine()
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 100, "y0": 200, "vx": 50, "vy": 30},
        )
        for _ in range(20):
            state1 = engine.step(0.1)

        engine.reset()
        for _ in range(20):
            state2 = engine.step(0.1)

        assert state1.targets[0].x == pytest.approx(state2.targets[0].x, rel=1e-12)
        assert state1.targets[0].y == pytest.approx(state2.targets[0].y, rel=1e-12)

    def test_step_count_resets(self) -> None:
        engine = SimulationEngine()
        engine.add_target()
        engine.step(0.1)
        engine.step(0.1)
        engine.reset()
        assert engine.step_count == 0
        assert engine.time == 0.0


class TestUserControlledNudge:
    def test_nudge_moves_beacon(self) -> None:
        engine = SimulationEngine()
        t = engine.add_target(
            trajectory_type="user_controlled",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0},
        )
        assert engine.nudge_target(t.target_id, 10.0, 0.0, -5.0) is True
        state = engine.step(1.0 / 30.0)
        assert state.targets[0].x == pytest.approx(1010.0)
        assert state.targets[0].z == pytest.approx(495.0)

    def test_nudge_rejected_for_scripted(self) -> None:
        engine = SimulationEngine()
        t = engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 0.0, "y0": 0.0, "z0": 100.0, "vx": 1.0, "vy": 0.0},
        )
        assert engine.nudge_target(t.target_id, 5.0, 5.0, 5.0) is False

    def test_nudge_missing_target(self) -> None:
        engine = SimulationEngine()
        assert engine.nudge_target(99, 1.0, 1.0, 1.0) is False

    def test_reset_returns_to_spawn(self) -> None:
        engine = SimulationEngine()
        t = engine.add_target(
            trajectory_type="user_controlled",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0},
        )
        engine.nudge_target(t.target_id, 25.0, 0.0, 10.0)
        assert engine.reset_target(t.target_id) is True
        state = engine.step(1.0 / 30.0)
        assert state.targets[0].x == pytest.approx(1000.0)
        assert state.targets[0].z == pytest.approx(500.0)


class TestBeaconReconfigure:
    """Live per-beacon editing preserves all fields through snapshots."""

    def test_reconfigure_all_fields(self):
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.world import WorldConfig
        e = SimulationEngine(WorldConfig())
        t = e.add_target(trajectory_type="straight_line",
                         trajectory_params={"x0": 1000.0, "y0": 1000.0,
                                            "z0": 500.0, "vx": 0.1, "vy": 0.1})
        assert e.reconfigure_beacon(
            t.target_id, trajectory_type="circular",
            trajectory_params={"cx": 1000.0, "cy": 1000.0, "cz": 500.0,
                               "radius": 5.0, "angular_speed_rad_s": 0.3},
            size_px=15.0, brightness=1.5, shape="circular", seed=7) is True
        got = [x for x in e.get_state().targets
               if x.target_id == t.target_id][0]
        assert got.trajectory_type == "circular"
        assert got.size_px == 15.0
        assert got.brightness == 1.5
        assert got.shape == "circular"
        assert got.seed == 7

    def test_reconfigure_unknown_id(self):
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.world import WorldConfig
        e = SimulationEngine(WorldConfig())
        assert e.reconfigure_beacon(999, size_px=12.0) is False

    def test_renderer_honors_per_target_size(self):
        import numpy as np

        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.target import WorldTargetState
        cam = VirtualCamera(CameraState(position_x=1000.0, position_y=1000.0,
                                        position_z=50.0))
        sensor = VirtualSensorRenderer(SensorConfig())
        mk = lambda sz: WorldTargetState(target_id=0, x=1000.0, y=1000.0,
                                         z=600.0, brightness=1.0, size_px=sz)
        a = sensor.render(cam, [mk(20.0)], 0.0, 0).image
        b = sensor.render(cam, [mk(10.0)], 0.0, 0).image
        c = sensor.render(cam, [mk(None)], 0.0, 0).image
        na = int(np.sum(a > 100))
        nb = int(np.sum(b > 100))
        nc = int(np.sum(c > 100))
        assert na > nb > 0
        # Default (None) matches legacy default-size rendering
        d = sensor.render(cam, [mk(10.0)], 0.0, 0).image
        assert nc == int(np.sum(d > 100))
