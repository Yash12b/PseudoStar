"""Scenario configuration tests.

Proves:
1. Multiple beacons have independent motion
2. Changing one trajectory does not alter others
3. Random seeds produce different trajectories
4. Primary beacon selection does not inject ground truth into perception
5. Terminal A can start misaligned
6. All 11 trajectory types are registered and functional
7. ScenarioConfig correctly auto-assigns IDs and primary beacon
"""

from __future__ import annotations

import math

import numpy as np

from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.scenario import BeaconConfig, ScenarioConfig, TerminalConfig
from fsoc_tracker.simulation.trajectory.registry import TrajectoryRegistry
from fsoc_tracker.simulation.world import WorldConfig

# ---------------------------------------------------------------------------
# 1. Multiple beacons have independent motion
# ---------------------------------------------------------------------------

class TestBeaconIndependence:
    """Each beacon moves according to its own trajectory."""

    def test_two_beacons_move_differently(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500,
                             trajectory="straight_line",
                             trajectory_params={"vx": 100, "vy": 0, "vz": 0}),
                BeaconConfig(x0=1000, y0=1000, z0=500,
                             trajectory="circular",
                             trajectory_params={"radius": 200, "angular_speed_rad_s": 0.3}),
            ],
        )
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)

        engine.step(1.0)
        state = engine.get_state()
        targets = state.get_active_targets()

        assert len(targets) == 2
        # Straight-line beacon moved in X
        t0 = targets[0]
        assert t0.x != 1000.0, "Straight-line beacon should have moved"
        # Circular beacon moved in a circle (x and y both change)
        t1 = targets[1]
        assert t1.x != 1000.0 or t1.y != 1000.0, "Circular beacon should have moved"

    def test_five_beacons_all_independent(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="straight_line",
                             trajectory_params={"vx": 50, "vy": 0, "vz": 0}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="circular",
                             trajectory_params={"radius": 100, "angular_speed_rad_s": 0.5}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="figure_8",
                             trajectory_params={"amplitude_x": 100, "amplitude_y": 50,
                                                "angular_speed_rad_s": 0.3}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="sinusoidal",
                             trajectory_params={"amplitude_x": 80, "freq_x": 0.2}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="spiral",
                             trajectory_params={"radius_start": 50, "angular_speed_rad_s": 0.4}),
            ],
        )
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)

        engine.step(2.0)
        state = engine.get_state()
        targets = state.get_active_targets()

        positions = [(t.x, t.y, t.z) for t in targets]
        # All 5 should be at different positions
        unique = set(positions)
        assert len(unique) == 5, f"Expected 5 unique positions, got {len(unique)}: {positions}"


# ---------------------------------------------------------------------------
# 2. Changing one trajectory does not alter others
# ---------------------------------------------------------------------------

class TestTrajectoryIsolation:
    """Modifying one beacon's trajectory doesn't affect others."""

    def test_changing_seed_of_one_beacon(self) -> None:
        base = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             trajectory_params={"seed": 10, "max_step": 50}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             trajectory_params={"seed": 20, "max_step": 50}),
            ],
        )
        engine1 = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine1.load_scenario(base)
        engine1.step(3.0)
        state1 = engine1.get_state()
        pos1_b0 = state1.targets[0].x
        pos1_b1 = state1.targets[1].x

        # Change only beacon 0's seed
        modified = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             trajectory_params={"seed": 99, "max_step": 50}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             trajectory_params={"seed": 20, "max_step": 50}),
            ],
        )
        engine2 = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine2.load_scenario(modified)
        engine2.step(3.0)
        state2 = engine2.get_state()
        pos2_b0 = state2.targets[0].x
        pos2_b1 = state2.targets[1].x

        # Beacon 0 moved differently
        assert pos1_b0 != pos2_b0, "Beacon 0 should differ with different seed"
        # Beacon 1 is identical
        assert pos1_b1 == pos2_b1, "Beacon 1 should be unchanged"


# ---------------------------------------------------------------------------
# 3. Random seeds produce different trajectories
# ---------------------------------------------------------------------------

class TestSeedIsolation:
    """Different seeds produce different trajectories; same seed produces same."""

    def test_same_seed_same_trajectory(self) -> None:
        def run_with_seed(seed: int) -> tuple[float, float]:
            engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
            engine.add_target(
                trajectory_type="random",
                trajectory_params={"x0": 1000, "y0": 1000, "z0": 500,
                                   "max_step": 100, "seed": seed},
            )
            engine.step(2.0)
            state = engine.get_state()
            return (state.targets[0].x, state.targets[0].y)

        a1 = run_with_seed(42)
        a2 = run_with_seed(42)
        assert a1 == a2, "Same seed must produce same trajectory"

    def test_different_seeds_different_trajectory(self) -> None:
        def run_with_seed(seed: int) -> tuple[float, float]:
            engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
            engine.add_target(
                trajectory_type="random",
                trajectory_params={"x0": 1000, "y0": 1000, "z0": 500,
                                   "max_step": 100, "seed": seed},
            )
            engine.step(2.0)
            state = engine.get_state()
            return (state.targets[0].x, state.targets[0].y)

        a1 = run_with_seed(1)
        a2 = run_with_seed(2)
        assert a1 != a2, "Different seeds must produce different trajectories"

    def test_beacon_seeds_independent_in_scenario(self) -> None:
        """Two beacons with different seeds in the same scenario move independently."""
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             seed=1, trajectory_params={"max_step": 80}),
                BeaconConfig(x0=1000, y0=1000, z0=500, trajectory="random",
                             seed=2, trajectory_params={"max_step": 80}),
            ],
        )
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)
        engine.step(3.0)
        state = engine.get_state()

        t0 = state.targets[0]
        t1 = state.targets[1]
        assert (t0.x, t0.y) != (t1.x, t1.y), "Different seeds must produce different positions"


# ---------------------------------------------------------------------------
# 4. Primary beacon selection does NOT inject ground truth into perception
# ---------------------------------------------------------------------------

class TestPrimaryBeaconNoGTInjection:
    """Primary beacon is a mission concept, not a perception input."""

    def test_primary_beacon_is_only_metadata(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(x0=1000, y0=1000, z0=500, is_primary=True,
                             trajectory="straight_line",
                             trajectory_params={"vx": 50, "vy": 0, "vz": 0}),
                BeaconConfig(x0=1200, y0=800, z0=400, is_primary=False,
                             trajectory="circular",
                             trajectory_params={"radius": 100, "angular_speed_rad_s": 0.2}),
            ],
        )
        assert scenario.primary_beacon_id == 0

        # Engine loads the scenario — no GT is exposed
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)

        # get_state() returns WorldTargetState — no "is_primary" field
        state = engine.get_state()
        for t in state.targets:
            assert not hasattr(t, "is_primary"), "WorldTargetState must not expose is_primary"

    def test_perception_cannot_access_world_position(self) -> None:
        """The detector and tracker consume image data, not world coords."""
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.tracking.tracker import KalmanTracker

        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()

        # Create a synthetic image with a bright spot
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[230:250, 310:330] = (255, 255, 255)

        # Detector works on image only
        det_result = detector.detect(image, timestamp_s=0.0, frame_index=0)
        assert det_result.detected

        # Tracker works on detections only
        trk_state = tracker.update(det_result.detections, timestamp_s=0.0)
        assert trk_state is not None

        # Neither has access to world coordinates
        assert not hasattr(det_result, "world_x")
        assert not hasattr(trk_state, "world_x")


# ---------------------------------------------------------------------------
# 5. Terminal A can start misaligned
# ---------------------------------------------------------------------------

class TestTerminalMisalignment:
    """Terminal A initial orientation is configurable."""

    def test_default_terminal_is_at_origin_orientation(self) -> None:
        terminal = TerminalConfig()
        assert terminal.yaw_deg == 0.0
        assert terminal.pitch_deg == 0.0

    def test_terminal_can_start_misaligned(self) -> None:
        scenario = ScenarioConfig(
            terminal=TerminalConfig(yaw_deg=15.0, pitch_deg=-5.0),
            beacons=[
                BeaconConfig(x0=1500, y0=1200, z0=800,
                             trajectory="straight_line",
                             trajectory_params={"vx": 0, "vy": 0, "vz": 0}),
            ],
        )
        assert scenario.terminal.yaw_deg == 15.0
        assert scenario.terminal.pitch_deg == -5.0

    def test_misaligned_terminal_stores_orientation(self) -> None:
        scenario = ScenarioConfig(
            terminal=TerminalConfig(yaw_deg=30.0, pitch_deg=10.0),
        )
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)
        state = engine.get_state()

        # Platform stores the terminal orientation
        assert state.platform.yaw_deg == 30.0
        assert state.platform.pitch_deg == 10.0

    def test_beacon_outside_initial_fov(self) -> None:
        """A beacon can start outside the camera's field of view."""
        # Camera at (1000,1000,50) with 4° HFOV — narrow cone
        # Beacon at (1800, 1800, 1500) is far outside
        scenario = ScenarioConfig(
            terminal=TerminalConfig(x=1000, y=1000, z=50, yaw_deg=0, pitch_deg=0),
            beacons=[
                BeaconConfig(x0=1800, y0=1800, z0=1500,
                             trajectory="straight_line",
                             trajectory_params={"vx": 0, "vy": 0, "vz": 0}),
            ],
        )
        engine = SimulationEngine(WorldConfig(width=2000, height=2000, random_seed=42))
        engine.load_scenario(scenario)
        engine.step(0.1)
        state = engine.get_state()
        t = state.targets[0]
        # Beacon stays at its position — not in FOV
        assert t.x == 1800.0


# ---------------------------------------------------------------------------
# 6. All 11 trajectory types registered and functional
# ---------------------------------------------------------------------------

class TestTrajectoryTypes:
    """All 11 trajectory types are registered and produce valid positions."""

    # (name, params) — each type gets its correct default params
    TRAJECTORY_CASES = [
        ("straight_line", {"x0": 1000, "y0": 1000, "z0": 500, "vx": 50}),
        ("circular", {"cx": 1000, "cy": 1000, "cz": 500, "radius": 200, "angular_speed_rad_s": 0.3}),
        ("figure_8", {"cx": 1000, "cy": 1000, "cz": 500, "amplitude_x": 100, "amplitude_y": 50, "angular_speed_rad_s": 0.3}),
        ("random", {"x0": 1000, "y0": 1000, "z0": 500, "max_step": 100, "seed": 42}),
        ("spiral", {"cx": 1000, "cy": 1000, "cz": 500, "radius_start": 50, "angular_speed_rad_s": 0.4}),
        ("sinusoidal", {"cx": 1000, "cy": 1000, "cz": 500, "amplitude_x": 80, "freq_x": 0.2}),
        ("random_walk", {"x0": 1000, "y0": 1000, "z0": 500, "step_size": 50, "seed": 42}),
        ("stop_go", {"x0": 1000, "y0": 1000, "z0": 500, "vx": 100}),
        ("sudden_reversal", {"x0": 1000, "y0": 1000, "z0": 500, "vx": 100}),
        ("accel_decel", {"x0": 1000, "y0": 1000, "z0": 500, "speed": 100}),
        ("user_controlled", {"x0": 1000, "y0": 1000, "z0": 500}),
    ]

    def test_all_types_registered(self) -> None:
        registry = TrajectoryRegistry()
        for name, _ in self.TRAJECTORY_CASES:
            assert name in registry.available, f"'{name}' not registered"

    def test_all_types_produce_positions(self) -> None:
        registry = TrajectoryRegistry()
        for name, params in self.TRAJECTORY_CASES:
            traj = registry.create(name, params)
            pos = traj.position(1.0)
            assert len(pos) == 3, f"'{name}' position must be (x, y, z)"
            assert all(math.isfinite(v) for v in pos), f"'{name}' position has non-finite values"

    def test_all_types_serializable(self) -> None:
        registry = TrajectoryRegistry()
        for name, params in self.TRAJECTORY_CASES:
            traj = registry.create(name, params)
            d = traj.to_dict()
            assert d["type"] == name
            traj2 = registry.from_dict(d)
            pos1 = traj.position(1.0)
            pos2 = traj2.position(1.0)
            assert pos1 == pos2, f"'{name}' round-trip serialization failed"


# ---------------------------------------------------------------------------
# 7. ScenarioConfig auto-assignment
# ---------------------------------------------------------------------------

class TestScenarioConfig:
    """ScenarioConfig correctly auto-assigns IDs and primary beacon."""

    def test_auto_assigns_beacon_ids(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(trajectory="straight_line"),
                BeaconConfig(trajectory="circular"),
                BeaconConfig(trajectory="figure_8"),
            ],
        )
        ids = [b.beacon_id for b in scenario.beacons]
        assert ids == [0, 1, 2]

    def test_auto_selects_primary_beacon(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(trajectory="straight_line"),
                BeaconConfig(trajectory="circular", is_primary=True),
            ],
        )
        assert scenario.primary_beacon_id == 1
        assert scenario.primary_beacon.is_primary is True

    def test_add_beacon_method(self) -> None:
        scenario = ScenarioConfig()
        b = scenario.add_beacon(x0=500, y0=600, z0=700, trajectory="random", seed=5)
        assert b.beacon_id == 0
        assert b.x0 == 500
        assert len(scenario.beacons) == 1

    def test_get_beacon_by_id(self) -> None:
        scenario = ScenarioConfig(
            beacons=[
                BeaconConfig(trajectory="straight_line"),
                BeaconConfig(trajectory="circular"),
            ],
        )
        b = scenario.get_beacon(1)
        assert b is not None
        assert b.trajectory == "circular"
        assert scenario.get_beacon(99) is None
