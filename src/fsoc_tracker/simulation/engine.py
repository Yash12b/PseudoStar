"""Simulation engine.

The engine manages world state, advances time, evaluates target
trajectories, applies boundary handling, and produces world-state
snapshots.  It is decoupled from rendering.
"""

from __future__ import annotations

from typing import Any

from fsoc_tracker.core.exceptions import SimulationError
from fsoc_tracker.simulation.boundaries import apply_boundary_3d
from fsoc_tracker.simulation.platform import PlatformState
from fsoc_tracker.simulation.target import WorldTargetState
from fsoc_tracker.simulation.trajectory.base import Trajectory
from fsoc_tracker.simulation.trajectory.registry import TrajectoryRegistry
from fsoc_tracker.simulation.world import WorldConfig, WorldState


class SimulationEngine:
    """Deterministic simulation engine.

    Manages targets with trajectories, advances time, and produces
    world-state snapshots.  No rendering is performed.

    Usage::

        engine = SimulationEngine(config)
        engine.add_target(target_config)
        engine.step(0.033)  # advance by 33 ms
        state = engine.get_state()
    """

    def __init__(self, config: WorldConfig | None = None) -> None:
        self._config = config or WorldConfig()
        self._time_s: float = 0.0
        self._step_count: int = 0
        self._targets: list[_ManagedTarget] = []
        self._platform = PlatformState()
        self._platform_motion: str = "static"
        self._platform_motion_params: dict[str, Any] = {}
        self._registry = TrajectoryRegistry()
        self._primary_beacon_id: int | None = None

    @property
    def config(self) -> WorldConfig:
        return self._config

    @property
    def time(self) -> float:
        return self._time_s

    @property
    def step_count(self) -> int:
        return self._step_count

    def load_scenario(self, scenario: Any) -> None:
        """Load a ScenarioConfig into the engine.

        Sets up Terminal A position/orientation and all beacons
        with their independent trajectories.

        Ground truth is NOT exposed through this method — it exists
        only in the engine internals for offline evaluation.
        """
        from fsoc_tracker.simulation.scenario import ScenarioConfig
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError(f"Expected ScenarioConfig, got {type(scenario)}")

        self.reset()
        self._targets.clear()

        # Set platform (Terminal A) position and orientation
        self._platform.x = scenario.terminal.x
        self._platform.y = scenario.terminal.y
        self._platform.z = scenario.terminal.z
        self._platform.yaw_deg = scenario.terminal.yaw_deg
        self._platform.pitch_deg = scenario.terminal.pitch_deg
        # Platform motion (Terminal A itself moves, e.g. two
        # satellites): validated name + params, stepped every frame.
        from fsoc_tracker.simulation.platform import PLATFORM_MOTIONS
        motion = str(getattr(scenario.terminal, "platform_motion", "static"))
        if motion not in PLATFORM_MOTIONS:
            raise ValueError(f"Unknown platform motion: {motion!r}")
        self._platform_motion = motion
        self._platform_motion_params = dict(
            getattr(scenario.terminal, "platform_motion_params", {}) or {})

        # Add each beacon with its independent trajectory
        for bc in scenario.beacons:
            # Merge seed into trajectory params for seedable types
            params = dict(bc.trajectory_params)
            if bc.trajectory in ("random", "random_walk") and "seed" not in params:
                params["seed"] = bc.seed

            # Set initial position in params if not already there
            if bc.trajectory in ("straight_line", "random", "random_walk"):
                params.setdefault("x0", bc.x0)
                params.setdefault("y0", bc.y0)
                params.setdefault("z0", bc.z0)
            elif bc.trajectory in ("circular", "figure_8", "spiral", "sinusoidal"):
                params.setdefault("cx", bc.x0)
                params.setdefault("cy", bc.y0)
                params.setdefault("cz", bc.z0)

            target = WorldTargetState(
                target_id=bc.beacon_id,
                active=bc.active,
                shape=getattr(bc, "shape", "square") or "square",
                width=bc.size_px / 640.0,  # normalize to image fraction
                height=bc.size_px / 480.0,
                brightness=bc.brightness,
                code=getattr(bc, "code", "") or "",
                size_px=float(bc.size_px),
                seed=bc.seed,
                spawn_time_s=self._time_s,
            )

            self.add_target(
                target=target,
                trajectory_type=bc.trajectory,
                trajectory_params=params,
            )

        # Apply the scenario's primary designation (engine-level only;
        # selection/tracking IDs in the GUI remain separate concepts).
        self._primary_beacon_id = None
        if scenario.primary_beacon_id is not None:
            self.set_primary_beacon(scenario.primary_beacon_id)

    def add_target(
        self,
        target: WorldTargetState | None = None,
        trajectory: Trajectory | None = None,
        trajectory_type: str | None = None,
        trajectory_params: dict[str, Any] | None = None,
    ) -> WorldTargetState:
        """Add a target to the world.

        Either provide a *trajectory* instance, or specify
        *trajectory_type* + *trajectory_params* for the registry.

        If no target is provided, a default target is created.

        Returns:
            The WorldTargetState that was added.
        """
        if target is None:
            target = WorldTargetState(
                target_id=len(self._targets),
                spawn_time_s=self._time_s,
            )

        if trajectory is None and trajectory_type is not None:
            trajectory = self._registry.create(trajectory_type, trajectory_params)

        if trajectory is not None:
            target.trajectory_type = trajectory_type or ""
            target.trajectory_params = trajectory.to_dict()
            # Evaluate immediately so snapshots are coherent even
            # before the first step (no origin-blob artifacts).
            try:
                px, py, pz = trajectory.position(self._time_s)
                vx, vy, vz = trajectory.velocity(self._time_s)
                target.x, target.y, target.z = px, py, pz
                target.vx, target.vy, target.vz = vx, vy, vz
            except Exception:
                pass

        self._targets.append(
            _ManagedTarget(target=target, trajectory=trajectory)
        )
        return target

    def remove_beacon(self, target_id: int) -> bool:
        """Remove a beacon from the world by ID. Returns True if removed."""
        for i, managed in enumerate(self._targets):
            if managed.target.target_id == target_id:
                del self._targets[i]
                if self._primary_beacon_id == target_id:
                    self._primary_beacon_id = None
                return True
        return False

    def reconfigure_beacon(
        self,
        target_id: int,
        trajectory_type: str | None = None,
        trajectory_params: dict | None = None,
        size_px: float | None = None,
        brightness: float | None = None,
        shape: str | None = None,
        seed: int | None = None,
    ) -> bool:
        """Edit a live beacon in place. Returns True if applied.

        Only the supplied fields change; everything else is preserved.
        Swapping the trajectory rebuilds it from the registry and
        re-evaluates the target state at the current time.
        """
        for managed in self._targets:
            if managed.target.target_id != target_id:
                continue
            target = managed.target
            if trajectory_type is not None:
                try:
                    traj = self._registry.create(
                        trajectory_type, dict(trajectory_params or {}))
                except Exception:
                    return False
                managed.trajectory = traj
                target.trajectory_type = trajectory_type
                target.trajectory_params = dict(trajectory_params or {})
                try:
                    px, py, pz = traj.position(self._time_s)
                    vx, vy, vz = traj.velocity(self._time_s)
                    target.x, target.y, target.z = px, py, pz
                    target.vx, target.vy, target.vz = vx, vy, vz
                except Exception:
                    pass
            if size_px is not None:
                target.size_px = float(size_px)
                target.width = float(size_px) / 640.0
                target.height = float(size_px) / 480.0
            if brightness is not None:
                target.brightness = float(brightness)
            if shape is not None:
                target.shape = str(shape)
            if seed is not None:
                target.seed = int(seed)
            return True
        return False

    def set_platform_pose(
        self, x: float, y: float, z: float,
        yaw_deg: float, pitch_deg: float,
    ) -> None:
        """Set Terminal A platform position and orientation."""
        self._platform.x = float(x)
        self._platform.y = float(y)
        self._platform.z = float(z)
        self._platform.yaw_deg = float(yaw_deg)
        self._platform.pitch_deg = float(pitch_deg)

    def get_terminal_pose(self) -> tuple[float, float, float, float, float]:
        """Terminal A pose as (x, y, z, yaw_deg, pitch_deg)."""
        p = self._platform
        return (p.x, p.y, p.z, p.yaw_deg, p.pitch_deg)

    def set_terminal_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
    ) -> None:
        """Place Terminal A at an arbitrary valid location.

        Orientation is set only when explicitly given — placing the
        terminal never auto-aims it at any beacon.
        """
        cfg = self._config
        self._platform.x = min(max(float(x), cfg.x_min), cfg.x_max)
        self._platform.y = min(max(float(y), cfg.y_min), cfg.y_max)
        self._platform.z = min(max(float(z), cfg.z_min), cfg.z_max)
        if yaw_deg is not None:
            self._platform.yaw_deg = float(yaw_deg)
        if pitch_deg is not None:
            self._platform.pitch_deg = float(pitch_deg)

    def set_terminal_a_orientation(self, yaw_deg: float, pitch_deg: float) -> None:
        """Set Terminal A orientation independently of any target."""
        self._platform.yaw_deg = float(yaw_deg)
        self._platform.pitch_deg = float(pitch_deg)

    def get_terminal_a_state(self) -> PlatformState:
        """Return the Terminal A platform state (authoritative copy)."""
        import copy
        return copy.deepcopy(self._platform)

    @property
    def primary_beacon_id(self) -> int | None:
        """ID of the mission primary beacon, or None."""
        return self._primary_beacon_id

    def set_primary_beacon(self, target_id: int) -> bool:
        """Designate exactly one primary beacon.

        Establishes the new primary relationship and drops the old one.
        Trajectories of all beacons are untouched, and the camera is
        never moved. Returns False if the ID does not exist.
        """
        for managed in self._targets:
            if managed.target.target_id == target_id:
                self._primary_beacon_id = target_id
                return True
        return False

    def clear_primary_beacon(self) -> None:
        """Remove the primary designation (beacons keep moving)."""
        self._primary_beacon_id = None

    def nudge_target(self, target_id: int, dx: float, dy: float, dz: float) -> bool:
        """Manually displace a user-controlled target (AI-blind challenge).

        Only trajectories exposing ``set_position`` (user_controlled)
        accept nudges. The keyboard command itself never reaches
        perception/tracking/AI — only the resulting rendered pixels do.

        Returns True if the nudge was accepted.
        """
        for managed in self._targets:
            if managed.target.target_id != target_id:
                continue
            traj = managed.trajectory
            if traj is not None and hasattr(traj, "set_position"):
                x, y, z = traj.position(self._time_s)
                traj.set_position(x + dx, y + dy, z + dz)
                return True
            return False
        return False

    def reset_target(self, target_id: int) -> bool:
        """Return a user-controlled target to its spawn position."""
        for managed in self._targets:
            if managed.target.target_id != target_id:
                continue
            traj = managed.trajectory
            params = managed.target.trajectory_params or {}
            if traj is not None and hasattr(traj, "set_position"):
                traj.set_position(
                    float(params.get("x0", managed.target.x)),
                    float(params.get("y0", managed.target.y)),
                    float(params.get("z0", managed.target.z)),
                )
                return True
            return False
        return False

    def add_target_from_dict(self, d: dict[str, Any]) -> WorldTargetState:
        """Add a target from a configuration dictionary."""
        traj_dict = d.get("trajectory")
        trajectory = None
        trajectory_type = d.get("trajectory_type", "")

        if traj_dict is not None:
            trajectory = self._registry.from_dict(traj_dict)
            trajectory_type = traj_dict.get("type", trajectory_type)

        target = WorldTargetState(
            target_id=d.get("target_id", len(self._targets)),
            active=d.get("active", True),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", 0.0),
            shape=d.get("shape", "square"),
            width=d.get("width", 0.01),
            height=d.get("height", 0.01),
            brightness=d.get("brightness", 1.0),
            code=d.get("code", "") or "",
            size_px=d.get("size_px"),
            seed=d.get("seed", 42),
            trajectory_type=trajectory_type,
            trajectory_params=d.get("trajectory_params", {}),
            spawn_time_s=d.get("spawn_time_s", self._time_s),
        )
        return self.add_target(target=target, trajectory=trajectory)

    def step(self, dt: float) -> WorldState:
        """Advance the simulation by *dt* seconds.

        Args:
            dt: Time step in seconds.  Must be positive.

        Returns:
            The world state after the step.

        Raises:
            SimulationError: If dt is not positive.
        """
        if dt <= 0:
            raise SimulationError(f"dt must be positive, got {dt}")

        self._time_s += dt
        self._step_count += 1

        for managed in self._targets:
            target = managed.target
            if not target.active:
                continue

            traj = managed.trajectory
            if traj is not None:
                # Evaluate trajectory at the new time
                if traj.is_finished(self._time_s):
                    target.active = False
                    continue

                px, py, pz = traj.position(self._time_s)
                vx, vy, vz = traj.velocity(self._time_s)
                ax, ay, az = traj.acceleration(self._time_s)

                target.x, target.y, target.z = px, py, pz
                target.vx, target.vy, target.vz = vx, vy, vz
                target.ax, target.ay, target.az = ax, ay, az
            else:
                # No trajectory: simple constant-velocity integration
                target.x += target.vx * dt
                target.y += target.vy * dt
                target.z += target.vz * dt

            # Apply boundary handling
            new_x, new_y, new_z, new_vx, new_vy, new_vz = apply_boundary_3d(
                target.x, target.y, target.z,
                target.vx, target.vy, target.vz,
                self._config.x_min, self._config.x_max,
                self._config.y_min, self._config.y_max,
                self._config.z_min, self._config.z_max,
                self._config.boundary_mode,
            )
            target.x, target.y, target.z = new_x, new_y, new_z
            target.vx, target.vy, target.vz = new_vx, new_vy, new_vz
            target.current_time_s = self._time_s

        # Platform motion (Terminal A itself moves): constant drift
        # integrates here; "static" only stamps time (Stage-2 behavior).
        from fsoc_tracker.simulation.platform import step_platform
        step_platform(self._platform, self._platform_motion,
                      self._platform_motion_params, dt)
        self._platform.timestamp_s = self._time_s

        return self.get_state()

    def update_to(self, time_s: float) -> WorldState:
        """Advance (or rewind) the simulation to a specific time.

        For rewinding, a full reset + re-simulate is performed.
        """
        if time_s < 0:
            raise SimulationError("target time must be non-negative")

        if time_s < self._time_s:
            self.reset()
            return self._fast_forward(time_s)
        return self._fast_forward(time_s)

    def _fast_forward(self, target_time: float) -> WorldState:
        """Step forward until reaching *target_time*."""
        while self._time_s < target_time:
            dt = min(0.1, target_time - self._time_s)
            self.step(dt)
        return self.get_state()

    def get_state(self) -> WorldState:
        """Return the current world state snapshot."""
        targets = [
            WorldTargetState(
                target_id=m.target.target_id,
                active=m.target.active,
                x=m.target.x,
                y=m.target.y,
                z=m.target.z,
                vx=m.target.vx,
                vy=m.target.vy,
                vz=m.target.vz,
                ax=m.target.ax,
                ay=m.target.ay,
                az=m.target.az,
                shape=m.target.shape,
                width=m.target.width,
                height=m.target.height,
                brightness=m.target.brightness,
                code=m.target.code,
                size_px=m.target.size_px,
                seed=m.target.seed,
                trajectory_type=m.target.trajectory_type,
                trajectory_params=m.target.trajectory_params,
                spawn_time_s=m.target.spawn_time_s,
                current_time_s=self._time_s,
                metadata=m.target.metadata,
            )
            for m in self._targets
        ]
        return WorldState(
            config=self._config,
            targets=targets,
            platform=PlatformState(
                x=self._platform.x,
                y=self._platform.y,
                z=self._platform.z,
                roll_deg=self._platform.roll_deg,
                pitch_deg=self._platform.pitch_deg,
                yaw_deg=self._platform.yaw_deg,
                vx=self._platform.vx,
                vy=self._platform.vy,
                vz=self._platform.vz,
                roll_rate_deg_s=self._platform.roll_rate_deg_s,
                pitch_rate_deg_s=self._platform.pitch_rate_deg_s,
                yaw_rate_deg_s=self._platform.yaw_rate_deg_s,
                timestamp_s=self._time_s,
            ),
            simulation_time_s=self._time_s,
            step_count=self._step_count,
        )

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._time_s = 0.0
        self._step_count = 0
        self._platform = PlatformState()
        for managed in self._targets:
            if managed.trajectory is not None:
                managed.trajectory.reset()
            # Restore initial target state from trajectory start
            if managed.trajectory is not None:
                px, py, pz = managed.trajectory.position(0.0)
                vx, vy, vz = managed.trajectory.velocity(0.0)
                managed.target.x, managed.target.y, managed.target.z = px, py, pz
                managed.target.vx, managed.target.vy, managed.target.vz = vx, vy, vz
            managed.target.active = True
            managed.target.current_time_s = 0.0


class _ManagedTarget:
    """Internal container pairing a target with its trajectory."""

    __slots__ = ("target", "trajectory")

    def __init__(self, target: WorldTargetState, trajectory: Trajectory | None) -> None:
        self.target = target
        self.trajectory = trajectory
