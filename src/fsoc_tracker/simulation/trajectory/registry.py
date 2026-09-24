"""Trajectory factory / registry.

Provides a clean API for creating trajectory instances from configuration.
Adding a new trajectory type requires only:
    1. Implement the Trajectory ABC
    2. Register it with TrajectoryRegistry
"""

from __future__ import annotations

from typing import Any

from fsoc_tracker.core.exceptions import ConfigurationError
from fsoc_tracker.simulation.trajectory.accel_decel import AccelDecelTrajectory
from fsoc_tracker.simulation.trajectory.base import Trajectory
from fsoc_tracker.simulation.trajectory.circular import CircularTrajectory
from fsoc_tracker.simulation.trajectory.figure_eight import FigureEightTrajectory
from fsoc_tracker.simulation.trajectory.random import RandomTrajectory
from fsoc_tracker.simulation.trajectory.random_walk import RandomWalkTrajectory
from fsoc_tracker.simulation.trajectory.sinusoidal import SinusoidalTrajectory
from fsoc_tracker.simulation.trajectory.spiral import SpiralTrajectory
from fsoc_tracker.simulation.trajectory.stop_go import StopGoTrajectory
from fsoc_tracker.simulation.trajectory.straight_line import StraightLineTrajectory
from fsoc_tracker.simulation.trajectory.sudden_reversal import SuddenReversalTrajectory
from fsoc_tracker.simulation.trajectory.user_controlled import UserControlledTrajectory


class TrajectoryRegistry:
    """Registry of trajectory types mapped to their classes.

    Usage::

        registry = TrajectoryRegistry()
        traj = registry.create("circular", {"radius": 200, "cx": 500})

        # Or register a custom type:
        registry.register("my_type", MyTrajectory)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Trajectory]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register("straight_line", StraightLineTrajectory)
        self.register("circular", CircularTrajectory)
        self.register("figure_8", FigureEightTrajectory)
        self.register("random", RandomTrajectory)
        self.register("spiral", SpiralTrajectory)
        self.register("sinusoidal", SinusoidalTrajectory)
        self.register("random_walk", RandomWalkTrajectory)
        self.register("stop_go", StopGoTrajectory)
        self.register("sudden_reversal", SuddenReversalTrajectory)
        self.register("accel_decel", AccelDecelTrajectory)
        self.register("user_controlled", UserControlledTrajectory)

    def register(self, name: str, cls: type[Trajectory]) -> None:
        """Register a trajectory class under the given name.

        Args:
            name: String identifier (e.g. "circular").
            cls: Class that implements the Trajectory interface.

        Raises:
            TypeError: If *cls* is not a Trajectory subclass.
        """
        if not (isinstance(cls, type) and issubclass(cls, Trajectory)):
            raise TypeError(f"{cls} is not a subclass of Trajectory")
        self._registry[name] = cls

    def create(self, name: str, params: dict[str, Any] | None = None) -> Trajectory:
        """Create a trajectory instance from configuration.

        Args:
            name: Trajectory type name (e.g. "circular").
            params: Parameters passed to the trajectory constructor.

        Returns:
            A configured Trajectory instance.

        Raises:
            ConfigurationError: If the trajectory type is unknown.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ConfigurationError(
                f"Unknown trajectory type '{name}'. Available: {available}"
            )
        cls = self._registry[name]
        return cls(**(params or {}))

    def from_dict(self, d: dict[str, Any]) -> Trajectory:
        """Create a trajectory from a serialized dictionary.

        The dictionary must contain a "type" key.
        """
        traj_type = d.get("type")
        if traj_type is None:
            raise ConfigurationError("Trajectory dict must contain a 'type' key")
        params = {k: v for k, v in d.items() if k != "type"}
        return self.create(traj_type, params)

    @property
    def available(self) -> list[str]:
        """List of registered trajectory type names."""
        return sorted(self._registry.keys())
