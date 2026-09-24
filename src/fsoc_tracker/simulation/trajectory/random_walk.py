"""Random walk trajectory.

Each step is a random displacement from the current position.
Uses seeded RNG for deterministic behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fsoc_tracker.simulation.trajectory.base import Trajectory


class RandomWalkTrajectory(Trajectory):
    """Random walk: each step is a random displacement from current position.

    Args:
        x0, y0, z0: Starting position.
        step_size: Maximum displacement per step (in each axis).
        step_duration: Seconds between steps.
        x_min, x_max, y_min, y_max, z_min, z_max: World bounds.
        seed: Random seed.
    """

    def __init__(
        self,
        x0: float = 1000.0,
        y0: float = 1000.0,
        z0: float = 500.0,
        step_size: float = 50.0,
        step_duration: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 2000.0,
        y_min: float = 0.0,
        y_max: float = 2000.0,
        z_min: float = 0.0,
        z_max: float = 2000.0,
        seed: int = 42,
    ) -> None:
        self._x0 = x0
        self._y0 = y0
        self._z0 = z0
        self._step_size = step_size
        self._step_duration = step_duration
        self._bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        self._seed = seed
        self._positions: list[tuple[float, float, float]] = []
        self._generate_walk()

    def _generate_walk(self) -> None:
        rng = np.random.default_rng(self._seed)
        x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
        self._positions = [(self._x0, self._y0, self._z0)]
        cx, cy, cz = self._x0, self._y0, self._z0
        for _ in range(500):
            dx = rng.uniform(-self._step_size, self._step_size)
            dy = rng.uniform(-self._step_size, self._step_size)
            dz = rng.uniform(-self._step_size * 0.2, self._step_size * 0.2)
            cx = max(x_min, min(x_max, cx + dx))
            cy = max(y_min, min(y_max, cy + dy))
            cz = max(z_min, min(z_max, cz + dz))
            self._positions.append((cx, cy, cz))

    def reset(self) -> None:
        self._generate_walk()

    def position(self, t: float) -> tuple[float, float, float]:
        if t <= 0:
            return self._positions[0]
        idx = int(t / self._step_duration)
        idx = min(idx, len(self._positions) - 2)
        local_t = (t - idx * self._step_duration) / self._step_duration
        local_t = max(0.0, min(1.0, local_t))
        p0 = self._positions[idx]
        p1 = self._positions[idx + 1]
        return (
            p0[0] + local_t * (p1[0] - p0[0]),
            p0[1] + local_t * (p1[1] - p0[1]),
            p0[2] + local_t * (p1[2] - p0[2]),
        )

    def velocity(self, t: float) -> tuple[float, float, float]:
        dt = 1e-4
        p0 = self.position(t - dt)
        p1 = self.position(t + dt)
        return ((p1[0] - p0[0]) / (2 * dt), (p1[1] - p0[1]) / (2 * dt), (p1[2] - p0[2]) / (2 * dt))

    def acceleration(self, t: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "random_walk",
            "x0": self._x0, "y0": self._y0, "z0": self._z0,
            "step_size": self._step_size, "step_duration": self._step_duration,
            "x_min": self._bounds[0], "x_max": self._bounds[1],
            "y_min": self._bounds[2], "y_max": self._bounds[3],
            "z_min": self._bounds[4], "z_max": self._bounds[5],
            "seed": self._seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RandomWalkTrajectory:
        return cls(
            x0=d.get("x0", 1000.0), y0=d.get("y0", 1000.0), z0=d.get("z0", 500.0),
            step_size=d.get("step_size", 50.0), step_duration=d.get("step_duration", 1.0),
            x_min=d.get("x_min", 0.0), x_max=d.get("x_max", 2000.0),
            y_min=d.get("y_min", 0.0), y_max=d.get("y_max", 2000.0),
            z_min=d.get("z_min", 0.0), z_max=d.get("z_max", 2000.0),
            seed=d.get("seed", 42),
        )
