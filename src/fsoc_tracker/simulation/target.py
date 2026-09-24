"""World-space target state for the simulation environment.

This is distinct from ``core.models.TargetState`` which represents
image-space target position.  ``WorldTargetState`` represents the
physical position of a target in the simulation world coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldTargetState:
    """State of a target in world coordinates.

    The world coordinate system is:
        X = horizontal (right)
        Y = vertical (up)
        Z = depth (forward)

    All positions are in world units (meters by default).
    Velocities are in world units per second.
    """

    target_id: int = 0
    active: bool = True

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    shape: str = "square"
    width: float = 0.01
    height: float = 0.01
    brightness: float = 1.0
    # Temporal identity code (binary string); "" = steady ON. The sensor
    # renderer multiplies brightness by the code bit at each frame index.
    code: str = ""
    # Explicit per-target pixel size. None = fall back to the sensor's
    # default (today's behavior everywhere); a value overrides it in
    # both FIXED and DISTANCE size modes.
    size_px: float | None = None

    trajectory_type: str = ""
    trajectory_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    spawn_time_s: float = 0.0
    current_time_s: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def velocity(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "target_id": self.target_id,
            "active": self.active,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "shape": self.shape,
            "width": self.width,
            "height": self.height,
            "brightness": self.brightness,
            "code": self.code,
            "size_px": self.size_px,
            "trajectory_type": self.trajectory_type,
            "trajectory_params": self.trajectory_params,
            "seed": self.seed,
            "spawn_time_s": self.spawn_time_s,
            "current_time_s": self.current_time_s,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldTargetState:
        """Deserialize from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
