"""Platform state for the simulation environment.

The platform represents the physical mounting of the FSOC terminal.
It may be stationary ("static", the default) or follow a
constant-velocity drift with constant attitude rates ("drift", e.g. a
satellite bus), stepped by the simulation engine every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PLATFORM_MOTIONS = ("static", "drift")


@dataclass
class PlatformState:
    """State of the FSOC terminal platform in world coordinates.

    Represents position, orientation, and angular velocity of the
    platform that carries the virtual camera.  Initially stationary.

    Attributes:
        x, y, z: Position in world coordinates (meters).
        roll_deg, pitch_deg, yaw_deg: Orientation in degrees.
        vx, vy, vz: Linear velocity (m/s).
        roll_rate_deg_s, pitch_rate_deg_s, yaw_rate_deg_s: Angular velocity (deg/s).
        timestamp_s: Current simulation time.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    roll_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0

    timestamp_s: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "roll_rate_deg_s": self.roll_rate_deg_s,
            "pitch_rate_deg_s": self.pitch_rate_deg_s,
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
            "timestamp_s": self.timestamp_s,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlatformState:
        """Deserialize from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def step_platform(
    platform: PlatformState,
    motion: str,
    params: dict[str, Any],
    dt: float,
) -> PlatformState:
    """Advance the platform pose by dt (in place, also returned).

    "static" only stamps time. "drift" integrates constant linear
    velocity (world units/s) and constant attitude rates (deg/s).
    Unknown motion names raise ValueError (fail loudly, never drift
    silently on a typo).
    """
    if dt <= 0:
        return platform
    if motion == "static":
        pass
    elif motion == "drift":
        vx = float(params.get("vx", 0.0))
        vy = float(params.get("vy", 0.0))
        vz = float(params.get("vz", 0.0))
        platform.x += vx * dt
        platform.y += vy * dt
        platform.z += vz * dt
        platform.vx, platform.vy, platform.vz = vx, vy, vz
        platform.yaw_deg += float(params.get("yaw_rate_deg_s", 0.0)) * dt
        platform.pitch_deg += float(params.get("pitch_rate_deg_s", 0.0)) * dt
        platform.roll_deg += float(params.get("roll_rate_deg_s", 0.0)) * dt
        platform.yaw_rate_deg_s = float(params.get("yaw_rate_deg_s", 0.0))
        platform.pitch_rate_deg_s = float(params.get("pitch_rate_deg_s", 0.0))
        platform.roll_rate_deg_s = float(params.get("roll_rate_deg_s", 0.0))
    else:
        raise ValueError(f"Unknown platform motion: {motion!r}")
    return platform


def apply_platform_delta_to_camera(
    camera_state: Any,
    prev: PlatformState,
    cur: PlatformState,
) -> None:
    """Strapdown carry: move the mount with its platform (in place).

    The camera keeps world-frame (inertially stabilized) pointing: the
    platform's translation shifts the viewpoint and its rotation turns
    the boresight by the same angles. Gimbal rate commands from the
    controller then operate on top, so platform motion appears to the
    loop exactly as what it is — a pointing disturbance the PID
    rejects within its 5 deg/s authority. No range-stop model (already
    a documented limitation).
    """
    dx, dy, dz, dyaw, dpitch, droll = _platform_delta_numbers(prev, cur)
    camera_state.position_x += dx
    camera_state.position_y += dy
    camera_state.position_z += dz
    camera_state.pan_deg += dyaw
    camera_state.tilt_deg += dpitch
    camera_state.roll_deg += droll


def _platform_delta_numbers(
    prev: PlatformState,
    cur: PlatformState,
) -> tuple[float, float, float, float, float, float]:
    """Coerce a platform delta to floats.

    Raises TypeError on non-numeric poses (miswired engine, mock
    objects) so callers skip the carry instead of poisoning camera
    state with junk arithmetic.
    """
    try:
        return (
            float(cur.x) - float(prev.x),
            float(cur.y) - float(prev.y),
            float(cur.z) - float(prev.z),
            float(cur.yaw_deg) - float(prev.yaw_deg),
            float(cur.pitch_deg) - float(prev.pitch_deg),
            float(cur.roll_deg) - float(prev.roll_deg),
        )
    except (TypeError, ValueError) as e:
        raise TypeError(f"non-numeric platform pose: {e}") from e
