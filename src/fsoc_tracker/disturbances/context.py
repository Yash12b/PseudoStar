"""Disturbance context — typed data passed to each disturbance module.

Contains everything a disturbance needs without coupling to the entire application.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraPoseContext:
    """Snapshot of camera pose for geometric disturbances."""

    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass
class PlatformStateContext:
    """Snapshot of platform state for motion disturbances."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class EnvironmentContext:
    """Snapshot of environment state."""

    ambient_light: float = 1.0
    temperature_c: float = 20.0
    wind_speed: float = 0.0
    wind_angle_deg: float = 0.0


@dataclass
class DisturbanceContext:
    """Typed context passed to every disturbance module.

    Contains all information needed for disturbance computation
    without coupling to the main application.
    """

    timestamp_s: float = 0.0
    dt: float = 0.0
    frame_index: int = 0

    image_width: int = 640
    image_height: int = 480

    camera_pose: CameraPoseContext = field(default_factory=CameraPoseContext)
    platform_state: PlatformStateContext = field(default_factory=PlatformStateContext)
    environment: EnvironmentContext = field(default_factory=EnvironmentContext)

    random_seed: int = 42

    def with_image_size(self, width: int, height: int) -> DisturbanceContext:
        """Return a copy with updated image dimensions."""
        return DisturbanceContext(
            timestamp_s=self.timestamp_s,
            dt=self.dt,
            frame_index=self.frame_index,
            image_width=width,
            image_height=height,
            camera_pose=self.camera_pose,
            platform_state=self.platform_state,
            environment=self.environment,
            random_seed=self.random_seed,
        )
