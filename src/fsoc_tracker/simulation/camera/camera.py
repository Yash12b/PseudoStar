"""VirtualCamera: main API for the virtual PTZ camera.

Handles rate-limited pan/tilt updates, projection, and state management.
Supports 4 camera view modes:
  - FREE_WORLD: Independent orbit camera for exploration.
  - TERMINAL_A_POV: Camera follows Terminal A's actual optical orientation.
  - FOLLOW: Camera tracks a target via PID commands.
  - SEARCH_TRACK: Camera performs search pattern, then locks via PID.

Camera movement is always caused by external commands (PID controller,
user input, or platform state), never by directly reading beacon world
position.
"""

from __future__ import annotations

from fsoc_tracker.simulation.camera.geometry import camera_rotation_matrix
from fsoc_tracker.simulation.camera.projection import (
    angle_to_pixel,
    compute_angular_error,
    pixel_to_angle,
    project_to_image,
)
from fsoc_tracker.simulation.camera.state import (
    AngularError,
    CameraIntrinsics,
    CameraLimits,
    CameraState,
    CameraViewMode,
    ProjectionResult,
)


class VirtualCamera:
    """Virtual PTZ camera with rate-limited control and projection.

    The camera is the authoritative source of:
      - position (x, y, z)
      - orientation (yaw/pan, pitch/tilt, roll)
      - FOV (hfov, vfov)
      - resolution (width, height)
      - pan rate, tilt rate (max speeds)

    All external systems (PID, search, user) drive the camera via:
      - set_target_pan_tilt(): PID/search commands
      - apply_platform_state(): Terminal A POV
      - set_position(): Position updates

    Usage::

        camera = VirtualCamera(state)
        camera.set_view_mode(CameraViewMode.TERMINAL_A_POV)
        camera.apply_platform_state(platform)
        camera.update(dt)
        result = camera.project_world_point((500, 500, 80))
    """

    def __init__(
        self,
        state: CameraState | None = None,
        limits: CameraLimits | None = None,
    ) -> None:
        self._state = state or CameraState()
        self._limits = limits or CameraLimits()
        self._target_pan_deg: float = self._state.pan_deg
        self._target_tilt_deg: float = self._state.tilt_deg
        self._rotation_matrix: list[list[float]] = []
        self._dirty = True
        self._rebuild_rotation()

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def rotation_matrix(self) -> list[list[float]]:
        return self._rotation_matrix

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._state.intrinsics

    @property
    def position(self) -> tuple[float, float, float]:
        return self._state.position

    @property
    def yaw(self) -> float:
        """Current yaw in degrees (alias for pan_deg)."""
        return self._state.pan_deg

    @property
    def pitch(self) -> float:
        """Current pitch in degrees (alias for tilt_deg)."""
        return self._state.tilt_deg

    @property
    def view_mode(self) -> CameraViewMode:
        return self._state.view_mode

    def set_view_mode(self, mode: CameraViewMode) -> None:
        """Set the camera view mode."""
        self._state.view_mode = mode

    def set_target_pan_tilt(self, pan_deg: float, tilt_deg: float) -> None:
        """Set the target pan/tilt angles (degrees).

        Called by PID controller or search controller to command
        the camera orientation.  The actual orientation moves toward
        this target at the configured max rate.
        """
        self._target_pan_deg = pan_deg
        self._target_tilt_deg = tilt_deg

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set camera position in world coordinates."""
        self._state.position_x = x
        self._state.position_y = y
        self._state.position_z = z
        self._dirty = True

    def apply_platform_state(
        self,
        platform_x: float,
        platform_y: float,
        platform_z: float,
        platform_yaw_deg: float,
        platform_pitch_deg: float,
    ) -> None:
        """Apply Terminal A's platform state to the camera (TERMINAL_A_POV).

        This sets the camera's position and orientation to match the
        physical platform — NOT the beacon's world position.  The camera
        sees what Terminal A's optics see.

        Called once per frame when in TERMINAL_A_POV mode.
        """
        self._state.position_x = platform_x
        self._state.position_y = platform_y
        self._state.position_z = platform_z
        self._state.pan_deg = platform_yaw_deg
        self._state.tilt_deg = platform_pitch_deg
        self._target_pan_deg = platform_yaw_deg
        self._target_tilt_deg = platform_pitch_deg
        self._dirty = True

    def update(self, dt: float) -> None:
        """Advance camera state by dt seconds.

        In TERMINAL_A_POV mode: orientation comes from apply_platform_state,
        so this just advances timestamp and rebuilds rotation.

        In other modes: moves pan/tilt toward target with rate limiting.
        """
        if dt <= 0:
            return

        if self._state.view_mode != CameraViewMode.TERMINAL_A_POV:
            # Rate-limited pan/tilt movement for non-POV modes
            if self._state.pan_deg != self._target_pan_deg or self._state.tilt_deg != self._target_tilt_deg:
                max_pan_step = self._state.max_pan_speed_deg_s * dt
                max_tilt_step = self._state.max_tilt_speed_deg_s * dt

                pan_delta = self._target_pan_deg - self._state.pan_deg
                tilt_delta = self._target_tilt_deg - self._state.tilt_deg

                pan_step = max(-max_pan_step, min(max_pan_step, pan_delta))
                tilt_step = max(-max_tilt_step, min(max_tilt_step, tilt_delta))

                self._state.pan_deg += pan_step
                self._state.tilt_deg += tilt_step

                self._state.pan_deg = self._limits.clamp_pan(self._state.pan_deg)
                self._state.tilt_deg = self._limits.clamp_tilt(self._state.tilt_deg)

                self._dirty = True

        self._state.timestamp_s += dt

        if self._dirty:
            self._rebuild_rotation()

    def _rebuild_rotation(self) -> None:
        self._rotation_matrix = camera_rotation_matrix(
            self._state.pan_deg,
            self._state.tilt_deg,
            self._state.roll_deg,
        )
        self._dirty = False

    def project_world_point(self, point: tuple[float, float, float]) -> ProjectionResult:
        """Project a world-space point into camera pixel coordinates.

        This is the single authoritative projection API.
        """
        return project_to_image(
            point,
            self._state.position,
            self._state.intrinsics,
            self._rotation_matrix,
        )

    def compute_error(self, target_world: tuple[float, float, float]) -> AngularError:
        """Compute angular error to a target in world coordinates."""
        h_deg, v_deg = compute_angular_error(
            target_world,
            self._state.position,
            self._rotation_matrix,
        )
        return AngularError(horizontal_angle_deg=h_deg, vertical_angle_deg=v_deg)

    def pixel_to_angle(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert pixel coordinates to angular offsets."""
        return pixel_to_angle(pixel_x, pixel_y, self._state.intrinsics)

    def angle_to_pixel(self, angle_h_deg: float, angle_v_deg: float) -> tuple[float, float]:
        """Convert angular offsets to pixel coordinates."""
        return angle_to_pixel(angle_h_deg, angle_v_deg, self._state.intrinsics)

    def reset(self) -> None:
        """Reset camera to initial state."""
        self._state.pan_deg = 0.0
        self._state.tilt_deg = 0.0
        self._target_pan_deg = 0.0
        self._target_tilt_deg = 0.0
        self._state.timestamp_s = 0.0
        self._dirty = True
        self._rebuild_rotation()
