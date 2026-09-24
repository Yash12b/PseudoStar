"""Coarse pointing controller.

Consumes TrackingState, computes pixel-to-angle error,
runs PID, and produces ControlCommand.

Never uses ground truth.
"""

from __future__ import annotations

import math

from fsoc_tracker.control.command import ControlCommand, ControlTelemetry
from fsoc_tracker.control.config import ControllerConfig, ControlMode
from fsoc_tracker.control.pid import PIDController
from fsoc_tracker.simulation.camera.projection import pixel_to_angle
from fsoc_tracker.simulation.camera.state import CameraIntrinsics
from fsoc_tracker.tracking.state import TrackingState, TrackState


class CoarsePointingController:
    """Closed-loop coarse pointing controller.

    Usage::

        controller = CoarsePointingController(config)
        command = controller.compute(tracking_state, camera_intrinsics, dt)
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self._config = config or ControllerConfig()
        self._pan_pid = PIDController(
            kp=self._config.pan_kp,
            ki=self._config.pan_ki,
            kd=self._config.pan_kd,
            output_limit=self._config.max_pan_rate_deg_s,
            integral_limit=self._config.integral_limit_pan,
            derivative_filter_alpha=self._config.derivative_filter_alpha,
            deadband=self._config.deadband_deg,
        )
        self._tilt_pid = PIDController(
            kp=self._config.tilt_kp,
            ki=self._config.tilt_ki,
            kd=self._config.tilt_kd,
            output_limit=self._config.max_tilt_rate_deg_s,
            integral_limit=self._config.integral_limit_tilt,
            derivative_filter_alpha=self._config.derivative_filter_alpha,
            deadband=self._config.deadband_deg,
        )
        self._mode = self._config.mode
        self._prediction_start_time_s: float = 0.0
        self._last_target_x: float = 0.0
        self._last_target_y: float = 0.0
        self._initialized: bool = False
        self._lead_dx_px: float = 0.0
        self._lead_dy_px: float = 0.0

    @property
    def config(self) -> ControllerConfig:
        return self._config

    def set_lead_enabled(self, enabled: bool) -> None:
        """Enable/disable lead-angle compensation at runtime.

        Used by the mission-brain gating: fast steady targets get lead,
        everything else keeps pure estimate tracking.
        """
        self._config.lead_compensation_enabled = bool(enabled)

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def pan_pid(self) -> PIDController:
        return self._pan_pid

    @property
    def tilt_pid(self) -> PIDController:
        return self._tilt_pid

    def compute(
        self,
        tracking_state: TrackingState,
        camera_intrinsics: CameraIntrinsics,
        dt: float,
        timestamp_s: float = 0.0,
    ) -> tuple[ControlCommand, ControlTelemetry]:
        """Compute control command from tracking state.

        Args:
            tracking_state: Current tracking state (from KalmanTracker).
            camera_intrinsics: Camera intrinsic parameters for pixel→angle conversion.
            dt: Control loop time step (seconds).
            timestamp_s: Current timestamp.

        Returns:
            (ControlCommand, ControlTelemetry) tuple.
        """
        telemetry = ControlTelemetry(
            timestamp_s=timestamp_s,
            dt=dt,
        )

        # Safety checks
        if dt <= 0 or not self._config.enabled:
            return (
                ControlCommand(timestamp_s=timestamp_s, control_mode=ControlMode.DISABLED),
                telemetry,
            )

        if math.isnan(dt) or math.isinf(dt):
            return (
                ControlCommand(timestamp_s=timestamp_s, control_mode=ControlMode.DISABLED),
                telemetry,
            )

        # Determine control mode based on tracking state
        self._update_mode(tracking_state, timestamp_s)
        telemetry.control_mode = self._mode

        # Safe stop and disabled produce zero output
        if self._mode in (ControlMode.DISABLED, ControlMode.SAFE_STOP):
            if (
                self._mode == ControlMode.DISABLED
                and tracking_state.state == TrackState.SEARCHING
                and self._config.search_enabled
            ):
                return self._search_command(timestamp_s, dt, telemetry)
            return self._zero_command(timestamp_s, telemetry)

        # Get target position
        target_x, target_y = self._get_target_position(tracking_state, timestamp_s)
        self._last_target_x = target_x
        self._last_target_y = target_y
        self._initialized = True
        telemetry.target_x = target_x
        telemetry.target_y = target_y

        # Image center
        cx = float(camera_intrinsics.cx)
        cy = float(camera_intrinsics.cy)
        telemetry.center_x = cx
        telemetry.center_y = cy

        # Pixel error
        pixel_error_x = target_x - cx
        pixel_error_y = target_y - cy
        telemetry.pixel_error_x = pixel_error_x
        telemetry.pixel_error_y = pixel_error_y

        # Convert pixel error to angular error using camera geometry
        # pixel_to_angle returns (h_angle_deg, v_angle_deg) from optical axis
        # For a point at (target_x, target_y), the angular offset is:
        h_angle, v_angle = pixel_to_angle(target_x, target_y, camera_intrinsics)
        telemetry.angular_error_pan_deg = h_angle
        telemetry.angular_error_tilt_deg = v_angle

        # Compute PID outputs
        pan_rate = self._pan_pid.update(h_angle, dt)
        tilt_rate = self._tilt_pid.update(v_angle, dt)

        # Fill telemetry PID terms
        telemetry.pan_p_term = self._pan_pid.p_term
        telemetry.pan_i_term = self._pan_pid.i_term
        telemetry.pan_d_term = self._pan_pid.d_term
        telemetry.pan_raw_derivative = self._pan_pid.raw_derivative
        telemetry.pan_filtered_derivative = self._pan_pid.filtered_derivative
        telemetry.pan_output_before_sat = self._pan_pid.output_before_saturation
        telemetry.pan_saturated = self._pan_pid.saturated
        telemetry.pan_in_deadband = self._pan_pid.in_deadband

        telemetry.tilt_p_term = self._tilt_pid.p_term
        telemetry.tilt_i_term = self._tilt_pid.i_term
        telemetry.tilt_d_term = self._tilt_pid.d_term
        telemetry.tilt_raw_derivative = self._tilt_pid.raw_derivative
        telemetry.tilt_filtered_derivative = self._tilt_pid.filtered_derivative
        telemetry.tilt_output_before_sat = self._tilt_pid.output_before_saturation
        telemetry.tilt_saturated = self._tilt_pid.saturated
        telemetry.tilt_in_deadband = self._tilt_pid.in_deadband

        telemetry.pan_command = pan_rate
        telemetry.tilt_command = tilt_rate
        telemetry.lead_dx_px = self._lead_dx_px
        telemetry.lead_dy_px = self._lead_dy_px

        telemetry.tracking_quality = tracking_state.quality
        telemetry.lock_status = tracking_state.locked
        telemetry.prediction_only = tracking_state.prediction_only

        # Build command
        command = ControlCommand(
            pan_rate_deg_s=pan_rate,
            tilt_rate_deg_s=tilt_rate,
            timestamp_s=timestamp_s,
            control_mode=self._mode,
            tracking_valid=tracking_state.state in (TrackState.TRACKING, TrackState.REACQUIRING, TrackState.ACQUIRING),
            pan_saturated=self._pan_pid.saturated,
            tilt_saturated=self._tilt_pid.saturated,
        )

        return command, telemetry

    def reset(self) -> None:
        """Reset controller state."""
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._mode = self._config.mode
        self._prediction_start_time_s = 0.0
        self._initialized = False
        self._lead_dx_px = 0.0
        self._lead_dy_px = 0.0

    def set_mode(self, mode: ControlMode) -> None:
        """Set controller mode."""
        self._mode = mode

    def _update_mode(self, tracking_state: TrackingState, timestamp_s: float) -> None:
        """Update control mode based on tracking state."""
        ts = tracking_state.state

        if ts == TrackState.TRACKING:
            self._mode = ControlMode.TRACK
            self._prediction_start_time_s = 0.0

        elif ts == TrackState.REACQUIRING or ts == TrackState.ACQUIRING:
            self._mode = ControlMode.TRACK

        elif ts == TrackState.SEARCHING:
            self._mode = ControlMode.DISABLED

        elif ts == TrackState.LOST:
            if self._config.prediction_control_enabled:
                if self._prediction_start_time_s == 0.0:
                    self._prediction_start_time_s = timestamp_s
                elapsed = timestamp_s - self._prediction_start_time_s
                if elapsed <= self._config.max_prediction_control_duration_s:
                    self._mode = ControlMode.PREDICT
                else:
                    self._mode = ControlMode.SAFE_STOP
            else:
                self._mode = ControlMode.SAFE_STOP

        elif ts == TrackState.NO_TRACK:
            self._mode = ControlMode.DISABLED

    def _search_command(
        self,
        timestamp_s: float,
        dt: float,
        telemetry: ControlTelemetry,
    ) -> tuple[ControlCommand, ControlTelemetry]:
        """Sweep pan while slowly scanning tilt during initial acquisition.

        SEARCHING has no image-space measurement, so this is an explicit
        open-loop acquisition maneuver. Once a detection exists, normal PID
        tracking takes over.
        """
        phase = (timestamp_s % self._config.search_period_s) / self._config.search_period_s
        pan_sign = 1.0 if phase < 0.5 else -1.0
        tilt_rate = (
            self._config.search_tilt_rate_deg_s
            * math.sin(2.0 * math.pi * phase)
        )
        pan_rate = pan_sign * self._config.search_pan_rate_deg_s
        telemetry.control_mode = ControlMode.HOLD
        telemetry.pan_command = pan_rate
        telemetry.tilt_command = tilt_rate
        command = ControlCommand(
            pan_rate_deg_s=pan_rate,
            tilt_rate_deg_s=tilt_rate,
            timestamp_s=timestamp_s,
            control_mode=ControlMode.HOLD,
            tracking_valid=False,
        )
        return command, telemetry

    def _get_target_position(
        self, tracking_state: TrackingState, timestamp_s: float,
    ) -> tuple[float, float]:
        """Get the target position to control toward."""
        if self._mode == ControlMode.PREDICT and tracking_state.prediction_only:
            return tracking_state.predicted_x, tracking_state.predicted_y

        if tracking_state.state in (TrackState.TRACKING, TrackState.REACQUIRING, TrackState.ACQUIRING):
            tx, ty = tracking_state.estimated_x, tracking_state.estimated_y
            return self._apply_lead(tracking_state, tx, ty)

        if self._initialized:
            return self._last_target_x, self._last_target_y

        return 0.0, 0.0

    def _apply_lead(
        self, tracking_state: TrackingState, tx: float, ty: float,
    ) -> tuple[float, float]:
        """Aim ahead along estimated velocity (bounded lead angle).

        Only when enabled, locked, and velocity is finite. The lead is
        capped so noisy velocity can never fling the aim point.
        """
        self._lead_dx_px = 0.0
        self._lead_dy_px = 0.0
        if not self._config.lead_compensation_enabled:
            return tx, ty
        if not tracking_state.locked:
            return tx, ty
        vx, vy = tracking_state.velocity_x, tracking_state.velocity_y
        if not (math.isfinite(vx) and math.isfinite(vy)):
            return tx, ty
        lead = self._config.lead_time_s
        dx, dy = vx * lead, vy * lead
        mag = math.hypot(dx, dy)
        cap = self._config.lead_max_px
        if mag > cap > 0:
            dx, dy = dx / mag * cap, dy / mag * cap
        self._lead_dx_px = dx
        self._lead_dy_px = dy
        return tx + dx, ty + dy

    def _zero_command(
        self, timestamp_s: float, telemetry: ControlTelemetry,
    ) -> tuple[ControlCommand, ControlTelemetry]:
        """Return zero command."""
        telemetry.control_mode = self._mode
        return (
            ControlCommand(
                timestamp_s=timestamp_s,
                control_mode=self._mode,
            ),
            telemetry,
        )


class CameraActuator:
    """Adapter between controller commands and camera state.

    Applies rate commands to the VirtualCamera. The camera's own
    update(dt) performs rate limiting — the actuator must NOT
    pre-clamp, or effective rate is halved.
    """

    def __init__(self) -> None:
        pass

    def apply_command(
        self,
        camera: object,
        command: ControlCommand,
        dt: float,
    ) -> None:
        """Apply a control command to a camera.

        Args:
            camera: VirtualCamera instance.
            command: ControlCommand from the controller.
            dt: Time step.
        """
        if dt <= 0:
            return

        if command.control_mode in (ControlMode.DISABLED, ControlMode.SAFE_STOP):
            return

        # Compute target angles from current + rate.
        # Do NOT pre-clamp — camera.update(dt) applies rate limiting.
        current_pan = camera.state.pan_deg
        current_tilt = camera.state.tilt_deg

        target_pan = current_pan + command.pan_rate_deg_s * dt
        target_tilt = current_tilt + command.tilt_rate_deg_s * dt

        camera.set_target_pan_tilt(target_pan, target_tilt)
        camera.update(dt)
