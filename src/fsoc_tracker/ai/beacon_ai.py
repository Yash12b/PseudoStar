"""FSOC Beacon AI — Ground-truth-free intelligence for beacon tracking and communication.

Consumes ONLY:
  - TrackingState (pixel-space estimates from KalmanTracker)
  - CameraIntrinsics (focal lengths, principal point)
  - Optional image for atmospheric estimation

NEVER accesses:
  - World-space positions
  - Simulation engine state
  - Ground truth

Handles:
1. Situation assessment from tracking state
2. Distance/angle estimation from pixel error + intrinsics
3. Predictive tracking (lead-angle computation)
4. Atmospheric disturbance compensation
5. Bidirectional message communication
6. Link status management
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fsoc_tracker.simulation.camera.projection import pixel_to_angle
from fsoc_tracker.simulation.camera.state import CameraIntrinsics


class BeaconStatus(Enum):
    UNKNOWN = "unknown"
    DETECTED = "detected"
    SELECTED = "selected"
    ACQUIRING = "acquiring"
    CONNECTED = "connected"
    COMMUNICATING = "communicating"
    LOST = "lost"


class LinkStatus(Enum):
    NO_LINK = "no_link"
    ACQUIRING = "acquiring"
    LOCKED = "locked"
    DEGRADED = "degraded"
    LOST = "lost"


class DisturbanceType(Enum):
    CLEAR = "clear"
    FOG = "fog"
    RAIN = "rain"
    TURBULENCE = "turbulence"
    SMOKE = "smoke"


@dataclass
class BeaconState:
    """Observable beacon state — derived from tracking, NOT ground truth."""
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    velocity_x_px_s: float = 0.0
    velocity_y_px_s: float = 0.0
    uncertainty_x_px: float = 0.0
    uncertainty_y_px: float = 0.0
    confidence: float = 0.0
    angular_error_h_deg: float = 0.0
    angular_error_v_deg: float = 0.0
    # Monocular cameras cannot measure absolute range: always None (N/A).
    # Never fabricate meters from tracking error.
    range_estimate_m: float | None = None
    status: BeaconStatus = BeaconStatus.UNKNOWN
    last_seen_s: float = 0.0


@dataclass
class AtmosphericState:
    """Atmospheric disturbance parameters estimated from image."""
    disturbance_type: DisturbanceType = DisturbanceType.CLEAR
    visibility_km: float = 10.0
    turbulence_cn2: float = 1e-15
    attenuation_db_km: float = 0.0
    beam_wander_px: float = 0.0
    scintillation_index: float = 0.0


@dataclass
class CommunicationMessage:
    """Message for bidirectional communication."""
    message_id: int = 0
    source: str = ""
    destination: str = ""
    payload: str = ""
    timestamp_s: float = 0.0
    status: str = "pending"
    encoding: str = "ascii"


@dataclass
class LinkState:
    """Optical link state — derived from angular error, NOT world positions."""
    status: LinkStatus = LinkStatus.NO_LINK
    angular_error_deg: float = 0.0
    beam_alignment_percent: float = 0.0
    link_quality_percent: float = 0.0
    # Monocular cameras cannot measure absolute range: always None (N/A).
    range_estimate_m: float | None = None
    bandwidth_mbps: float = 0.0


class FSocBeaconAI:
    """Ground-truth-free AI engine for FSOC beacon tracking and communication.

    This module NEVER accesses world-space positions or simulation state.
    All computations are based on:
      - Pixel-space tracking estimates (TrackingState)
      - Camera intrinsics (CameraIntrinsics)
      - Image analysis (optional, for atmospheric estimation)

    The AI provides:
      1. Situation assessment (is tracking healthy?)
      2. Distance/angle estimation (from pixel error + intrinsics)
      3. Predictive lead-angle (from velocity estimates)
      4. Atmospheric compensation (from image statistics)
      5. Link status management
      6. Bidirectional communication
    """

    def __init__(self, terminal_id: str = "TERM_A") -> None:
        self._terminal_id = terminal_id
        self._time_s = 0.0

        self._beacon = BeaconState()
        self._link_state = LinkState()
        self._atmospheric = AtmosphericState()
        self._messages: list[CommunicationMessage] = []
        self._message_counter = 0
        self._connected = False

        # Pixel-space Kalman state for lead-angle prediction
        self._kalman_x = 0.0
        self._kalman_y = 0.0
        self._kalman_vx = 0.0
        self._kalman_vy = 0.0
        self._kalman_initialized = False
        self._kalman_last_time = 0.0

    @property
    def beacon(self) -> BeaconState:
        return self._beacon

    @property
    def link_state(self) -> LinkState:
        return self._link_state

    @property
    def atmospheric(self) -> AtmosphericState:
        return self._atmospheric

    @property
    def messages(self) -> list[CommunicationMessage]:
        return list(self._messages)

    @property
    def connected(self) -> bool:
        return self._connected

    # ================================================================
    # 1. SITUATION ASSESSMENT
    # ================================================================

    def assess_situation(
        self,
        tracking_state: str,
        detected: bool,
        confidence: float,
        residual_px: float,
        uncertainty_x: float,
        uncertainty_y: float,
        velocity_x: float,
        velocity_y: float,
        time_since_detection: float,
    ) -> str:
        """Classify the current tracking situation from observables only."""
        if not detected or tracking_state in ("NO_TRACK", "LOST"):
            return "TARGET_LOST"
        if tracking_state == "SEARCHING":
            return "SEARCHING"
        if tracking_state == "ACQUIRING":
            return "ACQUIRING"
        if confidence < 0.3 or residual_px > 50:
            return "TRACK_DEGRADING"
        if uncertainty_x > 30 or uncertainty_y > 30:
            return "HIGH_UNCERTAINTY"
        speed = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
        if speed > 100:
            return "HIGH_SPEED_MOTION"
        if time_since_detection > 0.5:
            return "STALE_DETECTION"
        return "NORMAL_TRACKING"

    # ================================================================
    # 2. DISTANCE / ANGLE ESTIMATION (from pixel error + intrinsics)
    # ================================================================

    def estimate_angle_from_pixel(
        self, pixel_x: float, pixel_y: float, intrinsics: CameraIntrinsics,
    ) -> tuple[float, float]:
        """Convert pixel position to angular error from boresight (degrees).

        This is the SAME calculation used by the PID controller.
        No ground truth needed — only camera intrinsics.
        """
        return pixel_to_angle(pixel_x, pixel_y, intrinsics)

    def estimate_range_from_context(
        self,
        angular_error_h_deg: float,
        angular_error_v_deg: float,
        confidence: float,
        uncertainty_x: float,
    ) -> float | None:
        """Range is UNKNOWN to a monocular camera — always returns None.

        In a real FSOC system, range would come from a separate sensor
        (LiDAR, rangefinder, or known terminal positions). Fabricating
        meters from tracking error is prohibited: callers must treat
        None as N/A.
        """
        return None

    # ================================================================
    # 3. PREDICTIVE TRACKING (lead-angle from pixel velocity)
    # ================================================================

    def predict_pixel_position(self, horizon_s: float = 0.1) -> tuple[float, float]:
        """Predict where the beacon will be in pixel space.

        Uses a simple constant-velocity model on pixel coordinates.
        The existing KalmanTracker already does this internally,
        but this provides an independent prediction for lead-angle.
        """
        if not self._kalman_initialized:
            return (self._beacon.pixel_x, self._beacon.pixel_y)

        pred_x = self._kalman_x + self._kalman_vx * horizon_s
        pred_y = self._kalman_y + self._kalman_vy * horizon_s
        return (pred_x, pred_y)

    def _update_kalman(self, px: float, py: float, dt: float) -> None:
        """Simple pixel-space velocity estimator (not a full KF — the
        main KalmanTracker already handles that)."""
        if not self._kalman_initialized:
            self._kalman_x = px
            self._kalman_y = py
            self._kalman_vx = 0.0
            self._kalman_vy = 0.0
            self._kalman_initialized = True
            self._kalman_last_time = self._time_s
            return

        if dt <= 0:
            return

        # Exponential moving average for velocity
        alpha = 0.3
        new_vx = (px - self._kalman_x) / dt
        new_vy = (py - self._kalman_y) / dt
        self._kalman_vx = alpha * new_vx + (1 - alpha) * self._kalman_vx
        self._kalman_vy = alpha * new_vy + (1 - alpha) * self._kalman_vy
        self._kalman_x = px
        self._kalman_y = py

    # ================================================================
    # 4. BEAM COMMAND (from pixel error + intrinsics)
    # ================================================================

    def compute_beam_command(
        self,
        tracking_state: str,
        detected: bool,
        estimated_x: float,
        estimated_y: float,
        intrinsics: CameraIntrinsics,
        dt: float,
    ) -> tuple[float, float, bool]:
        """Compute pan/tilt rates to point beam at the beacon.

        Uses ONLY pixel-space estimates + camera intrinsics.
        This is the SAME math as the PID controller — provided here
        for the AI layer to have independent beam control capability.

        Returns: (pan_rate_deg_s, tilt_rate_deg_s, tracking_valid)
        """
        if not detected or tracking_state in ("NO_TRACK", "LOST"):
            return (0.0, 0.0, False)

        # Pixel error from image center
        estimated_x - intrinsics.cx
        estimated_y - intrinsics.cy

        # Convert to angular error
        ang_h, ang_v = pixel_to_angle(estimated_x, estimated_y, intrinsics)

        # PID gains (matching controller config)
        kp = 0.8
        pan_rate = -kp * ang_h  # negative because pan right = target moves left in image
        tilt_rate = kp * ang_v

        # Clamp to max rates (PS spec: 5 deg/s)
        pan_rate = max(-5.0, min(5.0, pan_rate))
        tilt_rate = max(-5.0, min(5.0, tilt_rate))

        # Update link state
        total_error = math.sqrt(ang_h ** 2 + ang_v ** 2)
        self._link_state.angular_error_deg = total_error
        self._link_state.beam_alignment_percent = max(0.0, 100.0 - total_error * 10.0)

        if total_error < 0.5:
            self._link_state.status = LinkStatus.LOCKED
            self._link_state.link_quality_percent = min(100.0, self._link_state.beam_alignment_percent)
        elif total_error < 2.0:
            self._link_state.status = LinkStatus.ACQUIRING
        else:
            self._link_state.status = LinkStatus.NO_LINK

        return (pan_rate, tilt_rate, True)

    # ================================================================
    # 5. ATMOSPHERIC ESTIMATION (from image only)
    # ================================================================

    def estimate_atmosphere(self, image: np.ndarray | None = None) -> AtmosphericState:
        """Estimate atmospheric conditions from image statistics.

        No ground truth needed — purely image-based analysis.
        """
        if image is None:
            self._atmospheric = AtmosphericState()
            return self._atmospheric

        mean_brightness = float(np.mean(image))
        std_brightness = float(np.std(image))

        cn2 = 1e-15 * (std_brightness / 50.0)

        if mean_brightness < 3.0:
            vis = 0.5
            dist_type = DisturbanceType.FOG
        elif mean_brightness < 5.0:
            vis = 2.0
            dist_type = DisturbanceType.SMOKE
        elif std_brightness > 30.0:
            vis = 5.0
            dist_type = DisturbanceType.TURBULENCE
        else:
            vis = 10.0
            dist_type = DisturbanceType.CLEAR

        beam_wander = cn2 * 1e15 * 2.0

        self._atmospheric = AtmosphericState(
            disturbance_type=dist_type,
            visibility_km=vis,
            turbulence_cn2=cn2,
            attenuation_db_km=(10.0 - vis) * 0.5,
            beam_wander_px=beam_wander,
            scintillation_index=std_brightness / 100.0,
        )

        return self._atmospheric

    def compensate_disturbance(self, pan_rate: float, tilt_rate: float) -> tuple[float, float]:
        """Compensate pan/tilt rates for atmospheric disturbances."""
        if self._atmospheric.disturbance_type == DisturbanceType.CLEAR:
            return (pan_rate, tilt_rate)

        if self._atmospheric.disturbance_type == DisturbanceType.TURBULENCE:
            pan_rate *= 0.8
            tilt_rate *= 0.8

        if self._atmospheric.disturbance_type == DisturbanceType.FOG:
            gain_boost = 1.0 + (10.0 - self._atmospheric.visibility_km) * 0.1
            pan_rate *= gain_boost
            tilt_rate *= gain_boost

        return (pan_rate, tilt_rate)

    # ================================================================
    # 6. COMMUNICATION
    # ================================================================

    def send_message(self, destination: str, payload: str) -> CommunicationMessage:
        """Send a message via the optical link."""
        self._message_counter += 1
        msg = CommunicationMessage(
            message_id=self._message_counter,
            source=self._terminal_id,
            destination=destination,
            payload=payload,
            timestamp_s=self._time_s,
            status="sending",
        )

        if self._link_state.status == LinkStatus.LOCKED:
            msg.status = "sent"
        else:
            msg.status = "failed"

        self._messages.append(msg)
        return msg

    def receive_message(self, source: str, payload: str) -> CommunicationMessage:
        """Receive a message from a beacon."""
        self._message_counter += 1
        msg = CommunicationMessage(
            message_id=self._message_counter,
            source=source,
            destination=self._terminal_id,
            payload=payload,
            timestamp_s=self._time_s,
            status="received",
        )
        self._messages.append(msg)
        return msg

    def simulate_beacon_response(self) -> CommunicationMessage | None:
        """Simulate a beacon response (for demo purposes)."""
        if self._link_state.status != LinkStatus.LOCKED:
            return None

        import random
        responses = [
            "ACK: Signal received",
            "STATUS: Operating normally",
            "DATA: Temperature=25C, Humidity=45%",
            "ACK: Message decoded",
            "STATUS: Battery=87%, Signal=GOOD",
        ]
        return self.receive_message("BEACON", random.choice(responses))

    # ================================================================
    # 7. MAIN UPDATE — consumes TrackingState, NOT world targets
    # ================================================================

    def update(
        self,
        dt: float,
        tracking_state: str,
        detected: bool,
        estimated_x: float,
        estimated_y: float,
        velocity_x: float,
        velocity_y: float,
        uncertainty_x: float,
        uncertainty_y: float,
        confidence: float,
        time_since_detection: float,
        intrinsics: CameraIntrinsics,
        image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Main AI update — called every frame.

        ALL inputs are observable quantities from the tracking pipeline.
        NO ground truth is accessed.

        Args:
            dt: Time step (seconds)
            tracking_state: State machine state string
            detected: Whether beacon is currently detected
            estimated_x/y: Kalman-filtered pixel position
            velocity_x/y: Estimated pixel velocity
            uncertainty_x/y: Kalman uncertainty
            confidence: Detection confidence
            time_since_detection: Seconds since last detection
            intrinsics: Camera intrinsic parameters
            image: Optional camera image for atmospheric estimation
        """
        self._time_s += dt

        # Update pixel-space Kalman for prediction
        if detected:
            self._update_kalman(estimated_x, estimated_y, dt)

        # Update beacon state from tracking
        self._beacon.pixel_x = estimated_x
        self._beacon.pixel_y = estimated_y
        self._beacon.velocity_x_px_s = velocity_x
        self._beacon.velocity_y_px_s = velocity_y
        self._beacon.uncertainty_x_px = uncertainty_x
        self._beacon.uncertainty_y_px = uncertainty_y
        self._beacon.confidence = confidence
        self._beacon.last_seen_s = self._time_s

        # Compute angular error from pixel position
        if detected:
            ang_h, ang_v = pixel_to_angle(estimated_x, estimated_y, intrinsics)
            self._beacon.angular_error_h_deg = ang_h
            self._beacon.angular_error_v_deg = ang_v
            self._beacon.range_estimate_m = self.estimate_range_from_context(
                ang_h, ang_v, confidence, uncertainty_x,
            )

        # Assess situation
        situation = self.assess_situation(
            tracking_state, detected, confidence,
            math.sqrt(
                (estimated_x - intrinsics.cx) ** 2 +
                (estimated_y - intrinsics.cy) ** 2
            ),
            uncertainty_x, uncertainty_y,
            velocity_x, velocity_y,
            time_since_detection,
        )

        # Update beacon status
        if situation == "TARGET_LOST":
            self._beacon.status = BeaconStatus.LOST
            self._connected = False
        elif situation == "ACQUIRING":
            self._beacon.status = BeaconStatus.ACQUIRING
        elif situation == "NORMAL_TRACKING":
            if self._beacon.status != BeaconStatus.CONNECTED:
                self._beacon.status = BeaconStatus.CONNECTED
            self._connected = True
        elif detected:
            self._beacon.status = BeaconStatus.DETECTED

        # Estimate atmosphere from image
        self.estimate_atmosphere(image)

        # Update link state
        if detected and self._beacon.status == BeaconStatus.CONNECTED:
            self._link_state.range_estimate_m = self._beacon.range_estimate_m
        elif not detected:
            self._link_state.status = LinkStatus.LOST
            self._connected = False

        return {
            "situation": situation,
            "beacon_status": self._beacon.status.value,
            "link_status": self._link_state.status.value,
            "link_quality": self._link_state.link_quality_percent,
            "angular_error_h": self._beacon.angular_error_h_deg,
            "angular_error_v": self._beacon.angular_error_v_deg,
            "range_estimate_m": self._beacon.range_estimate_m,
            "atmospheric": self._atmospheric.disturbance_type.value,
            "connected": self._connected,
            "messages_sent": sum(1 for m in self._messages if m.status == "sent"),
            "messages_received": sum(1 for m in self._messages if m.status == "received"),
        }
