"""Tracking state model.

Rich, strongly typed state representing everything the tracker
knows about a tracked target at a given moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TrackState(Enum):
    """Explicit finite-state-machine states for a track."""
    NO_TRACK = auto()
    SEARCHING = auto()
    ACQUIRING = auto()
    TRACKING = auto()
    LOST = auto()
    REACQUIRING = auto()


class TrackEvent(Enum):
    """Timestamped events for performance logging."""
    TRACK_INITIALIZED = auto()
    TRACK_ACQUIRED = auto()
    TRACK_UPDATED = auto()
    TRACK_PREDICTED = auto()
    TRACK_LOST = auto()
    TRACK_REACQUIRED = auto()
    ASSOCIATION_FAILED = auto()
    GATE_REJECTED = auto()


@dataclass
class TrackingEvent:
    """A single timestamped tracking event."""
    event: TrackEvent
    timestamp_s: float
    track_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingState:
    """Complete output of the tracking subsystem for one frame.

    All positions are in image pixel coordinates.
    Velocity is in px/s (FPS-independent).
    """

    state: TrackState = TrackState.NO_TRACK
    track_id: int = -1
    timestamp_s: float = 0.0

    # Estimated position (from Kalman filter or last detection)
    estimated_x: float = 0.0
    estimated_y: float = 0.0

    # Velocity (px/s)
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Predicted position (one-step-ahead prediction)
    predicted_x: float = 0.0
    predicted_y: float = 0.0

    # Raw detection that was associated (if any)
    detection_x: float = 0.0
    detection_y: float = 0.0
    detection_confidence: float = 0.0
    has_detection: bool = False

    # Uncertainty
    uncertainty_x: float = 0.0
    uncertainty_y: float = 0.0

    # Innovation / residual
    residual_x: float = 0.0
    residual_y: float = 0.0
    residual_magnitude: float = 0.0
    mahalanobis_distance: float = 0.0

    # Track quality
    quality: float = 0.0
    locked: bool = False

    # Timing
    track_age_s: float = 0.0
    detection_age_s: float = 0.0
    consecutive_detections: int = 0
    consecutive_misses: int = 0
    time_since_last_detection_s: float = 0.0

    # Acquisition / reacquisition
    acquisition_start_time_s: float = 0.0
    acquisition_success_time_s: float = 0.0
    acquisition_time_s: float = 0.0
    loss_time_s: float = 0.0
    reacquisition_time_s: float = 0.0
    reacquisition_duration_s: float = 0.0

    # Prediction-only flag
    prediction_only: bool = False

    # Set when this frame's update came from the rescue pass
    rescued_by_low_conf: bool = False

    # Coded-identity match fraction of the associated detection vs the
    # operator-configured expected code (0.5 = no identity evidence)
    identity_match: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "estimated_x": self.estimated_x,
            "estimated_y": self.estimated_y,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "predicted_x": self.predicted_x,
            "predicted_y": self.predicted_y,
            "has_detection": self.has_detection,
            "detection_confidence": self.detection_confidence,
            "uncertainty_x": self.uncertainty_x,
            "uncertainty_y": self.uncertainty_y,
            "residual_x": self.residual_x,
            "residual_y": self.residual_y,
            "residual_magnitude": self.residual_magnitude,
            "mahalanobis_distance": self.mahalanobis_distance,
            "quality": self.quality,
            "locked": self.locked,
            "track_age_s": self.track_age_s,
            "detection_age_s": self.detection_age_s,
            "consecutive_detections": self.consecutive_detections,
            "consecutive_misses": self.consecutive_misses,
            "time_since_last_detection_s": self.time_since_last_detection_s,
            "acquisition_time_s": self.acquisition_time_s,
            "reacquisition_duration_s": self.reacquisition_duration_s,
            "prediction_only": self.prediction_only,
            "identity_match": self.identity_match,
        }
