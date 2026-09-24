"""Tracking subsystem — temporal target tracking and state estimation.

Provides the KalmanTracker, Kalman filter, state machine,
association, configuration, events, and debug visualization.

The tracker consumes Detection objects from any perception backend
and produces TrackingState for the downstream controller.
"""

from fsoc_tracker.tracking.association import associate_nearest, euclidean_distance
from fsoc_tracker.tracking.config import (
    AssociationMethod,
    FilterType,
    TimestampGapPolicy,
    TrackerConfig,
)
from fsoc_tracker.tracking.events import TrackingMetrics, TrackingMetricsCollector
from fsoc_tracker.tracking.identity import (
    CODE_A,
    CODE_B,
    CodeIdentityTracker,
    code_bit_at,
)
from fsoc_tracker.tracking.kalman import KalmanFilter2D
from fsoc_tracker.tracking.state import (
    TrackEvent,
    TrackingEvent,
    TrackingState,
    TrackState,
)
from fsoc_tracker.tracking.state_machine import TrackStateMachine
from fsoc_tracker.tracking.tracker import KalmanTracker
from fsoc_tracker.tracking.visualization import render_tracking_debug

__all__ = [
    "AssociationMethod",
    "CODE_A",
    "CODE_B",
    "CodeIdentityTracker",
    "FilterType",
    "KalmanFilter2D",
    "KalmanTracker",
    "TimestampGapPolicy",
    "TrackerConfig",
    "TrackEvent",
    "TrackStateMachine",
    "TrackState",
    "TrackingEvent",
    "TrackingMetrics",
    "TrackingMetricsCollector",
    "TrackingState",
    "associate_nearest",
    "code_bit_at",
    "euclidean_distance",
    "render_tracking_debug",
]
