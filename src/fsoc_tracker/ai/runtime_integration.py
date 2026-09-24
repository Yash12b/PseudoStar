"""AI advisory types — decisions, safety gate, provenance.

The single production strategy path is ``ai.mission.AIMissionBrain``
(worker-proven, HUD-mapped). This module keeps the shared advisory
types: ``AIDecision`` (prediction + assessment + safety result) and
``SafetyGate`` (bounded-offset policy). The former ``AIIntegration``
orchestrator was removed: its output contract referenced an
unimplemented ``controller.adjust_offset``, so no production path
could consume it.

Classical/Kalman path is ALWAYS the production control loop.
AI is advisory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fsoc_tracker.ai.runtime_features import ObservableFeatures
from fsoc_tracker.ai.runtime_predictor import TemporalPrediction


@dataclass
class AIDecision:
    """Complete AI output for one frame."""
    # Temporal predictions
    prediction: TemporalPrediction = field(default_factory=TemporalPrediction)

    # Observable features used
    features: ObservableFeatures = field(default_factory=ObservableFeatures)

    # Situation assessment
    situation: str = "tracking"
    failure_risk: float = 0.0
    is_confident: bool = True

    # Safety gate result
    safety_passed: bool = True
    safety_block_reason: str = ""

    # Lead-angle offset (pixel space) — small correction
    lead_angle_dx: float = 0.0
    lead_angle_dy: float = 0.0

    # Provenance
    provenance: dict[str, Any] = field(default_factory=dict)

    # Timing
    inference_time_ms: float = 0.0


class SafetyGate:
    """Validates AI predictions before they reach the controller.

    Safety ALWAYS dominates AI. If AI output is suspicious, it is blocked.
    """

    def __init__(
        self,
        max_lead_angle_px: float = 10.0,
        max_prediction_displacement_px: float = 200.0,
        min_confidence: float = 0.3,
        max_failure_risk: float = 0.8,
    ) -> None:
        self._max_lead = max_lead_angle_px
        self._max_disp = max_prediction_displacement_px
        self._min_conf = min_confidence
        self._max_risk = max_failure_risk

    def check(
        self,
        features: ObservableFeatures,
        prediction: TemporalPrediction,
    ) -> tuple[bool, str]:
        """Check if AI output is safe to apply.

        Returns (passed, reason).
        """
        # Block if tracking is lost
        if features.track_state == "NO_TRACK":
            return False, "tracking_lost"

        # Block if AI has no confidence
        if not prediction.valid:
            return False, "prediction_invalid"

        # Block if prediction displacement is unreasonably large
        total_dx, total_dy = prediction.get_total_displacement()
        if abs(total_dx) > self._max_disp or abs(total_dy) > self._max_disp:
            return False, f"displacement_too_large ({total_dx:.1f}, {total_dy:.1f})"

        # Block if failure risk is too high
        if features.failure_risk > self._max_risk:
            return False, f"failure_risk_high ({features.failure_risk:.2f})"

        # Block if FOV margin is critically low and AI wants to push further out
        if features.fov_margin_x < 0.05:
            if abs(total_dx) > 5.0:
                return False, "fov_margin_critical_x"
        if features.fov_margin_y < 0.05:
            if abs(total_dy) > 5.0:
                return False, "fov_margin_critical_y"

        return True, ""
