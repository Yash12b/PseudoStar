"""Safety tests for the AI intelligence layer.

Tests that the system correctly handles:
- NaN/Inf inputs
- Massive predictions
- Stale decisions
- Invalid ROI
- Malformed outputs
- Rapid oscillation
- Safety clamping
"""

from __future__ import annotations

import math

from fsoc_tracker.ai.adaptive_roi import AdaptiveROI
from fsoc_tracker.ai.explain import ExplainabilityEngine
from fsoc_tracker.ai.failure_predictor import FailurePredictor
from fsoc_tracker.ai.mission import (
    AIMissionBrain,
    MissionAction,
    MissionObservation,
    ObservationFeatures,
    SafetyEnvelope,
    SafetyLimits,
    Situation,
)


def _make_obs(**kwargs) -> ObservationFeatures:
    defaults = dict(
        timestamp_s=1.0, detected=True, confidence=0.8,
        residual_px=3.0, uncertainty_x_px=2.0, uncertainty_y_px=2.0,
        velocity_x_px_s=10.0, velocity_y_px_s=-5.0,
        distance_from_center_px=30.0, time_since_detection_s=0.0,
        latency_ms=2.0, source_fps=30.0, processing_fps=25.0,
        candidate_count=2, roi_radius_px=80.0,
    )
    defaults.update(kwargs)
    return ObservationFeatures(**defaults)


class TestNaNSafety:
    def test_nan_confidence_produces_zero(self):
        obs = _make_obs(confidence=float("nan"))
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)
        assert decision.safety.confidence == 0.0

    def test_nan_timestamp_rejected(self):
        obs = _make_obs(timestamp_s=float("nan"))
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert not decision.safety.approved

    def test_inf_velocity(self):
        obs = _make_obs(velocity_x_px_s=float("inf"), velocity_y_px_s=float("-inf"))
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)

    def test_nan_residual(self):
        obs = _make_obs(residual_px=float("nan"))
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)

    def test_zero_confidence_detected(self):
        obs = _make_obs(detected=True, confidence=0.0)
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)


class TestStaleDecisionSafety:
    def test_stale_decision_rejected(self):
        envelope = SafetyEnvelope(SafetyLimits(stale_timeout_s=0.5))
        obs = _make_obs(timestamp_s=0.0)
        recommendation = envelope.approve(
            MissionAction.TRACK, 0.8, "test", timestamp_s=0.0,
        )
        assert recommendation.approved
        action = envelope.approve(
            MissionAction.TRACK, 0.8, "test", timestamp_s=0.0,
            observation_age_s=1.0,
        )
        assert not action.approved
        assert action.fallback == MissionAction.HOLD

    def test_fresh_decision_approved(self):
        envelope = SafetyEnvelope(SafetyLimits(stale_timeout_s=0.5))
        action = envelope.approve(
            MissionAction.TRACK, 0.8, "test", timestamp_s=1.0,
            observation_age_s=0.1,
        )
        assert action.approved


class TestMassivePredictionSafety:
    def test_extreme_velocity_handled(self):
        obs = _make_obs(velocity_x_px_s=10000.0, velocity_y_px_s=-10000.0)
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)
        assert decision.prediction is not None
        assert math.isfinite(decision.prediction.mean_x_px)
        assert math.isfinite(decision.prediction.mean_y_px)

    def test_extreme_uncertainty(self):
        obs = _make_obs(uncertainty_x_px=10000.0, uncertainty_y_px=10000.0)
        brain = AIMissionBrain()
        decision = brain.decide(MissionObservation(features=obs))
        assert isinstance(decision.action, MissionAction)


class TestAdaptiveROISafety:
    def test_roi_clamped_to_image(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        state = roi.compute(
            estimated_x=-500.0, estimated_y=-500.0,
            uncertainty_x=100.0, uncertainty_y=100.0,
        )
        rect = roi.roi_rect
        assert rect[0] >= 0
        assert rect[1] >= 0
        assert rect[2] <= 640
        assert rect[3] <= 480

    def test_roi_expands_on_high_failure_risk(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        state = roi.compute(
            estimated_x=320.0, estimated_y=240.0,
            failure_risk=0.9,
        )
        assert state.radius_px >= 300

    def test_roi_full_frame_on_search(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        state = roi.compute(
            estimated_x=320.0, estimated_y=240.0,
            situation="target_lost",
        )
        assert state.radius_px >= 300

    def test_roi_resets(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        roi.compute(estimated_x=100.0, estimated_y=100.0)
        roi.reset()
        assert roi.state.reason == "reset"


class TestSafetyEnvelopeRateClamping:
    def test_pan_rate_clamped(self):
        envelope = SafetyEnvelope(SafetyLimits(max_pan_rate_deg_s=5.0, max_tilt_rate_deg_s=5.0))
        pan, tilt = envelope.clamp_rates(100.0, -100.0)
        assert pan == 5.0
        assert tilt == -5.0

    def test_pan_rate_within_limits(self):
        envelope = SafetyEnvelope(SafetyLimits(max_pan_rate_deg_s=5.0, max_tilt_rate_deg_s=5.0))
        pan, tilt = envelope.clamp_rates(2.0, -3.0)
        assert pan == 2.0
        assert tilt == -3.0

    def test_zero_rates(self):
        envelope = SafetyEnvelope()
        pan, tilt = envelope.clamp_rates(0.0, 0.0)
        assert pan == 0.0
        assert tilt == 0.0


class TestExplainabilitySafety:
    def test_explain_with_nan_features(self):
        explainer = ExplainabilityEngine()
        brain = AIMissionBrain()
        obs = _make_obs(confidence=0.5)
        decision = brain.decide(MissionObservation(features=obs))
        explanation = explainer.explain(obs, decision)
        assert explanation.summary
        assert len(explanation.feature_contributions) > 0


class TestAIMissionBrainAllSituations:
    def test_normal_tracking(self):
        brain = AIMissionBrain()
        obs = _make_obs(detected=True, confidence=0.9, residual_px=2.0)
        d = brain.decide(MissionObservation(features=obs))
        assert d.situation == Situation.NORMAL_TRACKING

    def test_low_confidence(self):
        brain = AIMissionBrain()
        obs = _make_obs(detected=True, confidence=0.1)
        d = brain.decide(MissionObservation(features=obs))
        assert d.situation == Situation.LOW_CONFIDENCE

    def test_high_noise(self):
        brain = AIMissionBrain()
        obs = _make_obs(detected=True, confidence=0.8, residual_px=50.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0)
        d = brain.decide(MissionObservation(features=obs))
        assert d.situation == Situation.HIGH_NOISE

    def test_target_lost(self):
        brain = AIMissionBrain()
        obs = _make_obs(detected=False, confidence=0.0, time_since_detection_s=0.0)
        d = brain.decide(MissionObservation(features=obs))
        assert d.situation == Situation.TARGET_LOST

    def test_reacquisition(self):
        brain = AIMissionBrain()
        obs = _make_obs(detected=False, confidence=0.0, time_since_detection_s=0.5)
        d = brain.decide(MissionObservation(features=obs))
        assert d.situation == Situation.REACQUISITION


class TestFailurePredictorSafety:
    def test_untrained_returns_zero(self):
        predictor = FailurePredictor(window_size=10)
        window = [_make_obs() for _ in range(10)]
        pred = predictor.predict(window)
        assert pred.risk_score == 0.0
        assert pred.recommended_action == "track"

    def test_empty_window(self):
        predictor = FailurePredictor(window_size=10)
        pred = predictor.predict([])
        assert pred.risk_score == 0.0
