from __future__ import annotations

import json

from fsoc_tracker.ai.mission import (
    AIMissionBrain,
    DecisionLogger,
    ExpertPolicy,
    MissionAction,
    MissionDecisionExecutor,
    MissionObservation,
    MotionPredictor,
    ObservationFeatures,
    SafetyEnvelope,
    Situation,
    SituationClassifier,
)


def _features(**overrides):
    values = dict(
        timestamp_s=1.0,
        detected=True,
        confidence=0.9,
        residual_px=1.0,
        uncertainty_x_px=2.0,
        uncertainty_y_px=2.0,
        velocity_x_px_s=10.0,
        velocity_y_px_s=-5.0,
        distance_from_center_px=3.0,
        time_since_detection_s=0.0,
        latency_ms=2.0,
        source_fps=30.0,
        processing_fps=30.0,
        candidate_count=1,
    )
    values.update(overrides)
    return ObservationFeatures(**values)


def test_motion_prediction_is_uncertainty_aware():
    prediction = MotionPredictor().predict(_features(), 0.1)
    assert prediction.mean_x_px == 1.0
    assert prediction.mean_y_px == -0.5
    assert prediction.uncertainty_x_px > 2.0


def test_classifier_uses_observable_features_only():
    assert SituationClassifier().classify(_features(detected=False)) == Situation.TARGET_LOST
    assert SituationClassifier().classify(
        _features(detected=False, time_since_detection_s=0.2)
    ) == Situation.REACQUISITION
    assert SituationClassifier().classify(
        _features(confidence=0.4, time_since_detection_s=0.2)
    ) == Situation.DEGRADING_TRACK


def test_policy_is_deterministic_and_safe():
    decision = ExpertPolicy().decide(_features(detected=False))
    assert decision.action == MissionAction.LOCAL_SEARCH
    assert decision.approved

    stale = SafetyEnvelope().approve(
        MissionAction.GLOBAL_SEARCH, 0.9, "test", 1.0, observation_age_s=1.0
    )
    assert stale.action == MissionAction.HOLD
    assert not stale.approved


def test_safety_clamps_rates():
    assert SafetyEnvelope().clamp_rates(99.0, -99.0) == (5.0, -5.0)


def test_decision_log_is_bounded_and_serializable(tmp_path):
    path = tmp_path / "ai_decisions.jsonl"
    logger = DecisionLogger(path, max_records=1)
    features = _features()
    decision = ExpertPolicy().decide(features)
    logger.log(features, decision)
    logger.log(features, decision)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["decision"]["action"] == decision.action.value


def test_orchestrator_predicts_without_mutating_observation():
    observation = MissionObservation(_features())
    decision = AIMissionBrain().decide(observation)
    assert decision.prediction is not None
    assert decision.action == MissionAction.USE_ROI
    assert MissionDecisionExecutor().validate(decision, now_s=1.1) == MissionAction.USE_ROI


def test_executor_rejects_stale_decisions():
    decision = AIMissionBrain().decide(MissionObservation(_features()))
    assert MissionDecisionExecutor(stale_timeout_s=0.05).validate(
        decision, now_s=1.1
    ) == MissionAction.HOLD
