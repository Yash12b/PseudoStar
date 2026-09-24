"""Integration tests for the full AI intelligence layer."""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.ai.adaptive_roi import AdaptiveROI
from fsoc_tracker.ai.explain import ExplainabilityEngine
from fsoc_tracker.ai.failure_predictor import FailurePredictor
from fsoc_tracker.ai.mission import (
    AIMissionBrain,
    MissionAction,
    MissionObservation,
    ObservationFeatures,
    Situation,
)
from fsoc_tracker.control.adaptive import AdaptiveController
from fsoc_tracker.pipeline.pipeline import PipelineFrameResult, TrackingPipeline
from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
from fsoc_tracker.tracking.search import SearchController


class TestMissionBrainIntegration:
    def test_decide_returns_valid(self):
        brain = AIMissionBrain()
        features = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.8,
            residual_px=3.0, uncertainty_x_px=2.0, uncertainty_y_px=2.0,
            velocity_x_px_s=10.0, velocity_y_px_s=-5.0,
            distance_from_center_px=30.0, time_since_detection_s=0.0,
            latency_ms=2.0, source_fps=30.0, processing_fps=25.0,
            candidate_count=2, roi_radius_px=80.0,
        )
        obs = MissionObservation(features=features)
        decision = brain.decide(obs)
        assert isinstance(decision.action, MissionAction)
        assert isinstance(decision.situation, Situation)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.safety is not None

    def test_safety_always_present(self):
        brain = AIMissionBrain()
        features = ObservationFeatures(
            timestamp_s=1.0, detected=False, confidence=0.0,
            residual_px=0.0, uncertainty_x_px=10.0, uncertainty_y_px=10.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=200.0, time_since_detection_s=0.5,
            latency_ms=2.0, source_fps=30.0, processing_fps=25.0,
            candidate_count=0, roi_radius_px=80.0,
        )
        obs = MissionObservation(features=features)
        decision = brain.decide(obs)
        assert decision.safety is not None
        assert isinstance(decision.safety.approved, bool)


class TestPipelineAIIntegration:
    def test_pipeline_with_ai_modules(self):
        pipeline = TrackingPipeline(
            ai_brain=AIMissionBrain(),
            adaptive_roi=AdaptiveROI(),
            failure_predictor=FailurePredictor(),
            adaptive_kalman=AdaptiveKalmanManager(),
            adaptive_controller=AdaptiveController(),
            search_controller=SearchController(),
        )
        assert pipeline._ai_brain is not None
        assert pipeline._adaptive_roi is not None
        assert pipeline._failure_predictor is not None
        assert pipeline._adaptive_kalman is not None
        assert pipeline._adaptive_controller is not None
        assert pipeline._search_controller is not None

    def test_pipeline_backward_compatible(self):
        pipeline = TrackingPipeline()
        assert pipeline._ai_brain is None
        assert pipeline._adaptive_roi is None
        assert pipeline._failure_predictor is None

    def test_pipeline_reset_clears_window(self):
        pipeline = TrackingPipeline(ai_brain=AIMissionBrain())
        pipeline._feature_window.append("test")
        pipeline.reset()
        assert len(pipeline._feature_window) == 0

    def test_pipeline_frame_result_has_ai_fields(self):
        result = PipelineFrameResult()
        assert hasattr(result, "ai_decision")
        assert hasattr(result, "ai_features")
        assert hasattr(result, "ai_failure_risk")
        assert hasattr(result, "ai_ms")
        assert hasattr(result, "roi_rect")


class TestExplainabilityIntegration:
    def test_explain_decision(self):
        brain = AIMissionBrain()
        explainer = ExplainabilityEngine()
        features = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.3,
            residual_px=20.0, uncertainty_x_px=12.0, uncertainty_y_px=8.0,
            velocity_x_px_s=50.0, velocity_y_px_s=-20.0,
            distance_from_center_px=100.0, time_since_detection_s=0.0,
            latency_ms=2.0, source_fps=30.0, processing_fps=20.0,
            candidate_count=1, roi_radius_px=80.0,
        )
        obs = MissionObservation(features=features)
        decision = brain.decide(obs)
        explanation = explainer.explain(features, decision, failure_risk=0.3)
        assert explanation.summary
        assert len(explanation.feature_contributions) > 0
        assert explanation.situation in [s.value for s in Situation]
        assert explanation.action in [a.value for a in MissionAction]


class TestAdaptiveKalmanIntegration:
    def test_update_produces_scales(self):
        ak = AdaptiveKalmanManager()
        q, r = ak.update()
        assert q == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_apply_to_matrices(self):
        ak = AdaptiveKalmanManager()
        Q = np.eye(4)
        R = np.eye(2)
        Q_s, R_s = ak.apply_to_matrices(Q, R)
        assert Q_s.shape == (4, 4)
        assert R_s.shape == (2, 2)


class TestAdaptiveControllerIntegration:
    def test_compute_adaptation(self):
        ac = AdaptiveController()
        telemetry = ac.compute_adaptation(
            error_x=25.0, error_y=-10.0,
            velocity_x=30.0, velocity_y=-5.0,
        )
        assert telemetry.adaptive_active is True
        assert 0.3 <= telemetry.kp_scale_pan <= 2.0


class TestSearchControllerIntegration:
    def test_update(self):
        sc = SearchController()
        state = sc.update(0.033)
        assert state.frames_in_search == 1
