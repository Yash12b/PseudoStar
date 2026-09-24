"""Tests for AI integration into the tracking pipeline.

Validates:
- AI receives only observable features (no GT leakage)
- Temporal predictions appear in runtime
- Constant velocity fallback always works
- GRU graceful fallback when unavailable
- Safety gate blocks dangerous outputs
- Model provenance tracked correctly
- AI failure does not crash tracking
"""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.ai.runtime_features import (
    ObservableFeatures,
    RuntimeFeatureExtractor,
)
from fsoc_tracker.ai.runtime_integration import (
    SafetyGate,
)
from fsoc_tracker.ai.runtime_predictor import (
    HORIZONS_S,
    ConstantVelocityPredictor,
    GRUPredictor,
    RuntimePredictor,
)
from fsoc_tracker.simulation.camera.state import CameraIntrinsics
from fsoc_tracker.tracking.state import TrackingState, TrackState


def _make_state(
    x: float = 320.0,
    y: float = 240.0,
    vx: float = 50.0,
    vy: float = 0.0,
    detected: bool = True,
    confidence: float = 0.9,
    state: TrackState = TrackState.TRACKING,
    uncertainty_x: float = 5.0,
    uncertainty_y: float = 5.0,
    residual_x: float = 2.0,
    residual_y: float = 1.0,
    detection_age_s: float = 0.0,
    consecutive_misses: int = 0,
    consecutive_detections: int = 10,
) -> TrackingState:
    return TrackingState(
        estimated_x=x,
        estimated_y=y,
        velocity_x=vx,
        velocity_y=vy,
        has_detection=detected,
        detection_confidence=confidence,
        state=state,
        uncertainty_x=uncertainty_x,
        uncertainty_y=uncertainty_y,
        residual_x=residual_x,
        residual_y=residual_y,
        time_since_last_detection_s=detection_age_s,
        consecutive_misses=consecutive_misses,
        consecutive_detections=consecutive_detections,
    )


def _make_intrinsics(
    fov_h_deg: float = 4.0,
    fov_v_deg: float = 3.0,
    width: int = 640,
    height: int = 480,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        horizontal_fov_deg=fov_h_deg,
        vertical_fov_deg=fov_v_deg,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# ConstantVelocityPredictor tests
# ---------------------------------------------------------------------------

class TestConstantVelocityPredictor:
    def test_returns_all_horizons(self):
        cv = ConstantVelocityPredictor()
        pred = cv.predict(320, 240, 50, 0, 5, 5)
        assert len(pred.predictions) == len(HORIZONS_S)
        for p, h in zip(pred.predictions, HORIZONS_S, strict=True):
            assert p.horizon_s == h

    def test_zero_velocity(self):
        cv = ConstantVelocityPredictor()
        pred = cv.predict(320, 240, 0, 0, 5, 5)
        for p in pred.predictions:
            assert p.displacement_x_px == pytest.approx(0.0)
            assert p.displacement_y_px == pytest.approx(0.0)

    def test_linear_displacement(self):
        cv = ConstantVelocityPredictor()
        pred = cv.predict(320, 240, 100, 50, 5, 5)
        # 100ms horizon → dx = 100*0.1 = 10, dy = 50*0.1 = 5
        dx, dy = pred.get_displacement(0.1)
        assert dx == pytest.approx(10.0)
        assert dy == pytest.approx(5.0)

    def test_uncertainty_grows(self):
        cv = ConstantVelocityPredictor()
        pred = cv.predict(320, 240, 50, 0, 5, 5)
        u0 = pred.predictions[0].uncertainty_x_px
        u1 = pred.predictions[-1].uncertainty_x_px
        assert u1 > u0

    def test_provenance(self):
        cv = ConstantVelocityPredictor()
        assert cv.provenance.model_type == "constant_velocity"
        # Analytical deterministic baseline: never "trained".
        assert cv.provenance.trained is False

    def test_valid_result(self):
        cv = ConstantVelocityPredictor()
        pred = cv.predict(320, 240, 50, 0, 5, 5)
        assert pred.valid is True
        assert pred.fallback is False


# ---------------------------------------------------------------------------
# RuntimePredictor tests
# ---------------------------------------------------------------------------

class TestRuntimePredictor:
    def test_always_produces_prediction(self):
        rp = RuntimePredictor()
        pred = rp.predict(320, 240, 50, 0, 5, 5)
        assert pred.valid is True
        assert len(pred.predictions) == len(HORIZONS_S)

    def test_gru_not_available(self):
        rp = RuntimePredictor()
        assert rp.gru_available is False

    def test_gru_fallback_to_cv(self):
        rp = RuntimePredictor()
        # GRU not loaded, should use CV
        pred = rp.predict(320, 240, 50, 0, 5, 5)
        assert pred.method == "constant_velocity"

    def test_provenance_includes_both(self):
        rp = RuntimePredictor()
        prov = rp.provenance
        assert "cv" in prov
        assert "gru" in prov
        assert "active_method" in prov

    def test_interpolation(self):
        rp = RuntimePredictor()
        pred = rp.predict(320, 240, 100, 0, 5, 5)
        dx, dy = pred.get_displacement(0.075)  # Between 0.05 and 0.1
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_total_displacement(self):
        rp = RuntimePredictor()
        pred = rp.predict(320, 240, 100, 50, 5, 5)
        dx, dy = pred.get_total_displacement()
        assert dx > 0
        assert dy > 0


# ---------------------------------------------------------------------------
# ObservableFeatures tests
# ---------------------------------------------------------------------------

class TestObservableFeatures:
    def test_feature_vector_size(self):
        f = ObservableFeatures()
        vec = f.to_feature_vector()
        assert vec.shape == (15,)

    def test_no_gt_in_features(self):
        f = ObservableFeatures(
            estimated_x=320, estimated_y=240,
            velocity_x=50, velocity_y=0,
            has_detection=True, detection_confidence=0.9,
        )
        vec = f.to_feature_vector()
        # Should be finite
        assert np.all(np.isfinite(vec))

    def test_failure_risk_low_when_tracked(self):
        f = ObservableFeatures(
            has_detection=True,
            detection_confidence=0.9,
            detection_age_s=0.0,
            fov_margin_x=0.5,
            fov_margin_y=0.5,
            residual_magnitude=5.0,
        )
        assert f.failure_risk < 0.2

    def test_failure_risk_high_when_lost(self):
        f = ObservableFeatures(
            has_detection=False,
            detection_age_s=1.0,
            fov_margin_x=0.01,
            residual_magnitude=100.0,
        )
        assert f.failure_risk > 0.5


# ---------------------------------------------------------------------------
# RuntimeFeatureExtractor tests
# ---------------------------------------------------------------------------

class TestRuntimeFeatureExtractor:
    def test_extracts_from_tracking_state(self):
        ext = RuntimeFeatureExtractor(640, 480)
        state = _make_state(x=100, y=200, detected=True)
        features = ext.extract(state, timestamp_s=1.0, dt=0.033)
        assert features.estimated_x == 100.0
        assert features.estimated_y == 200.0
        assert features.has_detection is True

    def test_fov_margin(self):
        ext = RuntimeFeatureExtractor(640, 480)
        state = _make_state(x=320, y=240)  # center
        features = ext.extract(state, timestamp_s=1.0, dt=0.033)
        assert features.fov_margin_x == pytest.approx(1.0, abs=0.01)
        assert features.fov_margin_y == pytest.approx(1.0, abs=0.01)

    def test_fov_margin_at_edge(self):
        ext = RuntimeFeatureExtractor(640, 480)
        state = _make_state(x=0, y=0)  # top-left
        features = ext.extract(state, timestamp_s=1.0, dt=0.033)
        assert features.fov_margin_x == pytest.approx(0.0, abs=0.01)
        assert features.fov_margin_y == pytest.approx(0.0, abs=0.01)

    def test_history_accumulates(self):
        ext = RuntimeFeatureExtractor(640, 480)
        for i in range(5):
            state = _make_state(x=100 + i * 10, y=200)
            ext.extract(state, timestamp_s=float(i), dt=0.033)
        state = _make_state(x=150, y=200)
        features = ext.extract(state, timestamp_s=5.0, dt=0.033)
        assert len(features.position_history) == 6

    def test_reset_clears_history(self):
        ext = RuntimeFeatureExtractor(640, 480)
        state = _make_state()
        ext.extract(state, timestamp_s=1.0, dt=0.033)
        ext.reset()
        assert len(ext._position_history) == 0


# ---------------------------------------------------------------------------
# SafetyGate tests
# ---------------------------------------------------------------------------

class TestSafetyGate:
    def test_blocks_when_lost(self):
        gate = SafetyGate()
        f = ObservableFeatures(track_state="NO_TRACK")
        pred = RuntimePredictor().predict(320, 240, 50, 0, 5, 5)
        passed, reason = gate.check(f, pred)
        assert passed is False
        assert "lost" in reason

    def test_blocks_when_prediction_too_large(self):
        gate = SafetyGate(max_prediction_displacement_px=100)
        f = ObservableFeatures(track_state="TRACKING")
        from fsoc_tracker.ai.runtime_predictor import HorizonPrediction, TemporalPrediction
        pred = TemporalPrediction(
            predictions=[HorizonPrediction(
                horizon_s=0.5, displacement_x_px=500, displacement_y_px=0,
                uncertainty_x_px=10.0, uncertainty_y_px=10.0,
            )],
            valid=True,
        )
        passed, reason = gate.check(f, pred)
        assert passed is False

    def test_blocks_when_risk_high(self):
        gate = SafetyGate(max_failure_risk=0.5)
        f = ObservableFeatures(
            track_state="TRACKING",
            has_detection=False,
            detection_age_s=1.0,
            fov_margin_x=0.01,
            residual_magnitude=100.0,
        )
        pred = RuntimePredictor().predict(320, 240, 0, 0, 5, 5)
        passed, reason = gate.check(f, pred)
        assert passed is False

    def test_passes_when_healthy(self):
        gate = SafetyGate()
        f = ObservableFeatures(
            track_state="TRACKING",
            has_detection=True,
            detection_confidence=0.9,
            fov_margin_x=0.5,
            fov_margin_y=0.5,
            residual_magnitude=5.0,
        )
        pred = RuntimePredictor().predict(320, 240, 50, 0, 5, 5)
        passed, reason = gate.check(f, pred)
        assert passed is True
        assert reason == ""

    def test_blocks_fov_margin_critical(self):
        gate = SafetyGate()
        f = ObservableFeatures(
            track_state="TRACKING",
            has_detection=True,
            fov_margin_x=0.01,  # critical
            fov_margin_y=0.5,
        )
        from fsoc_tracker.ai.runtime_predictor import HorizonPrediction, TemporalPrediction
        pred = TemporalPrediction(
            predictions=[HorizonPrediction(
                horizon_s=0.1, displacement_x_px=20, displacement_y_px=0,
                uncertainty_x_px=10.0, uncertainty_y_px=10.0,
            )],
            valid=True,
        )
        passed, reason = gate.check(f, pred)
        assert passed is False


# ---------------------------------------------------------------------------
# NOTE: AIIntegration (the orchestrator class) was removed — its output
# contract referenced an unimplemented controller.adjust_offset, so no
# production path could consume it. The worker-proven AIMissionBrain is
# the single strategy path. Equivalent single-path coverage lives in
# test_ai_mission.py; shared types (SafetyGate, predictors, features)
# remain tested below and in TestGTBoundary.


# ---------------------------------------------------------------------------
# GT boundary tests — AI NEVER receives ground truth
# ---------------------------------------------------------------------------

class TestGTBoundary:
    """Ensure AI module never imports or accesses GT classes."""

    def test_runtime_predictor_no_gt_imports(self):
        import inspect
        source = inspect.getsource(RuntimePredictor)
        assert "WorldTruth" not in source
        assert "ground_truth" not in source.lower()
        assert "true_" not in source

    def test_runtime_features_no_gt_imports(self):
        import inspect
        source = inspect.getsource(RuntimeFeatureExtractor)
        assert "WorldTruth" not in source
        assert "ground_truth" not in source.lower()

    def test_runtime_integration_no_gt_imports(self):
        import inspect

        import fsoc_tracker.ai.runtime_integration as ri
        source = inspect.getsource(ri)
        assert "WorldTruth" not in source
        assert "ground_truth" not in source.lower()

    def test_observable_features_no_gt(self):
        f = ObservableFeatures()
        # Should not have any attribute starting with "true_"
        for attr in dir(f):
            if not attr.startswith("_"):
                assert not attr.startswith("true_"), f"GT leakage: {attr}"


# ---------------------------------------------------------------------------
# AI failure isolation — AI error never crashes tracking
# ---------------------------------------------------------------------------

class TestAIFailureIsolation:
    def test_predictor_survives_bad_state(self):
        rp = RuntimePredictor()
        # NaN state
        pred = rp.predict(float("nan"), float("nan"), float("nan"), float("nan"))
        assert pred is not None
        assert pred.valid is True

    def test_extractor_survives_extreme_state(self):
        ext = RuntimeFeatureExtractor(640, 480)
        state = _make_state(x=99999, y=-99999, vx=1e6, vy=-1e6)
        features = ext.extract(state, timestamp_s=1.0, dt=0.0)
        assert features.estimated_x == 99999
        assert np.isfinite(features.to_feature_vector()).all()

    def test_safety_blocks_extreme_predictions(self):
        gate = SafetyGate(max_prediction_displacement_px=100)
        f = ObservableFeatures(track_state="TRACKING", has_detection=True)
        rp = RuntimePredictor()
        pred = rp.predict(320, 240, 10000, 10000, 5, 5)
        passed, _ = gate.check(f, pred)
        assert passed is False


# ---------------------------------------------------------------------------
# Model provenance tests
# ---------------------------------------------------------------------------

class TestModelProvenance:
    def test_cv_provenance(self):
        cv = ConstantVelocityPredictor()
        prov = cv.provenance
        d = prov.to_dict()
        assert d["model_type"] == "constant_velocity"
        # Analytical deterministic baseline: never "trained".
        assert d["trained"] is False

    def test_gru_provenance_no_checkpoint(self):
        gru = GRUPredictor()
        assert gru.available is False
        prov = gru.provenance
        assert prov.trained is False

    def test_gru_rejects_raw_state_dict_checkpoint(self):
        # Raw state dicts (temporal-v* format) must NEVER load as
        # "trained": random weights labeled trained is a safety fault.
        import os
        ckpt = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "artifacts", "models", "temporal-v4", "temporal_gru.pt",
        )
        if not os.path.exists(ckpt):
            pytest.skip("temporal-v4 checkpoint not shipped")
        gru = GRUPredictor(checkpoint_path=ckpt)
        assert gru.available is False
        assert gru.provenance.trained is False

    def test_runtime_predictor_provenance(self):
        rp = RuntimePredictor()
        prov = rp.provenance
        assert "cv" in prov
        assert "gru" in prov
        assert prov["active_method"] == "constant_velocity"
