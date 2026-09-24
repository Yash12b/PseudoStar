"""Tests for temporal training, failure predictor, and adaptive ROI."""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.ai.adaptive_roi import AdaptiveROI
from fsoc_tracker.ai.failure_predictor import FailurePredictor
from fsoc_tracker.ai.mission import ObservationFeatures
from fsoc_tracker.ai.neural import torch_backend_available


class TestAdaptiveROI:
    def test_initial_state(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        assert roi.state.center_x == 320.0
        assert roi.state.center_y == 240.0
        assert roi.state.radius_px == 80.0

    def test_normal_tracking(self):
        roi = AdaptiveROI()
        state = roi.compute(320.0, 240.0, uncertainty_x=3.0, uncertainty_y=4.0, situation="normal_tracking")
        assert state.center_x == pytest.approx(320.0)
        assert state.center_y == pytest.approx(240.0)
        assert state.radius_px >= roi.min_radius
        assert state.radius_px <= roi.max_radius

    def test_high_failure_risk_goes_full_frame(self):
        roi = AdaptiveROI()
        state = roi.compute(320.0, 240.0, failure_risk=0.9)
        assert state.radius_px == roi.max_radius
        assert state.reason == "failure_risk_high"

    def test_target_lost_goes_full_frame(self):
        roi = AdaptiveROI()
        state = roi.compute(320.0, 240.0, situation="target_lost")
        assert state.radius_px == roi.max_radius

    def test_low_confidence_expands(self):
        roi = AdaptiveROI()
        state_norm = roi.compute(320.0, 240.0, situation="normal_tracking", uncertainty_x=5.0, uncertainty_y=5.0)
        state_low = roi.compute(320.0, 240.0, situation="low_confidence", uncertainty_x=5.0, uncertainty_y=5.0)
        assert state_low.radius_px >= state_norm.radius_px

    def test_global_search_action(self):
        roi = AdaptiveROI()
        state = roi.compute(320.0, 240.0, action="global_search")
        assert state.radius_px == roi.max_radius
        assert state.center_x == 320.0

    def test_roi_rect_clipped(self):
        roi = AdaptiveROI(image_width=640, image_height=480)
        roi.compute(0.0, 0.0, failure_risk=0.9)
        rect = roi.roi_rect
        assert rect[0] >= 0
        assert rect[1] >= 0
        assert rect[2] <= 640
        assert rect[3] <= 480

    def test_reset(self):
        roi = AdaptiveROI()
        roi.compute(320.0, 240.0, failure_risk=0.9)
        roi.reset()
        assert roi.state.reason == "reset"
        assert roi.state.radius_px == roi.default_radius


class TestFailurePredictor:
    def _make_window(self, rng: np.random.Generator, n: int = 10) -> list[ObservationFeatures]:
        return [ObservationFeatures(
            timestamp_s=float(j * 0.033),
            detected=bool(rng.random() > 0.2),
            confidence=float(rng.uniform(0.3, 1.0)),
            residual_px=float(rng.uniform(0, 20)),
            uncertainty_x_px=float(rng.uniform(1, 10)),
            uncertainty_y_px=float(rng.uniform(1, 10)),
            velocity_x_px_s=float(rng.uniform(-50, 50)),
            velocity_y_px_s=float(rng.uniform(-50, 50)),
            distance_from_center_px=float(rng.uniform(0, 200)),
            time_since_detection_s=0.0,
            latency_ms=2.0,
            source_fps=30.0,
            processing_fps=25.0,
            candidate_count=int(rng.integers(0, 5)),
            roi_radius_px=80.0,
        ) for j in range(n)]

    def test_untrained_returns_default(self):
        fp = FailurePredictor(window_size=5)
        w = self._make_window(np.random.default_rng(0), 5)
        pred = fp.predict(w)
        assert pred.risk_score == 0.0
        assert pred.recommended_action == "track"

    def test_train_and_predict(self):
        rng = np.random.default_rng(42)
        windows = [self._make_window(rng, 5) for _ in range(50)]
        labels = [bool(rng.random() > 0.8) for _ in range(50)]
        fp = FailurePredictor(window_size=5)
        acc = fp.fit(windows, labels)
        assert 0.0 <= acc <= 1.0
        assert fp.trained
        pred = fp.predict(windows[0])
        assert 0.0 <= pred.risk_score <= 1.0
        assert pred.recommended_action in ("track", "hold", "reacquire", "search")

    def test_evaluate(self):
        rng = np.random.default_rng(42)
        windows = [self._make_window(rng, 5) for _ in range(50)]
        labels = [bool(rng.random() > 0.8) for _ in range(50)]
        fp = FailurePredictor(window_size=5)
        fp.fit(windows, labels)
        metrics = fp.evaluate(windows[:20], labels[:20])
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_save_load(self, tmp_path):
        rng = np.random.default_rng(42)
        windows = [self._make_window(rng, 5) for _ in range(50)]
        labels = [bool(rng.random() > 0.8) for _ in range(50)]
        fp = FailurePredictor(window_size=5)
        fp.fit(windows, labels)
        fp.save(tmp_path / "test.npz")
        loaded = FailurePredictor.load(tmp_path / "test.npz")
        assert loaded.trained
        assert loaded.predict(windows[0]).risk_score == pytest.approx(
            fp.predict(windows[0]).risk_score
        )


@pytest.mark.skipif(not torch_backend_available(), reason="PyTorch not available")
class TestTemporalTraining:
    def test_build_and_forward(self):
        import torch

        from fsoc_tracker.ai.neural import build_temporal_predictor

        model = build_temporal_predictor(feature_dim=15, hidden_dim=16, horizons_s=(0.025, 0.05, 0.1))
        x = torch.randn(2, 10, 15)
        output = model(x)
        assert "displacement" in output
        assert output["displacement"].shape == (2, 3, 2)
        assert "uncertainty" in output
        assert output["uncertainty"].shape == (2, 3, 2)
