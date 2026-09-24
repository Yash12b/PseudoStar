"""Failure predictor — predicts tracking loss probability.

NumPy-only linear model that predicts whether tracking will fail
within the next 0.5 seconds based on a sliding window of
runtime-observable features.

Architecture: regularized logistic regression (NumPy-only).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fsoc_tracker.ai.learned import observation_vector
from fsoc_tracker.ai.mission import ObservationFeatures


@dataclass
class FailurePrediction:
    """Output from the failure predictor."""
    risk_score: float  # 0.0 = no risk, 1.0 = certain failure
    recommended_action: str  # "hold", "search", "reacquire", "track"
    confidence: float
    window_size: int
    method: str = "logistic_regression"


class FailurePredictor:
    """Regularized logistic regression failure predictor.

    Uses a sliding window of observation features to predict
    whether tracking will fail within the next 0.5s.
    """

    def __init__(self, window_size: int = 10, ridge: float = 1e-3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.window_size = window_size
        self.ridge = ridge
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    @property
    def trained(self) -> bool:
        return self._weights is not None

    def _extract_features(
        self, window: Sequence[ObservationFeatures]
    ) -> np.ndarray:
        """Extract a fixed-size feature vector from a sliding window.

        For each frame in the window, we use the 15-dim observation vector.
        We also add temporal statistics: mean, std, trend over the window.
        """
        vectors = [observation_vector(f) for f in window]
        stacked = np.stack(vectors)  # (T, 15)

        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0) + 1e-8

        if len(stacked) >= 2:
            trend = stacked[-1] - stacked[0]
        else:
            trend = np.zeros_like(mean)

        last = stacked[-1]

        return np.concatenate([last, mean, std, trend])  # 60-dim

    def fit(
        self,
        windows: Sequence[Sequence[ObservationFeatures]],
        labels: Sequence[bool],
    ) -> float:
        """Train the failure predictor.

        Args:
            windows: list of feature windows
            labels: True if tracking fails within 0.5s of window end

        Returns:
            Training accuracy.
        """
        if len(windows) != len(labels):
            raise ValueError("windows and labels must have equal length")

        X = np.stack([self._extract_features(w) for w in windows])
        y = np.array([1.0 if lbl else 0.0 for lbl in labels], dtype=np.float64)

        self._feature_mean = np.mean(X, axis=0)
        self._feature_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        reg = self.ridge * np.eye(X_norm.shape[1], dtype=np.float64)
        X_with_bias = np.column_stack([X_norm, np.ones(len(X_norm))])
        reg_full = np.block([
            [reg, np.zeros((reg.shape[0], 1))],
            [np.zeros((1, reg.shape[1])), np.array([[0.0]])],
        ])
        weights = np.linalg.solve(X_with_bias.T @ X_with_bias + reg_full, X_with_bias.T @ y)
        self._weights = weights[:-1]
        self._bias = float(weights[-1])

        preds = self.predict_batch(windows)
        accuracy = float(np.mean([p > 0.5 for p in preds] == y))
        return accuracy

    def predict(self, window: Sequence[ObservationFeatures]) -> FailurePrediction:
        """Predict failure risk from a feature window."""
        if not self.trained:
            return FailurePrediction(
                risk_score=0.0,
                recommended_action="track",
                confidence=0.0,
                window_size=len(window),
            )

        x = self._extract_features(window)
        x_norm = (x - self._feature_mean) / self._feature_std
        logit = float(x_norm @ self._weights + self._bias)
        risk = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

        if risk > 0.8:
            action = "search"
        elif risk > 0.5:
            action = "reacquire"
        elif risk > 0.3:
            action = "hold"
        else:
            action = "track"

        return FailurePrediction(
            risk_score=float(risk),
            recommended_action=action,
            confidence=abs(risk - 0.5) * 2.0,
            window_size=len(window),
        )

    def predict_batch(self, windows: Sequence[Sequence[ObservationFeatures]]) -> list[float]:
        """Predict risk scores for multiple windows."""
        return [self.predict(w).risk_score for w in windows]

    def evaluate(
        self,
        windows: Sequence[Sequence[ObservationFeatures]],
        labels: Sequence[bool],
    ) -> dict[str, float]:
        """Evaluate on held-out data."""
        if not self.trained:
            raise RuntimeError("predictor is not trained")

        risk_scores = self.predict_batch(windows)
        y = np.array([1.0 if lbl else 0.0 for lbl in labels])
        risk_arr = np.array(risk_scores)

        tp = float(np.sum((risk_arr > 0.5) & (y > 0.5)))
        fp = float(np.sum((risk_arr > 0.5) & (y < 0.5)))
        tn = float(np.sum((risk_arr <= 0.5) & (y < 0.5)))
        fn = float(np.sum((risk_arr <= 0.5) & (y > 0.5)))

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        bce = float(-np.mean(y * np.log(risk_arr + 1e-7) + (1 - y) * np.log(1 - risk_arr + 1e-7)))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "bce_loss": bce,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def save(self, path: str | Path) -> None:
        if not self.trained:
            raise RuntimeError("cannot save untrained predictor")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            weights=self._weights,
            bias=np.array([self._bias]),
            feature_mean=self._feature_mean,
            feature_std=self._feature_std,
            metadata=json.dumps({
                "window_size": self.window_size,
                "ridge": self.ridge,
                "version": "FailurePredictor-logistic-v1",
            }),
        )

    @classmethod
    def load(cls, path: str | Path) -> FailurePredictor:
        data = np.load(path)
        metadata = json.loads(str(data["metadata"]))
        predictor = cls(
            window_size=int(metadata["window_size"]),
            ridge=float(metadata["ridge"]),
        )
        predictor._weights = np.asarray(data["weights"])
        predictor._bias = float(data["bias"][0])
        predictor._feature_mean = np.asarray(data["feature_mean"])
        predictor._feature_std = np.asarray(data["feature_std"])
        return predictor
