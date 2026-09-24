"""Runtime multi-horizon temporal predictor.

Production fallback: constant-velocity baseline.
Experimental: GRU-based predictor (loaded if checkpoint available).

All inputs are runtime-observable features only.
All outputs are pixel-space displacement predictions + uncertainty.
Model provenance is tracked for telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HORIZONS_S: list[float] = [0.025, 0.05, 0.1, 0.25, 0.5]
FEATURE_DIM = 15


@dataclass
class ModelProvenance:
    """Track model identity, version, and training provenance."""
    model_type: str = "constant_velocity"
    version: str = "cv-v1"
    trained: bool = False
    training_date: str = ""
    training_rmse_px: float = 0.0
    source: str = "builtin"
    checkpoint_path: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "trained": self.trained,
            "training_date": self.training_date,
            "training_rmse_px": round(self.training_rmse_px, 2),
            "source": self.source,
            "checkpoint_path": self.checkpoint_path,
            "notes": self.notes,
        }


@dataclass
class HorizonPrediction:
    """Prediction at a single time horizon."""
    horizon_s: float
    displacement_x_px: float
    displacement_y_px: float
    uncertainty_x_px: float
    uncertainty_y_px: float
    method: str = "constant_velocity"
    confidence: float = 1.0


@dataclass
class TemporalPrediction:
    """Full multi-horizon prediction result."""
    predictions: list[HorizonPrediction] = field(default_factory=list)
    method: str = "constant_velocity"
    provenance: ModelProvenance = field(default_factory=ModelProvenance)
    timestamp_s: float = 0.0
    valid: bool = True
    fallback: bool = False

    def get_displacement(self, horizon_s: float) -> tuple[float, float]:
        """Get (dx, dy) displacement for a given horizon."""
        for p in self.predictions:
            if abs(p.horizon_s - horizon_s) < 1e-6:
                return (p.displacement_x_px, p.displacement_y_px)
        # Interpolate if exact horizon not found
        if len(self.predictions) >= 2:
            horizons = [p.horizon_s for p in self.predictions]
            dxs = [p.displacement_x_px for p in self.predictions]
            dys = [p.displacement_y_px for p in self.predictions]
            dx = float(np.interp(horizon_s, horizons, dxs))
            dy = float(np.interp(horizon_s, horizons, dys))
            return (dx, dy)
        return (0.0, 0.0)

    def get_total_displacement(self) -> tuple[float, float]:
        """Get total displacement at the longest horizon."""
        if self.predictions:
            last = self.predictions[-1]
            return (last.displacement_x_px, last.displacement_y_px)
        return (0.0, 0.0)


class ConstantVelocityPredictor:
    """Production baseline: constant-velocity extrapolation.

    Known-good deterministic baseline (~4.51 px RMSE; eval protocol:
    ~6 test sequences x 1 timestep, no shipped metrics.json --
    see docs/ai/final-audit.md). Always available, no dependencies.
    """

    def __init__(self, uncertainty_growth_px_s: float = 10.0) -> None:
        self._growth = uncertainty_growth_px_s
        self._provenance = ModelProvenance(
            model_type="constant_velocity",
            version="cv-v1",
            trained=False,
            notes="deterministic_baseline — analytical, no training required",
        )

    @property
    def provenance(self) -> ModelProvenance:
        return self._provenance

    def predict(
        self,
        estimated_x: float,
        estimated_y: float,
        velocity_x: float,
        velocity_y: float,
        uncertainty_x: float,
        uncertainty_y: float,
        horizons_s: list[float] | None = None,
    ) -> TemporalPrediction:
        """Predict future pixel displacements using constant velocity."""
        horizons = horizons_s or HORIZONS_S
        predictions = []

        for h in horizons:
            dx = velocity_x * h
            dy = velocity_y * h
            ux = max(0.0, uncertainty_x) + self._growth * h
            uy = max(0.0, uncertainty_y) + self._growth * h
            predictions.append(HorizonPrediction(
                horizon_s=h,
                displacement_x_px=dx,
                displacement_y_px=dy,
                uncertainty_x_px=ux,
                uncertainty_y_px=uy,
                method="constant_velocity",
                confidence=1.0,
            ))

        return TemporalPrediction(
            predictions=predictions,
            method="constant_velocity",
            provenance=self._provenance,
            valid=True,
            fallback=False,
        )


class GRUPredictor:
    """Experimental GRU-based predictor.

    Loaded from checkpoint if available. Falls back gracefully.
    Known result: ~35 px RMSE (worse than CV baseline; same thin eval
    protocol -- see docs/ai/final-audit.md).
    Status: EXPERIMENTAL / TRAINED — not production.
    """

    def __init__(self, checkpoint_path: str = "") -> None:
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._available = False
        self._provenance = ModelProvenance(
            model_type="gru_temporal",
            version="gru-v1",
            trained=False,
            checkpoint_path=checkpoint_path,
            notes="EXPERIMENTAL — CV baseline is production fallback",
        )

        if checkpoint_path:
            self._try_load(checkpoint_path)

    def _try_load(self, path: str) -> None:
        """Attempt to load GRU checkpoint. Fails silently."""
        try:
            from fsoc_tracker.ai.neural import torch_backend_available
            if not torch_backend_available():
                return

            import torch

            from fsoc_tracker.ai.neural import build_temporal_predictor

            model = build_temporal_predictor(
                feature_dim=FEATURE_DIM,
                hidden_dim=32,
                horizons_s=tuple(HORIZONS_S),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                self._provenance.training_rmse_px = state.get("best_val_rmse", 0.0)
                self._provenance.training_date = state.get("training_date", "")
                self._provenance.trained = True
                self._provenance.source = "checkpoint"
                self._available = True
            else:
                # Raw state dicts (e.g. temporal-v*/temporal_gru.pt) and
                # incompatible architectures must NEVER run as "trained":
                # fall back to the deterministic baseline instead.
                self._provenance.trained = False
                self._provenance.source = "checkpoint-rejected"
                self._provenance.notes = (
                    "checkpoint format not loadable; CV baseline active"
                )
                self._available = False
                return
            model.eval()
            self._model = model
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def provenance(self) -> ModelProvenance:
        return self._provenance

    def predict(
        self,
        feature_vector: np.ndarray,
        sequence: np.ndarray | None = None,
        horizons_s: list[float] | None = None,
    ) -> TemporalPrediction | None:
        """Run GRU inference. Returns None if model unavailable."""
        if not self._available or self._model is None:
            return None

        horizons = horizons_s or HORIZONS_S

        try:
            import torch

            with torch.no_grad():
                if sequence is not None and sequence.ndim == 3:
                    # (1, T, FEATURE_DIM)
                    inp = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                else:
                    # Single feature vector → repeat as sequence
                    inp = torch.tensor(
                        feature_vector, dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)

                output = self._model(inp)

                if isinstance(output, dict):
                    dx = output.get("displacement", torch.zeros(1, len(horizons), 2))
                    ux = output.get("uncertainty", torch.ones(1, len(horizons), 2))
                else:
                    dx, ux = output

                dx_np = dx.squeeze(0).cpu().numpy()
                ux_np = ux.squeeze(0).cpu().numpy()

            predictions = []
            for i, h in enumerate(horizons):
                predictions.append(HorizonPrediction(
                    horizon_s=h,
                    displacement_x_px=float(dx_np[i, 0]),
                    displacement_y_px=float(dx_np[i, 1]),
                    uncertainty_x_px=float(abs(ux_np[i, 0])),
                    uncertainty_y_px=float(abs(ux_np[i, 1])),
                    method="gru_temporal",
                    confidence=0.5,  # GRU is experimental
                ))

            return TemporalPrediction(
                predictions=predictions,
                method="gru_temporal",
                provenance=self._provenance,
                valid=True,
                fallback=False,
            )
        except Exception:
            return None


class RuntimePredictor:
    """Multi-horizon temporal predictor with automatic fallback.

    Production path: ConstantVelocityPredictor (always available).
    Experimental path: GRUPredictor (if checkpoint loaded).

    Usage::

        predictor = RuntimePredictor()
        prediction = predictor.predict(
            estimated_x=320, estimated_y=240,
            velocity_x=50, velocity_y=0,
            uncertainty_x=5, uncertainty_y=5,
        )
        dx, dy = prediction.get_displacement(0.1)  # 100ms horizon
    """

    def __init__(self, gru_checkpoint_path: str = "") -> None:
        self._cv = ConstantVelocityPredictor()
        self._gru = GRUPredictor(gru_checkpoint_path)
        self._history: list[tuple[float, float, float]] = []  # (x, y, timestamp)
        self._max_history = 20

    @property
    def provenance(self) -> dict[str, Any]:
        """Combined provenance for telemetry."""
        return {
            "cv": self._cv.provenance.to_dict(),
            "gru": self._gru.provenance.to_dict(),
            "active_method": "gru_temporal" if self._gru.available else "constant_velocity",
        }

    @property
    def gru_available(self) -> bool:
        return self._gru.available

    def predict(
        self,
        estimated_x: float,
        estimated_y: float,
        velocity_x: float,
        velocity_y: float,
        uncertainty_x: float = 0.0,
        uncertainty_y: float = 0.0,
        timestamp_s: float = 0.0,
        feature_vector: np.ndarray | None = None,
        horizons_s: list[float] | None = None,
    ) -> TemporalPrediction:
        """Produce multi-horizon predictions.

        Always produces CV predictions. If GRU is available and confident,
        uses GRU instead. Falls back to CV on any GRU failure.
        """
        # Try GRU if available
        if self._gru.available and feature_vector is not None:
            gru_pred = self._gru.predict(feature_vector, horizons_s=horizons_s)
            if gru_pred is not None and gru_pred.valid:
                # Sanity check: GRU displacement should be within 3x CV
                cv_pred = self._cv.predict(
                    estimated_x, estimated_y,
                    velocity_x, velocity_y,
                    uncertainty_x, uncertainty_y,
                    horizons_s,
                )
                for gp, cp in zip(gru_pred.predictions, cv_pred.predictions, strict=False):
                    dx_ratio = abs(gp.displacement_x_px) / max(abs(cp.displacement_x_px), 1.0)
                    dy_ratio = abs(gp.displacement_y_px) / max(abs(cp.displacement_y_px), 1.0)
                    if dx_ratio > 3.0 or dy_ratio > 3.0:
                        # GRU output unreasonable — fall back to CV
                        gru_pred.fallback = True
                        break

                if not gru_pred.fallback:
                    gru_pred.timestamp_s = timestamp_s
                    return gru_pred

        # Production path: constant velocity
        pred = self._cv.predict(
            estimated_x, estimated_y,
            velocity_x, velocity_y,
            uncertainty_x, uncertainty_y,
            horizons_s,
        )
        pred.timestamp_s = timestamp_s
        return pred
