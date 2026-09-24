"""Perception backend factory (classical / ai / hybrid).

Single place that maps a backend name to a working detector honoring
the project rule: trained weights or classical fallback, never silent
random weights. Used by the GUI worker and the benchmark CLI alike.
"""
from __future__ import annotations

from typing import Any

from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig

DEFAULT_AI_WEIGHTS = "artifacts/models/beacon-numpy-v1/beacon_cnn.npz"


def build_backend(
    name: str,
    perc_cfg: PerceptionConfig | None = None,
    weights_path: str | None = None,
    log: Any | None = None,
):
    """Build the named perception backend with safe fallback.

    Returns (detector, backend_in_effect, note). ``backend_in_effect``
    may differ from ``name`` when fallback engaged.
    """
    def _say(level: str, msg: str) -> None:
        if log is not None:
            try:
                log.emit(level, msg)
            except Exception:
                pass

    cfg = perc_cfg or PerceptionConfig()
    backend = (name or "classical").lower()
    weights = weights_path or DEFAULT_AI_WEIGHTS
    if backend == "ai":
        try:
            from fsoc_tracker.ai.inference import AIBeaconDetector
            det = AIBeaconDetector(config=cfg)
            det.load_weights(weights)
            if not det.model.trained:
                raise RuntimeError("weights did not validate as trained")
            _say("INFO", "AI backend: trained BeaconCNN loaded")
            return det, "ai", "trained"
        except Exception as e:
            _say("WARNING", f"AI backend unavailable ({e}); classical fallback")
    elif backend == "hybrid":
        try:
            from fsoc_tracker.ai.hybrid import HybridBeaconDetector
            det = HybridBeaconDetector(config=cfg, weights_path=weights)
            if det.ai_available:
                _say("INFO", "Hybrid backend: AI+classical")
            else:
                _say("WARNING", "Hybrid AI part unavailable; classical core")
            return det, "hybrid", "ai" if det.ai_available else "classical-core"
        except Exception as e:
            _say("WARNING", f"Hybrid backend unavailable ({e}); classical fallback")
    return ClassicalBeaconDetector(config=cfg), "classical", "default"
