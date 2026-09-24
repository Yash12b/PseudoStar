"""Hybrid beacon detector — classical + AI fusion.

Combines the classical bright-spot detector with the AI heatmap CNN
to produce more robust detections. The two backends run independently
and their results are fused.
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from fsoc_tracker.ai.config import AIModelConfig
from fsoc_tracker.ai.inference import AIBeaconDetector
from fsoc_tracker.perception.base import PerceptionEngine
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.perception.models import (
    BeaconDetection,
    PerceptionResult,
    PerceptionStatus,
)


class FusionPolicy(str, Enum):
    """How to combine AI and classical detections."""

    BOTH_AGREE = "both_agree"
    """Only detect if both AI and classical agree."""

    EITHER = "either"
    """Detect if either AI or classical detects."""

    AI_WITH_CLASSICAL_CONFIRM = "ai_with_classical_confirm"
    """AI primary, classical must confirm nearby."""

    CLASSICAL_WITH_AI_CONFIRM = "classical_with_ai_confirm"
    """Classical primary, AI must confirm nearby."""

    AI_OR_CLASSICAL = "ai_or_classical"
    """AI preferred, fallback to classical if AI fails."""

    AI_UNLESS_CONFLICT = "ai_unless_conflict"
    """AI always preferred unless classical has higher confidence."""


class HybridBeaconDetector(PerceptionEngine):
    """Hybrid beacon detector combining classical CV and AI backends.

    Both backends run independently. Results are fused according to
    the configured FusionPolicy.

    Fallback: if AI model fails to load, runs in classical-only mode.

    Usage::

        detector = HybridBeaconDetector(
            fusion_policy=FusionPolicy.AI_OR_CLASSICAL,
        )
        result = detector.detect(image, timestamp_s=1.0)
    """

    def __init__(
        self,
        model_config: AIModelConfig | None = None,
        config: PerceptionConfig | None = None,
        fusion_policy: FusionPolicy = FusionPolicy.AI_OR_CLASSICAL,
        proximity_threshold: float = 10.0,
        weights_path: str | None = None,
    ) -> None:
        super().__init__(config)
        self._fusion_policy = fusion_policy
        self._proximity_threshold = proximity_threshold

        # Classical backend (always available)
        self._classical = ClassicalBeaconDetector(config)

        # AI backend (may fail to load). ai_available is True ONLY with
        # trained weights loaded — never for random initialization.
        self._ai: AIBeaconDetector | None = None
        self._ai_available = False
        try:
            self._ai = AIBeaconDetector(model_config, config)
            if weights_path is not None:
                self._ai.load_weights(weights_path)
            self._ai_available = bool(self._ai.model.trained)
        except Exception:
            self._ai_available = False

    @property
    def name(self) -> str:
        return f"hybrid_{self._fusion_policy.value}"

    @property
    def ai_available(self) -> bool:
        return self._ai_available

    @property
    def fusion_policy(self) -> FusionPolicy:
        return self._fusion_policy

    def detect(
        self,
        image: np.ndarray,
        timestamp_s: float = 0.0,
        frame_index: int = 0,
    ) -> PerceptionResult:
        """Run hybrid detection.

        Args:
            image: (H, W) grayscale uint8 image
            timestamp_s: frame timestamp
            frame_index: frame index

        Returns:
            PerceptionResult with fused detections.
        """
        t0 = time.perf_counter()

        # Classical detection
        classical_result = self._classical.detect(image, timestamp_s, frame_index)

        # AI detection (if available)
        ai_result = None
        if self._ai_available and self._ai is not None:
            try:
                ai_result = self._ai.detect(image, timestamp_s, frame_index)
            except Exception:
                ai_result = None

        # Fuse results
        fused = self._fuse(classical_result, ai_result, timestamp_s, frame_index)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fused.processing_time_ms = elapsed_ms

        return fused

    def _fuse(
        self,
        classical: PerceptionResult,
        ai: PerceptionResult | None,
        timestamp_s: float,
        frame_index: int,
    ) -> PerceptionResult:
        """Fuse classical and AI results according to fusion policy."""
        policy = self._fusion_policy

        # If AI is not available, use classical only
        if ai is None or not self._ai_available:
            return classical

        c_det = classical.primary_detection
        a_det = ai.primary_detection

        c_has = c_det is not None and c_det.detected
        a_has = a_det is not None and a_det.detected

        # Check proximity (are detections close to each other?)
        proximity_ok = False
        if c_has and a_has:
            dx = c_det.center_x - a_det.center_x
            dy = c_det.center_y - a_det.center_y
            dist = (dx * dx + dy * dy) ** 0.5
            proximity_ok = dist <= self._proximity_threshold

        if policy == FusionPolicy.BOTH_AGREE:
            if c_has and a_has and proximity_ok:
                return self._make_result(a_det, "hybrid_both_agree", timestamp_s, frame_index)
            return PerceptionResult(
                detector_name=self.name,
                status=PerceptionStatus.NO_TARGET,
                frame_timestamp=timestamp_s,
                frame_index=frame_index,
            )

        if policy == FusionPolicy.EITHER:
            if a_has:
                return self._make_result(a_det, "hybrid_ai", timestamp_s, frame_index)
            if c_has:
                return self._make_result(c_det, "hybrid_classical", timestamp_s, frame_index)
            return PerceptionResult(
                detector_name=self.name,
                status=PerceptionStatus.NO_TARGET,
                frame_timestamp=timestamp_s,
                frame_index=frame_index,
            )

        if policy == FusionPolicy.AI_WITH_CLASSICAL_CONFIRM:
            if a_has:
                if c_has and proximity_ok:
                    return self._make_result(a_det, "hybrid_ai_confirmed", timestamp_s, frame_index)
                # AI detected but classical didn't confirm — still use AI but lower confidence
                det = BeaconDetection(
                    detected=a_det.confidence > 0.5,
                    center_x=a_det.center_x,
                    center_y=a_det.center_y,
                    bbox=a_det.bbox,
                    confidence=a_det.confidence * 0.8,
                    algorithm="hybrid_ai_unconfirmed",
                    visibility_state=PerceptionStatus.UNCERTAIN,
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                )
                return self._make_result(det, "hybrid_ai_unconfirmed", timestamp_s, frame_index)
            if c_has:
                return self._make_result(c_det, "hybrid_classical_only", timestamp_s, frame_index)
            return PerceptionResult(
                detector_name=self.name,
                status=PerceptionStatus.NO_TARGET,
                frame_timestamp=timestamp_s,
                frame_index=frame_index,
            )

        if policy == FusionPolicy.CLASSICAL_WITH_AI_CONFIRM:
            if c_has:
                if a_has and proximity_ok:
                    return self._make_result(c_det, "hybrid_classical_confirmed", timestamp_s, frame_index)
                return self._make_result(c_det, "hybrid_classical_unconfirmed", timestamp_s, frame_index)
            if a_has:
                return self._make_result(a_det, "hybrid_ai_only", timestamp_s, frame_index)
            return PerceptionResult(
                detector_name=self.name,
                status=PerceptionStatus.NO_TARGET,
                frame_timestamp=timestamp_s,
                frame_index=frame_index,
            )

        if policy in (FusionPolicy.AI_OR_CLASSICAL, FusionPolicy.AI_UNLESS_CONFLICT):
            if a_has:
                # Check if classical strongly disagrees
                if c_has and not proximity_ok and c_det.confidence > a_det.confidence * 1.5:
                    # Classical has much higher confidence — suspicious
                    return self._make_result(c_det, "hybrid_classical_override", timestamp_s, frame_index)
                return self._make_result(a_det, "hybrid_ai_primary", timestamp_s, frame_index)
            if c_has:
                return self._make_result(c_det, "hybrid_classical_fallback", timestamp_s, frame_index)
            return PerceptionResult(
                detector_name=self.name,
                status=PerceptionStatus.NO_TARGET,
                frame_timestamp=timestamp_s,
                frame_index=frame_index,
            )

        # Default: classical
        return classical

    def _make_result(
        self,
        detection: BeaconDetection,
        algorithm: str,
        timestamp_s: float,
        frame_index: int,
    ) -> PerceptionResult:
        """Create a PerceptionResult with a single detection."""
        det = BeaconDetection(
            detected=detection.detected,
            target_class=detection.target_class,
            center_x=detection.center_x,
            center_y=detection.center_y,
            bbox=detection.bbox,
            width=detection.width,
            height=detection.height,
            confidence=detection.confidence,
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            algorithm=algorithm,
            visibility_state=detection.visibility_state,
        )
        return PerceptionResult(
            detections=[det],
            primary_detection=det,
            detector_name=self.name,
            status=det.visibility_state,
            frame_timestamp=timestamp_s,
            frame_index=frame_index,
            image_width=detection.bbox[2] if detection.bbox else 0,
            image_height=detection.bbox[3] if detection.bbox else 0,
        )
