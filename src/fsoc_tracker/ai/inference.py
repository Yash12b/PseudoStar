"""AI beacon detector — PerceptionEngine implementation.

Wraps the BeaconCNN model into the standard PerceptionEngine interface.
Supports both full-frame and ROI inference modes.
"""

from __future__ import annotations

import time

import numpy as np

from fsoc_tracker.ai.config import AIModelConfig
from fsoc_tracker.ai.coordinates import model_to_source
from fsoc_tracker.ai.model import BeaconCNN
from fsoc_tracker.perception.base import PerceptionEngine
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.perception.models import (
    BeaconDetection,
    PerceptionResult,
    PerceptionStatus,
    TargetClass,
)


class AIBeaconDetector(PerceptionEngine):
    """AI-based beacon detector using a tiny heatmap CNN.

    This detector is specialized for small, bright, localized optical
    beacon targets (5-20 pixels) under various disturbances.

    Usage::

        detector = AIBeaconDetector(model_config)
        detector.load_weights("path/to/weights.npz")
        result = detector.detect(image, timestamp_s=1.0)
    """

    def __init__(
        self,
        model_config: AIModelConfig | None = None,
        config: PerceptionConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._model_config = model_config or AIModelConfig()
        self._model = BeaconCNN(self._model_config)
        self._model.initialize()
        self._scale_factor = 1.0

    @property
    def name(self) -> str:
        return "ai_heatmap_cnn"

    @property
    def model(self) -> BeaconCNN:
        return self._model

    def load_weights(self, path: str) -> None:
        """Load model weights from .npz file.

        Args:
            path: path to .npz weights file
        """
        data = np.load(path)
        weights = {k: data[k] for k in data.files}
        self._model.set_weights(weights)

    def save_weights(self, path: str) -> None:
        """Save model weights to .npz file.

        Args:
            path: path to save .npz file
        """
        weights = self._model.get_weights()
        np.savez(path, **weights)

    def detect(
        self,
        image: np.ndarray,
        timestamp_s: float = 0.0,
        frame_index: int = 0,
    ) -> PerceptionResult:
        """Run AI beacon detection on a single frame.

        Args:
            image: (H, W) grayscale uint8 image
            timestamp_s: frame timestamp
            frame_index: sequential frame index

        Returns:
            PerceptionResult with detections.
        """
        t0 = time.perf_counter()

        h, w = image.shape[:2]
        # Accept BGR video frames like the classical path (which converts
        # in preprocessing); the model needs single-channel input.
        gray = image
        if gray.ndim == 3:
            try:
                import cv2
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            except ImportError:
                gray = gray[:, :, 0]

        # Compute scale factor from original to model input
        scale_x = w / self._model_config.input_width
        scale_y = h / self._model_config.input_height
        self._scale_factor = max(scale_x, scale_y)

        # Normalize and resize for model input
        img_float = gray.astype(np.float32) / 255.0

        if img_float.shape != (self._model_config.input_height, self._model_config.input_width):
            img_float = self._resize_for_model(img_float)

        # Run model
        model_output = self._model.forward(img_float)

        # Extract detections from heatmap
        detections = self._extract_detections(
            model_output.heatmap, w, h, timestamp_s, frame_index,
        )

        # Select primary detection
        primary = None
        status = PerceptionStatus.NO_TARGET
        if detections:
            # Sort by confidence
            detections.sort(key=lambda d: d.confidence, reverse=True)
            best = detections[0]
            if best.confidence >= self._model_config.confidence_threshold:
                primary = best
                status = PerceptionStatus.DETECTED
            else:
                status = PerceptionStatus.CANDIDATE

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PerceptionResult(
            detections=detections,
            primary_detection=primary,
            processing_time_ms=elapsed_ms,
            frame_timestamp=timestamp_s,
            frame_index=frame_index,
            detector_name=self.name,
            status=status,
            image_width=w,
            image_height=h,
            diagnostics={
                "model_confidence": model_output.confidence,
                "model_peaks": model_output.num_peaks,
                "model_trained": self._model.trained,
                "weights_source": self._model.weights_source,
            },
        )

    def _resize_for_model(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model input dimensions."""
        h, w = image.shape
        target_h = self._model_config.input_height
        target_w = self._model_config.input_width
        try:
            import cv2
            return cv2.resize(image, (target_w, target_h),
                              interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass
        result = np.zeros((target_h, target_w), dtype=np.float32)
        for i in range(target_h):
            for j in range(target_w):
                src_i = min(int(i * h / target_h), h - 1)
                src_j = min(int(j * w / target_w), w - 1)
                result[i, j] = image[src_i, src_j]
        return result

    def _extract_detections(
        self,
        heatmap: np.ndarray,
        orig_width: int,
        orig_height: int,
        timestamp_s: float,
        frame_index: int,
    ) -> list[BeaconDetection]:
        """Extract BeaconDetection objects from heatmap peaks.

        Args:
            heatmap: (H_model, W_model) predicted heatmap
            orig_width, orig_height: original image dimensions
            timestamp_s: frame timestamp
            frame_index: frame index

        Returns:
            List of BeaconDetection objects.
        """
        detections = []
        hm_h, hm_w = heatmap.shape

        # Find peaks as connected blobs above threshold, each localized
        # by intensity-weighted centroid (robust to plateau heatmaps;
        # argmax + fixed suppression scatters across flat peaks).
        import cv2
        thr = float(self._model_config.peak_threshold)
        mask = (heatmap >= thr).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        blobs = []
        for label_id in range(1, num):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            ys, xs = np.where(labels == label_id)
            wts = heatmap[ys, xs].astype(np.float64)
            total = float(np.sum(wts))
            if total <= 0:
                continue
            cx = float(np.sum(xs * wts) / total)
            cy = float(np.sum(ys * wts) / total)
            conf = float(np.max(wts))
            blobs.append((conf, area, cx, cy))
        blobs.sort(key=lambda t: t[0], reverse=True)
        for _ in range(min(self._model_config.max_detections, len(blobs))):
            conf, area, cx, cy = blobs.pop(0)
            # Map back to original image coordinates
            center_x, center_y = model_to_source(
                cx, cy, hm_w, hm_h, orig_width, orig_height)
            conf = float(conf)

            # Estimate bounding box from heatmap spread
            bbox_half = max(2, int(self._model_config.heatmap_sigma * 2))
            bbox = (
                max(0, int(center_x) - bbox_half),
                max(0, int(center_y) - bbox_half),
                min(orig_width, int(center_x) + bbox_half),
                min(orig_height, int(center_y) + bbox_half),
            )

            detection = BeaconDetection(
                detected=conf >= self._model_config.confidence_threshold,
                target_class=TargetClass.BEACON,
                center_x=center_x,
                center_y=center_y,
                bbox=bbox,
                width=float(bbox[2] - bbox[0]),
                height=float(bbox[3] - bbox[1]),
                confidence=conf,
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                algorithm=self.name,
                visibility_state=(
                    PerceptionStatus.DETECTED
                    if conf >= self._model_config.confidence_threshold
                    else PerceptionStatus.CANDIDATE
                ),
                diagnostics={"heatmap_value": conf},
            )
            detections.append(detection)

        return detections
