"""Adaptive Region of Interest (ROI) for beacon detection.

Dynamically adjusts detection ROI based on tracking state:
  - STABLE: shrink ROI to focus computation
  - UNCERTAIN: expand ROI to capture movement
  - LOST: full frame (ROI = entire image)
  - REPEATED_MISSES: force full frame

The ROI must NEVER permanently blind the detector — it always
eventually resets to full frame or shrinks back.

All inputs are runtime-observable. No ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


class ROIState(Enum):
    """ROI adaptive state."""
    STABLE = auto()
    SHRINKING = auto()
    UNCERTAIN = auto()
    EXPANDING = auto()
    LOST = auto()
    FULL_FRAME = auto()
    RECOVERY = auto()


@dataclass
class ROIConfig:
    """Configuration for adaptive ROI."""
    image_width: int = 640
    image_height: int = 480

    # ROI bounds
    min_roi_fraction: float = 0.15   # Minimum ROI as fraction of image
    max_roi_fraction: float = 1.0    # Maximum ROI (full frame)
    initial_roi_fraction: float = 0.5

    # Stable → shrink
    stable_frames_to_shrink: int = 15
    shrink_rate: float = 0.02        # Fraction reduction per frame

    # Uncertain → expand
    uncertainty_expand_rate: float = 0.05
    confidence_low_threshold: float = 0.4
    confidence_high_threshold: float = 0.7

    # Lost → full frame
    lost_frames_to_full: int = 3

    # Repeated miss → force full frame
    miss_count_to_force_full: int = 5

    # Recovery: after reacquisition, gradually shrink
    recovery_shrink_rate: float = 0.01
    recovery_stable_frames: int = 10


@dataclass
class ROIDiagnostics:
    """Diagnostic info about current ROI state."""
    state: str = "STABLE"
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 640
    roi_y2: int = 480
    roi_fraction: float = 1.0
    roi_center_x: float = 320.0
    roi_center_y: float = 240.0
    consecutive_detections: int = 0
    consecutive_misses: int = 0
    frames_in_state: int = 0

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "region": [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2],
            "fraction": round(self.roi_fraction, 3),
            "center": [round(self.roi_center_x, 1), round(self.roi_center_y, 1)],
            "consec_det": self.consecutive_detections,
            "consec_miss": self.consecutive_misses,
            "frames_in_state": self.frames_in_state,
        }


class AdaptiveROI:
    """Dynamically adjusts detection ROI based on tracking state.

    The ROI NEVER permanently blinds the detector:
    - After N missed detections → force full frame
    - After reacquisition → gradually shrink back
    - Maximum duration in any non-full-frame state → reset to full

    Usage::

        roi = AdaptiveROI(config)
        region = roi.update(
            detected=True, confidence=0.8,
            estimated_x=320, estimated_y=240,
        )
        # region is (x1, y1, x2, y2) or None for full frame
    """

    def __init__(self, config: ROIConfig | None = None) -> None:
        self._config = config or ROIConfig()
        self._state = ROIState.FULL_FRAME
        self._roi_fraction = self._config.initial_roi_fraction
        self._center_x = self._config.image_width / 2.0
        self._center_y = self._config.image_height / 2.0
        self._consecutive_detections = 0
        self._consecutive_misses = 0
        self._frames_in_state = 0
        self._total_frames = 0
        self._forced_full_frame_count = 0

    @property
    def diagnostics(self) -> ROIDiagnostics:
        x1, y1, x2, y2 = self._compute_roi()
        return ROIDiagnostics(
            state=self._state.name,
            roi_x1=x1, roi_y1=y1, roi_x2=x2, roi_y2=y2,
            roi_fraction=self._roi_fraction,
            roi_center_x=self._center_x,
            roi_center_y=self._center_y,
            consecutive_detections=self._consecutive_detections,
            consecutive_misses=self._consecutive_misses,
            frames_in_state=self._frames_in_state,
        )

    @property
    def region(self) -> tuple[int, int, int, int] | None:
        """Current ROI region. None means full frame."""
        return self._get_region()

    @property
    def state_name(self) -> str:
        """Current ROI mode name (lowercase) for telemetry."""
        return str(self._state.name).lower()

    def update(
        self,
        detected: bool,
        confidence: float,
        estimated_x: float,
        estimated_y: float,
        uncertainty_x: float = 5.0,
        uncertainty_y: float = 5.0,
    ) -> tuple[int, int, int, int] | None:
        """Update ROI state and return detection region.

        Returns:
            (x1, y1, x2, y2) bounding box, or None for full frame.
        """
        self._total_frames += 1
        self._frames_in_state += 1
        self._center_x = estimated_x
        self._center_y = estimated_y

        cfg = self._config

        if detected:
            self._consecutive_detections += 1
            self._consecutive_misses = 0
        else:
            self._consecutive_detections = 0
            self._consecutive_misses += 1

        # State transitions
        if self._state == ROIState.FULL_FRAME:
            if detected and confidence > cfg.confidence_high_threshold:
                self._transition(ROIState.SHRINKING)
            # Stay in FULL_FRAME if not detected

        elif self._state == ROIState.STABLE:
            if detected and confidence > cfg.confidence_high_threshold:
                if self._consecutive_detections > cfg.stable_frames_to_shrink:
                    self._transition(ROIState.SHRINKING)
            elif not detected:
                self._transition(ROIState.EXPANDING)
            elif confidence < cfg.confidence_low_threshold:
                self._transition(ROIState.UNCERTAIN)

        elif self._state == ROIState.SHRINKING:
            if not detected:
                self._transition(ROIState.EXPANDING)
            elif confidence < cfg.confidence_low_threshold:
                self._transition(ROIState.UNCERTAIN)
            else:
                # Shrink ROI
                self._roi_fraction = max(
                    cfg.min_roi_fraction,
                    self._roi_fraction - cfg.shrink_rate,
                )
                if self._roi_fraction <= cfg.min_roi_fraction:
                    self._transition(ROIState.STABLE)

        elif self._state == ROIState.UNCERTAIN:
            if not detected:
                self._transition(ROIState.EXPANDING)
            elif confidence > cfg.confidence_high_threshold:
                self._transition(ROIState.SHRINKING)
            else:
                # Expand ROI
                self._roi_fraction = min(
                    cfg.max_roi_fraction,
                    self._roi_fraction + cfg.uncertainty_expand_rate,
                )

        elif self._state == ROIState.EXPANDING:
            if detected and confidence > cfg.confidence_high_threshold:
                self._transition(ROIState.RECOVERY)
            else:
                self._roi_fraction = min(
                    cfg.max_roi_fraction,
                    self._roi_fraction + cfg.uncertainty_expand_rate,
                )
                if self._roi_fraction >= cfg.max_roi_fraction:
                    self._transition(ROIState.LOST)

        elif self._state == ROIState.LOST:
            if detected:
                self._transition(ROIState.RECOVERY)
            elif self._frames_in_state >= cfg.lost_frames_to_full:
                self._roi_fraction = cfg.max_roi_fraction
                # Stay in LOST with full frame

        elif self._state == ROIState.RECOVERY:
            if detected and self._consecutive_detections > cfg.recovery_stable_frames:
                self._transition(ROIState.STABLE)
            elif not detected:
                self._transition(ROIState.EXPANDING)

        # Safety: force full frame after too many consecutive misses
        if self._consecutive_misses >= cfg.miss_count_to_force_full:
            self._force_full_frame("repeated_misses")

        # Safety: force full frame if stuck too long
        if (
            self._state != ROIState.FULL_FRAME
            and self._frames_in_state > 100
        ):
            self._force_full_frame("timeout")

        return self._get_region()

    def _transition(self, new_state: ROIState) -> None:
        self._state = new_state
        self._frames_in_state = 0

    def _force_full_frame(self, reason: str = "") -> None:
        """Force full frame — ROI must never permanently blind detector."""
        self._state = ROIState.FULL_FRAME
        self._roi_fraction = self._config.max_roi_fraction
        self._frames_in_state = 0
        self._consecutive_misses = 0
        self._forced_full_frame_count += 1

    def _compute_roi(self) -> tuple[int, int, int, int]:
        """Compute ROI bounding box from current center and fraction."""
        cfg = self._config
        w, h = cfg.image_width, cfg.image_height

        roi_w = w * self._roi_fraction
        roi_h = h * self._roi_fraction

        cx = max(roi_w / 2, min(w - roi_w / 2, self._center_x))
        cy = max(roi_h / 2, min(h - roi_h / 2, self._center_y))

        x1 = max(0, int(cx - roi_w / 2))
        y1 = max(0, int(cy - roi_h / 2))
        x2 = min(w, int(cx + roi_w / 2))
        y2 = min(h, int(cy + roi_h / 2))

        return (x1, y1, x2, y2)

    def _get_region(self) -> tuple[int, int, int, int] | None:
        """Get current ROI region. None means full frame."""
        if self._state == ROIState.FULL_FRAME or self._roi_fraction >= 0.95:
            return None
        return self._compute_roi()

    def reset(self) -> None:
        """Reset to full frame."""
        self._state = ROIState.FULL_FRAME
        self._roi_fraction = self._config.max_roi_fraction
        self._consecutive_detections = 0
        self._consecutive_misses = 0
        self._frames_in_state = 0
        self._forced_full_frame_count = 0

    def expand_for_risk(self, failure_risk: float,
                        threshold: float = 0.7) -> bool:
        """Preemptively widen the ROI when failure risk is high.

        Moves STABLE/SHRINKING states to EXPANDING so the region grows
        over subsequent frames *before* misses accumulate. Never shrinks
        or blinds: FULL_FRAME/EXPANDING/LOST/RECOVERY are untouched.
        Returns True if a transition was triggered.
        """
        if not math.isfinite(failure_risk) or failure_risk < threshold:
            return False
        if self._state in (ROIState.STABLE, ROIState.SHRINKING):
            self._transition(ROIState.EXPANDING)
            return True
        return False
