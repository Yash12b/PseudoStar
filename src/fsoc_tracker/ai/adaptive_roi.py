"""Adaptive ROI module.

Dynamically adjusts the region of interest for perception based on:
- Predicted position from temporal model
- Current situation class
- Failure risk score
- Tracking uncertainty

Outputs an ROI center and radius that the perception engine can use
to crop the frame before detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ROISizeMode(str, Enum):
    """How ROI size is determined."""
    FIXED = "fixed"
    UNCERTAINTY_BASED = "uncertainty_based"
    SITUATION_AWARE = "situation_aware"


@dataclass
class ROIState:
    """Current adaptive ROI configuration."""
    center_x: float
    center_y: float
    radius_px: float
    mode: ROISizeMode
    reason: str
    confidence: float


class AdaptiveROI:
    """Dynamically adjusts perception ROI based on AI state.

    Normal tracking: small ROI centered on predicted position
    Low confidence / high noise: expand ROI
    Failure risk > 0.7: full-frame search
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        default_radius_px: float = 80.0,
        min_radius_px: float = 30.0,
        max_radius_px: float = 400.0,
        full_frame_threshold: float = 0.7,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.default_radius = default_radius_px
        self.min_radius = min_radius_px
        self.max_radius = max_radius_px
        self.full_frame_threshold = full_frame_threshold
        self._last_state = ROIState(
            center_x=image_width / 2.0,
            center_y=image_height / 2.0,
            radius_px=default_radius_px,
            mode=ROISizeMode.FIXED,
            reason="initialization",
            confidence=1.0,
        )

    @property
    def state(self) -> ROIState:
        return self._last_state

    @property
    def roi_rect(self) -> tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) clipped to image bounds."""
        cx = self._last_state.center_x
        cy = self._last_state.center_y
        r = self._last_state.radius_px
        x_min = max(0, int(cx - r))
        y_min = max(0, int(cy - r))
        x_max = min(self.image_width, int(cx + r))
        y_max = min(self.image_height, int(cy + r))
        return x_min, y_min, x_max, y_max

    def compute(
        self,
        estimated_x: float,
        estimated_y: float,
        predicted_x: float | None = None,
        predicted_y: float | None = None,
        uncertainty_x: float = 5.0,
        uncertainty_y: float = 5.0,
        situation: str = "normal_tracking",
        failure_risk: float = 0.0,
        action: str = "track",
    ) -> ROIState:
        """Compute adaptive ROI based on current state.

        Args:
            estimated_x, estimated_y: current estimated target position
            predicted_x, predicted_y: predicted position (from temporal model)
            uncertainty_x, uncertainty_y: position uncertainty
            situation: current situation class
            failure_risk: failure risk score [0, 1]
            action: recommended mission action

        Returns:
            ROIState with center, radius, mode, reason
        """
        center_x = predicted_x if predicted_x is not None else estimated_x
        center_y = predicted_y if predicted_y is not None else estimated_y

        if failure_risk > self.full_frame_threshold:
            return ROIState(
                center_x=self.image_width / 2.0,
                center_y=self.image_height / 2.0,
                radius_px=self.max_radius,
                mode=ROISizeMode.FIXED,
                reason="failure_risk_high",
                confidence=1.0 - failure_risk,
            )

        if action in ("global_search", "local_search"):
            radius = self.max_radius if action == "global_search" else self.default_radius * 2
            return ROIState(
                center_x=self.image_width / 2.0,
                center_y=self.image_height / 2.0,
                radius_px=radius,
                mode=ROISizeMode.FIXED,
                reason=action,
                confidence=0.5,
            )

        if situation in ("target_lost", "reacquisition"):
            return ROIState(
                center_x=self.image_width / 2.0,
                center_y=self.image_height / 2.0,
                radius_px=self.max_radius,
                mode=ROISizeMode.FIXED,
                reason=f"situation_{situation}",
                confidence=0.3,
            )

        uncertainty_factor = max(uncertainty_x, uncertainty_y) / 5.0
        base_radius = self.default_radius * max(1.0, uncertainty_factor)

        if situation in ("low_confidence", "high_noise"):
            base_radius *= 1.5
            reason = f"situation_{situation}_expanded"
        else:
            reason = "normal_tracking"

        radius = max(self.min_radius, min(self.max_radius, base_radius))

        center_x = max(radius, min(self.image_width - radius, center_x))
        center_y = max(radius, min(self.image_height - radius, center_y))

        self._last_state = ROIState(
            center_x=center_x,
            center_y=center_y,
            radius_px=radius,
            mode=ROISizeMode.UNCERTAINTY_BASED,
            reason=reason,
            confidence=max(0.0, 1.0 - failure_risk),
        )
        return self._last_state

    def reset(self) -> None:
        self._last_state = ROIState(
            center_x=self.image_width / 2.0,
            center_y=self.image_height / 2.0,
            radius_px=self.default_radius,
            mode=ROISizeMode.FIXED,
            reason="reset",
            confidence=1.0,
        )
