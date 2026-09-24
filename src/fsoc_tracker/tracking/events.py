"""Performance logging hooks for the tracking subsystem.

Provides structured events that future Stage 10 can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fsoc_tracker.tracking.state import TrackEvent, TrackingEvent


@dataclass
class TrackingMetrics:
    """Aggregated tracking performance metrics."""

    total_frames: int = 0
    frames_with_detection: int = 0
    frames_predicted_only: int = 0

    acquisition_count: int = 0
    loss_count: int = 0
    reacquisition_count: int = 0

    acquisition_times: list[float] = field(default_factory=list)
    reacquisition_times: list[float] = field(default_factory=list)
    miss_durations: list[float] = field(default_factory=list)

    residuals: list[float] = field(default_factory=list)
    uncertainties: list[float] = field(default_factory=list)
    processing_times_ms: list[float] = field(default_factory=list)

    @property
    def target_loss_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.loss_count / self.total_frames

    @property
    def mean_acquisition_time(self) -> float:
        if not self.acquisition_times:
            return 0.0
        return sum(self.acquisition_times) / len(self.acquisition_times)

    @property
    def mean_reacquisition_time(self) -> float:
        if not self.reacquisition_times:
            return 0.0
        return sum(self.reacquisition_times) / len(self.reacquisition_times)

    @property
    def mean_residual(self) -> float:
        if not self.residuals:
            return 0.0
        return sum(self.residuals) / len(self.residuals)

    @property
    def rmse_residual(self) -> float:
        if not self.residuals:
            return 0.0
        return (sum(r**2 for r in self.residuals) / len(self.residuals)) ** 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "frames_with_detection": self.frames_with_detection,
            "frames_predicted_only": self.frames_predicted_only,
            "acquisition_count": self.acquisition_count,
            "loss_count": self.loss_count,
            "reacquisition_count": self.reacquisition_count,
            "target_loss_rate": self.target_loss_rate,
            "mean_acquisition_time_s": self.mean_acquisition_time,
            "mean_reacquisition_time_s": self.mean_reacquisition_time,
            "mean_residual_px": self.mean_residual,
            "rmse_residual_px": self.rmse_residual,
        }


class TrackingMetricsCollector:
    """Collects structured tracking events into metrics."""

    def __init__(self) -> None:
        self._metrics = TrackingMetrics()

    @property
    def metrics(self) -> TrackingMetrics:
        return self._metrics

    def record_event(self, event: TrackingEvent) -> None:
        """Record a single tracking event."""
        if event.event == TrackEvent.TRACK_ACQUIRED:
            self._metrics.acquisition_count += 1
            acq_time = event.metadata.get("acquisition_time_s", 0.0)
            if acq_time > 0:
                self._metrics.acquisition_times.append(acq_time)

        elif event.event == TrackEvent.TRACK_LOST:
            self._metrics.loss_count += 1
            miss_dur = event.metadata.get("miss_duration_s", 0.0)
            if miss_dur > 0:
                self._metrics.miss_durations.append(miss_dur)

        elif event.event == TrackEvent.TRACK_REACQUIRED:
            self._metrics.reacquisition_count += 1
            reacq_dur = event.metadata.get("reacquisition_duration_s", 0.0)
            if reacq_dur > 0:
                self._metrics.reacquisition_times.append(reacq_dur)

    def record_frame(
        self,
        has_detection: bool,
        prediction_only: bool,
        residual: float,
        uncertainty: float,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Record per-frame tracking data."""
        self._metrics.total_frames += 1
        if has_detection:
            self._metrics.frames_with_detection += 1
        if prediction_only:
            self._metrics.frames_predicted_only += 1
        self._metrics.residuals.append(residual)
        self._metrics.uncertainties.append(uncertainty)
        if processing_time_ms > 0:
            self._metrics.processing_times_ms.append(processing_time_ms)

    def reset(self) -> None:
        self._metrics = TrackingMetrics()
