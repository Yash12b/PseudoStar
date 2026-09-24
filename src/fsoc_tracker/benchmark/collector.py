"""Streaming metrics collector.

Processes results frame-by-frame without storing unlimited images.
Collects perception, tracking, control, ground-truth, and disturbance metrics.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fsoc_tracker.benchmark.models import (
    AcquisitionMetrics,
    BenchmarkResult,
    FrameMetrics,
    LatencyMetrics,
    LossMetrics,
    PerformanceMetrics,
    PipelineLatency,
    ReacquisitionMetrics,
    TrackingMetricsSummary,
)


class MetricsCollector:
    """Streaming metrics collector that processes one frame at a time."""

    def __init__(self, tracking_error_threshold_px: float = 10.0) -> None:
        self._threshold_px = tracking_error_threshold_px
        self._frames: list[FrameMetrics] = []
        self._events: list[dict[str, Any]] = []
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0
        self._prev_timestamp: float | None = None
        self._prev_locked: bool = False
        self._lock_start_time: float | None = None
        self._locked_time_s: float = 0.0
        self._evaluable_time_s: float = 0.0
        self._first_detection_s: float | None = None
        self._acquisition_start_s: float | None = None
        self._acquisition_done: bool = False
        self._stable_acquisition_s: float | None = None
        self._loss_start_s: float | None = None
        self._reappear_s: float | None = None
        self._reacq_times: list[float] = []
        self._reacq_from_reappearance: list[float] = []
        self._loss_events: list[dict[str, Any]] = []
        self._acq_count: int = 0
        self._processing_times: list[float] = []
        self._perception_times: list[float] = []
        self._tracking_times: list[float] = []
        self._control_times: list[float] = []
        self._failure_categories: dict[str, int] = {}

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def stop(self) -> None:
        self._wall_end = time.perf_counter()

    @property
    def frames(self) -> list[FrameMetrics]:
        return self._frames

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    def record_frame(self, fm: FrameMetrics) -> None:
        self._frames.append(fm)
        self._categorize_failure(fm)
        if fm.processing_time_ms > 0:
            self._processing_times.append(fm.processing_time_ms)
        if fm.perception_time_ms > 0:
            self._perception_times.append(fm.perception_time_ms)
        if fm.tracking_time_ms > 0:
            self._tracking_times.append(fm.tracking_time_ms)
        if fm.control_time_ms > 0:
            self._control_times.append(fm.control_time_ms)

        locked = fm.lock_status
        ts = fm.timestamp_s

        if self._prev_locked and not locked:
            self._events.append({"type": "loss", "timestamp_s": ts, "frame_index": fm.frame_index})
            self._loss_start_s = ts
            self._reappear_s = None
            self._loss_events.append({"start_s": ts, "frame_index": fm.frame_index})
        elif not self._prev_locked and locked:
            self._lock_start_time = ts
            self._events.append({
                "type": "acquired",
                "timestamp_s": ts,
                "frame_index": fm.frame_index,
            })
            if self._loss_start_s is not None:
                reacq_dur = ts - self._loss_start_s
                self._reacq_times.append(reacq_dur)
                if self._reappear_s is not None:
                    self._reacq_from_reappearance.append(ts - self._reappear_s)
                self._events.append({
                    "type": "reacquired",
                    "timestamp_s": ts,
                    "frame_index": fm.frame_index,
                    "duration_s": reacq_dur,
                })
                self._loss_start_s = None
                self._reappear_s = None
        # First GT-visible frame inside a loss episode marks reappearance.
        if (self._loss_start_s is not None and self._reappear_s is None
                and fm.true_x is not None and fm.true_y is not None):
            self._reappear_s = ts

        # Frame duration is the only valid duration source for both offline
        # media and live input. Wall-clock processing time is unrelated to
        # the source timeline and must not affect lock/loss metrics.
        self._evaluable_time_s += max(fm.dt, 0.0)
        if locked:
            self._locked_time_s += max(fm.dt, 0.0)
        self._prev_locked = locked

        if self._first_detection_s is None and fm.detected:
            self._first_detection_s = ts
            self._events.append({"type": "first_detection", "timestamp_s": ts, "frame_index": fm.frame_index})

        if not self._acquisition_done and fm.detected:
            if self._acquisition_start_s is None:
                self._acquisition_start_s = ts

        if not self._acquisition_done and locked:
            if self._acquisition_start_s is not None:
                self._stable_acquisition_s = ts - self._acquisition_start_s
                self._acquisition_done = True
                self._acq_count += 1
                self._events.append({
                    "type": "acquisition_complete",
                    "timestamp_s": ts,
                    "frame_index": fm.frame_index,
                    "duration_s": self._stable_acquisition_s,
                })

    def record_event(self, event_type: str, timestamp_s: float, **kwargs: Any) -> None:
        self._events.append({"type": event_type, "timestamp_s": timestamp_s, **kwargs})

    def _categorize_failure(self, fm: FrameMetrics) -> None:
        """Classify measurable frame failures for post-run diagnosis."""
        has_truth = fm.true_x is not None and fm.true_y is not None
        categories: list[str] = []
        if has_truth and not fm.detected:
            categories.append("false_negative")
        if has_truth and fm.detected and fm.error_px is not None:
            if fm.error_px > self._threshold_px:
                categories.append("bad_centroid")
            if not fm.lock_status:
                categories.append("track_loss")
            if fm.error_px > self._threshold_px * 3.0:
                categories.append("association_error")
        if fm.source_fps and fm.source_fps > 0:
            nominal_dt = 1.0 / fm.source_fps
            if fm.dt > nominal_dt * 1.5:
                categories.append("frame_drop")
            if fm.processing_time_ms > nominal_dt * 2000.0:
                categories.append("latency_spike")
        for category in categories:
            self._failure_categories[category] = self._failure_categories.get(category, 0) + 1

    def build_result(self) -> BenchmarkResult:
        frames_with_gt = [f for f in self._frames if f.true_x is not None and f.true_y is not None]
        errors = [f.error_px for f in frames_with_gt if f.error_px is not None]
        component_errors = [
            (f.error_x, f.error_y)
            for f in frames_with_gt
            if f.error_x is not None and f.error_y is not None
        ]
        errors_x = [x for x, _ in component_errors]
        errors_y = [y for _, y in component_errors]

        tracking = TrackingMetricsSummary(frames_with_ground_truth=len(frames_with_gt), frames_evaluated=len(errors))
        if errors:
            ea = np.array(errors)
            exa = np.array(errors_x)
            eya = np.array(errors_y)
            tracking.mean_error_px = float(np.mean(ea))
            tracking.rmse_px = float(np.sqrt(np.mean(ea ** 2)))
            if len(component_errors) > 0:
                tracking.rmse_x_px = float(np.sqrt(np.mean(exa ** 2)))
                tracking.rmse_y_px = float(np.sqrt(np.mean(eya ** 2)))
                tracking.mae_x_px = float(np.mean(np.abs(exa)))
                tracking.mae_y_px = float(np.mean(np.abs(eya)))
            tracking.max_error_px = float(np.max(ea))
            tracking.p50_error_px = float(np.percentile(ea, 50))
            tracking.p90_error_px = float(np.percentile(ea, 90))
            tracking.p95_error_px = float(np.percentile(ea, 95))
            tracking.p99_error_px = float(np.percentile(ea, 99))
            within = sum(1 for e in errors if e <= self._threshold_px)
            tracking.within_threshold_percent = (within / len(errors)) * 100.0

        from fsoc_tracker.benchmark.models import DetectionMetrics
        detection = DetectionMetrics()
        for f in self._frames:
            if f.true_x is None or f.true_y is None:
                # No ground truth: judgeable only as ghost or quiet.
                if f.detected:
                    detection.false_positives += 1
                else:
                    detection.true_negatives += 1
                detection.evaluated_frames += 1
                continue
            detection.evaluated_frames += 1
            if f.detected and f.error_px is not None and f.error_px <= self._threshold_px:
                detection.true_positives += 1
            elif f.detected:
                detection.false_positives += 1
            else:
                detection.false_negatives += 1
        denom_p = detection.true_positives + detection.false_positives
        denom_r = detection.true_positives + detection.false_negatives
        if denom_p > 0:
            detection.precision = detection.true_positives / denom_p
        if denom_r > 0:
            detection.recall = detection.true_positives / denom_r

        from fsoc_tracker.benchmark.models import MOTMetrics
        mot = MOTMetrics()
        gt_frames = [f for f in self._frames
                     if f.true_x is not None and f.true_y is not None]
        mot.gt_frames = len(gt_frames)
        if gt_frames:
            tp_errs = [f.error_px for f in gt_frames
                       if f.detected and f.error_px is not None
                       and f.error_px <= self._threshold_px]
            if tp_errs:
                mot.motp_px = float(sum(tp_errs) / len(tp_errs))
            switches = 0
            last_id = None
            for f in gt_frames:
                matched = (f.detected and f.error_px is not None
                           and f.error_px <= self._threshold_px)
                if matched:
                    if last_id is not None and f.track_id != last_id:
                        switches += 1
                    last_id = f.track_id
            mot.id_switches = switches
            mot.mota = 1.0 - (detection.false_negatives
                              + detection.false_positives
                              + switches) / len(gt_frames)
            idtp = detection.true_positives
            idf_denom = 2 * idtp + detection.false_positives + detection.false_negatives
            if idf_denom > 0:
                mot.idf1 = 2 * idtp / idf_denom

        evaluable = len(frames_with_gt)
        loss = LossMetrics(
            loss_event_count=len(self._loss_events),
            lock_retention_percent=min((self._locked_time_s / max(self._evaluable_time_s, 1e-9)) * 100.0, 100.0)
            if self._evaluable_time_s > 0 else None,
            locked_frames=sum(1 for f in self._frames if f.lock_status),
            evaluable_frames=evaluable,
            total_locked_time_s=self._locked_time_s,
            total_evaluable_time_s=self._evaluable_time_s,
        )
        if self._evaluable_time_s > 0:
            locked_frac = min(self._locked_time_s / self._evaluable_time_s, 1.0)
            loss.loss_rate_percent = (1.0 - locked_frac) * 100.0

        reacq = ReacquisitionMetrics(
            times=self._reacq_times,
            total_events=len(self._reacq_times),
            failed_reacquisitions=len(self._loss_events) - len(self._reacq_times),
        )
        if self._reacq_times:
            ra = np.array(self._reacq_times)
            reacq.mean_s = float(np.mean(ra))
            reacq.median_s = float(np.median(ra))
            reacq.max_s = float(np.max(ra))
            reacq.p95_s = float(np.percentile(ra, 95))
        if self._reacq_from_reappearance:
            reacq.times_from_reappearance = list(self._reacq_from_reappearance)
            reacq.mean_from_reappearance_s = float(
                np.mean(np.array(self._reacq_from_reappearance)))

        perf = PerformanceMetrics(frames_processed=len(self._frames))
        wall = self._wall_end - self._wall_start if self._wall_end > self._wall_start else 0.0
        perf.wall_time_s = wall
        if self._processing_times:
            pt = np.array(self._processing_times)
            perf.mean_processing_ms = float(np.mean(pt))
            perf.p95_processing_ms = float(np.percentile(pt, 95))
            perf.max_processing_ms = float(np.max(pt))
            perf.processing_fps = 1000.0 / max(float(np.mean(pt)), 0.001)
        if wall > 0:
            perf.throughput_fps = len(self._frames) / wall

        latency = PipelineLatency(
            total=LatencyMetrics(
                mean_ms=perf.mean_processing_ms,
                p95_ms=perf.p95_processing_ms,
                max_ms=perf.max_processing_ms,
            ),
        )
        if self._perception_times:
            pt_arr = np.array(self._perception_times)
            latency.perception.mean_ms = float(np.mean(pt_arr))
            latency.perception.p95_ms = float(np.percentile(pt_arr, 95))
        if self._tracking_times:
            tt_arr = np.array(self._tracking_times)
            latency.tracking.mean_ms = float(np.mean(tt_arr))
            latency.tracking.p95_ms = float(np.percentile(tt_arr, 95))
        if self._control_times:
            ct_arr = np.array(self._control_times)
            latency.control.mean_ms = float(np.mean(ct_arr))
            latency.control.p95_ms = float(np.percentile(ct_arr, 95))

        acq = AcquisitionMetrics(
            first_detection_s=self._first_detection_s,
            stable_acquisition_s=self._stable_acquisition_s,
            total_acquisitions=self._acq_count,
        )
        if self._stable_acquisition_s is not None and self._acquisition_start_s is not None:
            acq.acquisition_duration_s = self._stable_acquisition_s

        src_fps = None
        if self._frames:
            ts_list = [f.source_fps for f in self._frames if f.source_fps is not None]
            if ts_list:
                src_fps = float(np.mean(ts_list))

        return BenchmarkResult(
            duration=wall,
            source_fps=src_fps,
            processed_fps=perf.processing_fps,
            frame_count=len(self._frames),
            acquisition=acq,
            tracking=tracking,
            detection=detection,
            mot=mot,
            loss=loss,
            reacquisition=reacq,
            latency=latency,
            performance=perf,
            failure_categories=dict(self._failure_categories),
        )

    def reset(self) -> None:
        self._frames.clear()
        self._events.clear()
        self._prev_timestamp = None
        self._prev_locked = False
        self._lock_start_time = None
        self._locked_time_s = 0.0
        self._evaluable_time_s = 0.0
        self._first_detection_s = None
        self._acquisition_start_s = None
        self._acquisition_done = False
        self._stable_acquisition_s = None
        self._loss_start_s = None
        self._reacq_times.clear()
        self._loss_events.clear()
        self._acq_count = 0
        self._processing_times.clear()
        self._perception_times.clear()
        self._tracking_times.clear()
        self._control_times.clear()
        self._failure_categories.clear()
        self._wall_start = 0.0
        self._wall_end = 0.0
