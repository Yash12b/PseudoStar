"""Benchmark engine — offline/video-frame scoring harness.

Division of labor (do not merge without reason):
- ``benchmark/runner.py`` runs closed-loop SIMULATION benchmarks
  (it owns engine stepping, camera actuation, and link/comm probes).
- THIS module scores pre-recorded frame sequences: external video
  files and synthetic offline series. It never steps a world or moves
  a camera. Ground truth arrives only via explicit providers/sinks.
"""

from __future__ import annotations

import time

import numpy as np

from fsoc_tracker.benchmark.collector import FrameMetrics, MetricsCollector
from fsoc_tracker.benchmark.ground_truth import (
    GroundTruthProvider,
    NullGroundTruthProvider,
)
from fsoc_tracker.benchmark.models import (
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkSession,
    BenchmarkThresholds,
)
from fsoc_tracker.benchmark.profiler import PerformanceProfiler
from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import Frame
from fsoc_tracker.core.time import compute_dt
from fsoc_tracker.perception.base import PerceptionEngine
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.tracking.tracker import KalmanTracker


class BenchmarkEngine:
    """Runs a benchmark session and collects metrics."""

    def __init__(
        self,
        detector: PerceptionEngine | None = None,
        tracker: KalmanTracker | None = None,
        ground_truth: GroundTruthProvider | None = None,
        thresholds: BenchmarkThresholds | None = None,
        tracking_error_threshold_px: float = 10.0,
    ) -> None:
        self._detector = detector or ClassicalBeaconDetector()
        self._tracker = tracker or KalmanTracker()
        self._ground_truth = ground_truth or NullGroundTruthProvider()
        self._thresholds = thresholds or BenchmarkThresholds()
        self._collector = MetricsCollector(tracking_error_threshold_px)
        self._profiler = PerformanceProfiler()

    @property
    def detector(self) -> PerceptionEngine:
        return self._detector

    @property
    def tracker(self) -> KalmanTracker:
        return self._tracker

    @property
    def ground_truth(self) -> GroundTruthProvider:
        return self._ground_truth

    def run_source(
        self,
        source: FrameSource,
        session: BenchmarkSession | None = None,
        max_frames: int | None = None,
    ) -> BenchmarkResult:
        if session is None:
            session = BenchmarkSession(
                mode=BenchmarkMode.EXTERNAL_VIDEO,
                source_name=source.source_id,
                source_fps=source.nominal_fps,
                detector_name=self._detector.name,
            )

        self._collector.reset()
        self._collector.start()
        self._tracker.reset()
        self._profiler.reset()

        frame_idx = 0
        prev_ts: float | None = None

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                break

            dt = (
                compute_dt(
                    frame.timestamp_s,
                    prev_ts,
                    nominal_fps=frame.nominal_fps,
                    fallback_dt=0.0,
                )
                if prev_ts is not None
                else (1.0 / frame.nominal_fps if frame.nominal_fps and frame.nominal_fps > 0 else 0.0)
            )
            prev_ts = frame.timestamp_s

            self._profiler.start_stage("perception")
            det_result = self._detector.detect(frame.image, frame.timestamp_s, frame.frame_index)
            perc_ms = self._profiler.end_stage()

            self._profiler.start_stage("tracking")
            trk_state = self._tracker.update(
                list(det_result.detections),
                frame.timestamp_s,
            )
            trk_ms = self._profiler.end_stage()

            gt_point = self._ground_truth.get(frame.frame_index)

            error_x: float | None = None
            error_y: float | None = None
            error_px: float | None = None
            true_x: float | None = None
            true_y: float | None = None

            if gt_point is not None and gt_point.visible:
                true_x = gt_point.target_x
                true_y = gt_point.target_y
                error_x = trk_state.estimated_x - true_x
                error_y = trk_state.estimated_y - true_y
                error_px = float(np.sqrt(error_x ** 2 + error_y ** 2))

            fm = FrameMetrics(
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                dt=dt,
                source_fps=frame.nominal_fps,
                perception_time_ms=perc_ms,
                tracking_time_ms=trk_ms,
                detected=det_result.detected,
                detection_confidence=det_result.primary_detection.confidence if det_result.primary_detection else 0.0,
                detection_x=det_result.primary_detection.center_x if det_result.primary_detection else 0.0,
                detection_y=det_result.primary_detection.center_y if det_result.primary_detection else 0.0,
                candidate_count=det_result.num_candidates,
                track_x=trk_state.estimated_x,
                track_y=trk_state.estimated_y,
                track_state=trk_state.state.name,
                track_id=trk_state.track_id,
                lock_status=trk_state.locked,
                prediction_only=trk_state.prediction_only,
                velocity_x=trk_state.velocity_x,
                velocity_y=trk_state.velocity_y,
                uncertainty_x=trk_state.uncertainty_x,
                uncertainty_y=trk_state.uncertainty_y,
                residual=trk_state.residual_magnitude,
                true_x=true_x,
                true_y=true_y,
                error_x=error_x,
                error_y=error_y,
                error_px=error_px,
                squared_error=error_x ** 2 + error_y ** 2 if error_x is not None and error_y is not None else None,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
            self._collector.record_frame(fm)
            frame_idx += 1

        self._collector.stop()
        result = self._collector.build_result()
        result.session = session
        result.session.frame_count = frame_idx
        result.session.duration_s = result.duration
        result.session.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.session.status = "completed"
        result.frame_count = frame_idx
        result.detector_name = self._detector.name
        result.threshold_results = self._thresholds.evaluate(result)
        return result

    def run_frames(
        self,
        frames: list[Frame],
        session: BenchmarkSession | None = None,
    ) -> BenchmarkResult:
        if session is None:
            session = BenchmarkSession(
                mode=BenchmarkMode.SYNTHETIC,
                source_name="synthetic_frames",
                detector_name=self._detector.name,
            )

        self._collector.reset()
        self._collector.start()
        self._tracker.reset()
        self._profiler.reset()

        prev_ts: float | None = None
        for frame in frames:
            t0 = time.perf_counter()
            dt = (
                compute_dt(
                    frame.timestamp_s,
                    prev_ts,
                    nominal_fps=frame.nominal_fps,
                    fallback_dt=0.0,
                )
                if prev_ts is not None
                else (1.0 / frame.nominal_fps if frame.nominal_fps and frame.nominal_fps > 0 else 0.0)
            )
            prev_ts = frame.timestamp_s

            self._profiler.start_stage("perception")
            det_result = self._detector.detect(frame.image, frame.timestamp_s, frame.frame_index)
            perc_ms = self._profiler.end_stage()

            self._profiler.start_stage("tracking")
            trk_state = self._tracker.update(
                list(det_result.detections),
                frame.timestamp_s,
            )
            trk_ms = self._profiler.end_stage()

            gt_point = self._ground_truth.get(frame.frame_index)

            error_x = error_y = error_px = true_x = true_y = None
            if gt_point is not None and gt_point.visible:
                true_x = gt_point.target_x
                true_y = gt_point.target_y
                error_x = trk_state.estimated_x - true_x
                error_y = trk_state.estimated_y - true_y
                error_px = float(np.sqrt(error_x ** 2 + error_y ** 2))

            fm = FrameMetrics(
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                dt=dt,
                source_fps=frame.nominal_fps,
                perception_time_ms=perc_ms,
                tracking_time_ms=trk_ms,
                detected=det_result.detected,
                detection_confidence=det_result.primary_detection.confidence if det_result.primary_detection else 0.0,
                detection_x=det_result.primary_detection.center_x if det_result.primary_detection else 0.0,
                detection_y=det_result.primary_detection.center_y if det_result.primary_detection else 0.0,
                candidate_count=det_result.num_candidates,
                track_x=trk_state.estimated_x,
                track_y=trk_state.estimated_y,
                track_state=trk_state.state.name,
                track_id=trk_state.track_id,
                lock_status=trk_state.locked,
                prediction_only=trk_state.prediction_only,
                velocity_x=trk_state.velocity_x,
                velocity_y=trk_state.velocity_y,
                uncertainty_x=trk_state.uncertainty_x,
                uncertainty_y=trk_state.uncertainty_y,
                residual=trk_state.residual_magnitude,
                true_x=true_x,
                true_y=true_y,
                error_x=error_x,
                error_y=error_y,
                error_px=error_px,
                squared_error=error_x ** 2 + error_y ** 2 if error_x is not None and error_y is not None else None,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
            self._collector.record_frame(fm)

        self._collector.stop()
        result = self._collector.build_result()
        result.session = session
        result.session.frame_count = len(frames)
        result.session.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.session.status = "completed"
        result.frame_count = len(frames)
        result.detector_name = self._detector.name
        result.threshold_results = self._thresholds.evaluate(result)
        return result

    def get_profiler(self) -> PerformanceProfiler:
        return self._profiler

    def get_collector(self) -> MetricsCollector:
        return self._collector
