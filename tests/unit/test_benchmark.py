"""Comprehensive tests for Stage 10 — Benchmark Engine.

Test categories:
A. Models (Session, Scenario, Result, Thresholds)
B. MetricsCollector (streaming, frame-by-frame)
C. PerformanceProfiler (stage timing)
D. GroundTruthProvider (synthetic, sidecar, null)
E. VideoBenchmarkSource (OpenCV)
F. LiveCameraSource
G. BenchmarkEngine (synthetic pipeline)
H. BenchmarkEngine (external video pipeline)
I. BenchmarkThresholds (SIH evaluation)
J. Export (CSV, JSON, event log)
K. Report (HTML)
L. Plots
M. Experiment runner
N. CI regression benchmark
O. End-to-end synthetic benchmark
P. End-to-end external video benchmark
Q. FPS independence
R. Coordinate mapping
S. Deterministic replay
T. Performance budgets
U. SIH scorecard
V. Edge cases
W. Compositional (all backends)
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from fsoc_tracker.benchmark.collector import FrameMetrics, MetricsCollector
from fsoc_tracker.benchmark.engine import BenchmarkEngine
from fsoc_tracker.benchmark.export import export_csv, export_event_log, export_json
from fsoc_tracker.benchmark.ground_truth import (
    GroundTruthPoint,
    NullGroundTruthProvider,
    SidecarGroundTruthProvider,
    SyntheticGroundTruthProvider,
)
from fsoc_tracker.benchmark.live_source import LiveCameraSource
from fsoc_tracker.benchmark.models import (
    AcquisitionMetrics,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkScenario,
    BenchmarkSession,
    BenchmarkThresholds,
    LossMetrics,
    PerformanceMetrics,
    ReacquisitionMetrics,
    ThresholdVerdict,
    TrackingMetricsSummary,
)
from fsoc_tracker.benchmark.plots import generate_plots
from fsoc_tracker.benchmark.profiler import PerformanceProfiler
from fsoc_tracker.benchmark.report import generate_html_report
from fsoc_tracker.benchmark.run import ci_regression_benchmark, run_scenario
from fsoc_tracker.benchmark.video_source import VideoBenchmarkSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.pipeline.sources import VideoSource
from fsoc_tracker.tracking.tracker import KalmanTracker

# ---------------------------------------------------------------------------
# Helpers

def _beacon_frame(
    w: int = 640, h: int = 480,
    cx: float = 320, cy: float = 240,
    size: float = 10.0, brightness: float = 200.0,
    ts: float = 0.0, idx: int = 0,
    noise_sigma: float = 3.0,
) -> Frame:
    rng = np.random.default_rng(idx + 42)
    img = np.full((h, w), 5.0, dtype=np.float64)
    sigma = size / 3.0
    for y in range(max(0, int(cy) - 15), min(h, int(cy) + 15)):
        for x in range(max(0, int(cx) - 15), min(w, int(cx) + 15)):
            d = (x - cx) ** 2 + (y - cy) ** 2
            img[y, x] += brightness * math.exp(-d / (2 * sigma * sigma))
    img += rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Frame(
        image=img, width=w, height=h, channels=1,
        color_model=ColorModel.GRAY, source_id="test",
        source_type=SourceType.SYNTHETIC,
        frame_index=idx, timestamp_s=ts, nominal_fps=30.0,
    )


def _dark_frame(
    w: int = 640, h: int = 480,
    ts: float = 0.0, idx: int = 0,
) -> Frame:
    rng = np.random.default_rng(idx + 99)
    img = np.full((h, w), 5.0, dtype=np.float64)
    img += rng.normal(0, 2, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Frame(
        image=img, width=w, height=h, channels=1,
        color_model=ColorModel.GRAY, source_id="test",
        source_type=SourceType.SYNTHETIC,
        frame_index=idx, timestamp_s=ts, nominal_fps=30.0,
    )


def _motion_frames(
    num_frames: int = 100,
    w: int = 320, h: int = 240,
    fps: float = 30.0,
) -> list[Frame]:
    frames = []
    for i in range(num_frames):
        ts = i / fps
        cx = w // 2 + int(w * 0.15 * math.sin(2 * math.pi * i / num_frames))
        cy = h // 2 + int(h * 0.1 * math.cos(2 * math.pi * i / (num_frames * 0.7)))
        frames.append(_beacon_frame(w, h, cx, cy, 10.0, 200.0, ts, i))
    return frames


# ---------------------------------------------------------------------------
# A. Models

class TestModels:
    def test_benchmark_session_defaults(self):
        s = BenchmarkSession()
        assert s.session_id
        assert s.mode == BenchmarkMode.SYNTHETIC
        assert s.status == "pending"

    def test_benchmark_session_to_metadata(self):
        s = BenchmarkSession(session_id="abc123", source_fps=30.0)
        m = s.to_metadata()
        assert m.session_id == "abc123"
        assert m.source_fps == 30.0

    def test_benchmark_scenario(self):
        sc = BenchmarkScenario(name="test_scenario", mode=BenchmarkMode.EXTERNAL_VIDEO)
        assert sc.name == "test_scenario"
        assert sc.mode == BenchmarkMode.EXTERNAL_VIDEO

    def test_benchmark_thresholds_default(self):
        t = BenchmarkThresholds()
        assert t.acquisition_max_s == 2.0
        assert t.tracking_error_max_px == 10.0
        assert t.target_loss_max_percent == 5.0
        assert t.reacquisition_max_s == 1.0
        assert t.minimum_processing_fps == 20.0

    def test_benchmark_thresholds_evaluate(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 1.5
        r.tracking.rmse_px = 8.0
        r.loss.loss_rate_percent = 3.0
        r.reacquisition.mean_s = 0.5
        r.performance.processing_fps = 60.0
        results = t.evaluate(r)
        assert len(results) == 5
        assert all(isinstance(rr.verdict, ThresholdVerdict) for rr in results)

    def test_threshold_verdict_pass(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 1.0
        r.tracking.rmse_px = 5.0
        r.loss.loss_rate_percent = 1.0
        r.reacquisition.mean_s = 0.3
        r.reacquisition.total_events = 1
        r.performance.processing_fps = 100.0
        results = t.evaluate(r)
        passes = [rr for rr in results if rr.verdict == ThresholdVerdict.PASS]
        assert len(passes) >= 4

    def test_threshold_verdict_all_fail(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 5.0
        r.tracking.rmse_px = 20.0
        r.loss.loss_rate_percent = 10.0
        r.reacquisition.mean_s = 3.0
        r.reacquisition.total_events = 1
        r.performance.processing_fps = 10.0
        results = t.evaluate(r)
        fails_or_ne = [rr for rr in results if rr.verdict in (ThresholdVerdict.FAIL, ThresholdVerdict.NOT_EVALUABLE)]
        assert len(fails_or_ne) >= 4

    def test_threshold_verdict_not_evaluable(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        results = t.evaluate(r)
        ne = [rr for rr in results if rr.verdict == ThresholdVerdict.NOT_EVALUABLE]
        assert len(ne) >= 3

    def test_benchmark_result_summary_table(self):
        r = BenchmarkResult()
        table = r.summary_table()
        assert "Metric" in table
        assert "Result" in table

    def test_acquisition_metrics(self):
        a = AcquisitionMetrics(first_detection_s=0.5, stable_acquisition_s=1.2)
        assert a.first_detection_s == 0.5

    def test_tracking_metrics_summary(self):
        t = TrackingMetricsSummary(rmse_px=5.0, within_threshold_percent=95.0)
        assert t.rmse_px == 5.0

    def test_loss_metrics(self):
        l = LossMetrics(loss_rate_percent=2.0, lock_retention_percent=98.0)
        assert l.loss_rate_percent == 2.0

    def test_reacquisition_metrics(self):
        r = ReacquisitionMetrics(times=[0.5, 0.8, 1.2], mean_s=0.83)
        assert r.mean_s == 0.83

    def test_performance_metrics(self):
        p = PerformanceMetrics(processing_fps=60.0, mean_processing_ms=16.0)
        assert p.processing_fps == 60.0


# ---------------------------------------------------------------------------
# B. MetricsCollector

class TestMetricsCollector:
    def test_empty_collector(self):
        c = MetricsCollector()
        r = c.build_result()
        assert r.frame_count == 0

    def test_single_frame(self):
        c = MetricsCollector()
        c.start()
        fm = FrameMetrics(frame_index=0, timestamp_s=0.0, dt=0.033, lock_status=False, error_px=5.0)
        c.record_frame(fm)
        c.stop()
        r = c.build_result()
        assert r.frame_count == 1

    def test_lock_detection(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0, lock_status=False))
        c.record_frame(FrameMetrics(frame_index=1, timestamp_s=0.033, lock_status=True))
        c.record_frame(FrameMetrics(frame_index=2, timestamp_s=0.066, lock_status=True))
        c.record_frame(FrameMetrics(frame_index=3, timestamp_s=0.1, lock_status=False))
        c.record_frame(FrameMetrics(frame_index=4, timestamp_s=0.133, lock_status=True))
        c.stop()
        r = c.build_result()
        assert r.loss.loss_event_count >= 1
        assert r.loss.locked_frames >= 2

    def test_ground_truth_error(self):
        c = MetricsCollector(tracking_error_threshold_px=10.0)
        c.start()
        for i in range(50):
            c.record_frame(FrameMetrics(
                frame_index=i, timestamp_s=i * 0.033,
                lock_status=True, true_x=100.0, true_y=100.0,
                error_x=float(i % 5), error_y=float(i % 3),
                error_px=float(math.sqrt((i % 5) ** 2 + (i % 3) ** 2)),
            ))
        c.stop()
        r = c.build_result()
        assert r.tracking.frames_with_ground_truth == 50
        assert r.tracking.rmse_px is not None
        assert r.tracking.rmse_px >= 0

    def test_acquisition_detection(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0, detected=False))
        c.record_frame(FrameMetrics(frame_index=1, timestamp_s=0.033, detected=True))
        c.record_frame(FrameMetrics(frame_index=2, timestamp_s=0.066, detected=True, lock_status=True))
        c.stop()
        r = c.build_result()
        assert r.acquisition.first_detection_s is not None
        assert r.acquisition.stable_acquisition_s is not None

    def test_reacquisition_detection(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0, lock_status=True))
        c.record_frame(FrameMetrics(frame_index=1, timestamp_s=0.033, lock_status=True))
        c.record_frame(FrameMetrics(frame_index=2, timestamp_s=0.066, lock_status=False))
        c.record_frame(FrameMetrics(frame_index=3, timestamp_s=0.1, lock_status=False))
        c.record_frame(FrameMetrics(frame_index=4, timestamp_s=0.133, lock_status=True))
        c.stop()
        r = c.build_result()
        assert r.reacquisition.total_events >= 1

    def test_performance_metrics(self):
        c = MetricsCollector()
        c.start()
        for i in range(10):
            c.record_frame(FrameMetrics(
                frame_index=i, timestamp_s=i * 0.033,
                processing_time_ms=10.0 + i,
            ))
        c.stop()
        r = c.build_result()
        assert r.performance.processing_fps is not None
        assert r.performance.processing_fps > 0

    def test_reset(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0))
        c.stop()
        c.reset()
        r = c.build_result()
        assert r.frame_count == 0

    def test_failure_categories(self):
        c = MetricsCollector(tracking_error_threshold_px=10.0)
        c.record_frame(FrameMetrics(
            frame_index=0, timestamp_s=0.0, dt=1 / 30, source_fps=30.0,
            detected=False, lock_status=False, true_x=10.0, true_y=10.0,
            processing_time_ms=1.0,
        ))
        c.record_frame(FrameMetrics(
            frame_index=1, timestamp_s=0.2, dt=0.2, source_fps=30.0,
            detected=True, lock_status=False, true_x=10.0, true_y=10.0,
            error_px=40.0, processing_time_ms=100.0,
        ))
        result = c.build_result()
        assert result.failure_categories == {
            "false_negative": 1,
            "bad_centroid": 1,
            "track_loss": 1,
            "association_error": 1,
            "frame_drop": 1,
            "latency_spike": 1,
        }

    def test_events(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0, detected=False))
        c.record_frame(FrameMetrics(frame_index=1, timestamp_s=0.033, detected=True))
        c.record_frame(FrameMetrics(frame_index=2, timestamp_s=0.066, lock_status=True))
        c.stop()
        assert len(c.events) >= 1

    def test_lock_retention(self):
        c = MetricsCollector()
        c.start()
        for i in range(10):
            c.record_frame(FrameMetrics(
                frame_index=i, timestamp_s=i * 0.033,
                lock_status=True, dt=0.033,
            ))
        c.stop()
        r = c.build_result()
        assert r.loss.lock_retention_percent is not None
        assert r.loss.lock_retention_percent >= 99.0

    def test_lock_metrics_use_source_time_not_processing_clock(self):
        c = MetricsCollector()
        c.start()
        c.record_frame(FrameMetrics(frame_index=0, timestamp_s=0.0, dt=0.5, lock_status=True))
        c.record_frame(FrameMetrics(frame_index=1, timestamp_s=0.5, dt=0.5, lock_status=False))
        time.sleep(0.01)
        c.stop()
        result = c.build_result()
        assert result.loss.total_locked_time_s == pytest.approx(0.5)
        assert result.loss.total_evaluable_time_s == pytest.approx(1.0)
        assert result.loss.loss_rate_percent == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# C. PerformanceProfiler

class TestProfiler:
    def test_record_stage(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 10.0)
        stats = p.get_stats("perception")
        assert stats.mean_ms == 10.0

    def test_start_end_stage(self):
        p = PerformanceProfiler()
        p.start_stage("tracking")
        time.sleep(0.001)
        elapsed = p.end_stage()
        assert elapsed > 0

    def test_get_all_stats(self):
        p = PerformanceProfiler()
        for stage in ("preprocessing", "perception", "tracking", "control"):
            p.record_stage(stage, 5.0)
        stats = p.get_all_stats()
        assert len(stats) == 7

    def test_frame_timings(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 10.0)
        p.record_stage("tracking", 5.0)
        t = p.get_frame_timings()
        assert t.perception_ms == 10.0
        assert t.tracking_ms == 5.0

    def test_check_budget(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 100.0)
        violations = p.check_budget({"perception": 50.0})
        assert len(violations) == 1
        assert violations[0]["overrun_percent"] > 0

    def test_check_budget_no_violation(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 10.0)
        violations = p.check_budget({"perception": 50.0})
        assert len(violations) == 0

    def test_reset(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 10.0)
        p.reset()
        stats = p.get_stats("perception")
        assert stats.count == 0


# ---------------------------------------------------------------------------
# D. GroundTruthProvider

class TestGroundTruth:
    def test_null_provider(self):
        p = NullGroundTruthProvider()
        assert not p.has_ground_truth
        assert p.get(0) is None

    def test_synthetic_provider(self):
        p = SyntheticGroundTruthProvider()
        p.add_point(GroundTruthPoint(frame_index=0, target_x=100.0, target_y=200.0))
        p.add_point(GroundTruthPoint(frame_index=5, target_x=150.0, target_y=250.0))
        assert p.has_ground_truth
        assert p.frame_count == 2
        pt = p.get(0)
        assert pt is not None
        assert pt.target_x == 100.0
        assert p.get(5) is not None
        assert p.get(3) is None

    def test_synthetic_provider_range(self):
        p = SyntheticGroundTruthProvider()
        for i in range(10):
            p.add_point(GroundTruthPoint(frame_index=i, target_x=float(i * 10)))
        pts = p.get_range(0, 10)
        assert len(pts) == 10
        assert pts[3].target_x == 30.0

    def test_sidecar_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"frame_index": 0, "x": 100, "y": 200},
                {"frame_index": 5, "x": 150, "y": 250},
            ], f)
            f.flush()
            p = SidecarGroundTruthProvider(f.name)
            assert p.has_ground_truth
            assert p.frame_count == 2
            pt = p.get(0)
            assert pt is not None
            assert pt.target_x == 100.0

    def test_sidecar_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"frame_index": 0, "x": 10, "y": 20}\n')
            f.write('{"frame_index": 1, "x": 30, "y": 40}\n')
            f.flush()
            p = SidecarGroundTruthProvider(f.name)
            assert p.has_ground_truth
            assert p.get(0).target_x == 10.0

    def test_sidecar_missing_file(self):
        p = SidecarGroundTruthProvider("/nonexistent/path.json")
        assert not p.has_ground_truth


# ---------------------------------------------------------------------------
# E. VideoBenchmarkSource

class TestVideoSource:
    def _create_test_video(self, path: str, num_frames: int = 30, fps: float = 30.0) -> None:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (64, 48), isColor=False)
        rng = np.random.default_rng(42)
        for i in range(num_frames):
            img = rng.integers(0, 50, (48, 64), dtype=np.uint8)
            cx, cy = 32 + int(10 * math.sin(2 * math.pi * i / num_frames)), 24
            cv2.circle(img, (cx, cy), 5, 200, -1)
            writer.write(img)
        writer.release()

    def test_video_readable(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 30)
        src = VideoBenchmarkSource(path)
        src.open()
        assert src.is_open()
        assert src.nominal_fps is not None
        frames = []
        while True:
            frame = src.read()
            if frame is None:
                break
            frames.append(frame)
        src.release()
        assert len(frames) == 30

    def test_video_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 10)
        src = VideoBenchmarkSource(path)
        src.open()
        assert src.width == 64
        assert src.height == 48
        assert src.nominal_fps is not None
        src.release()

    def test_video_missing_file(self):
        src = VideoBenchmarkSource("/nonexistent/video.mp4")
        with pytest.raises((FileNotFoundError, RuntimeError)):
            src.open()

    def test_video_frame_dimensions(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 5)
        src = VideoBenchmarkSource(path)
        src.open()
        frame = src.read()
        assert frame is not None
        assert frame.width == 64
        assert frame.height == 48
        src.release()

    def test_video_timestamps_follow_media_rate(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 6, fps=60.0)
        src = VideoSource(path)
        src.open()
        frames = [src.read() for _ in range(3)]
        src.release()
        assert all(frame is not None for frame in frames)
        timestamps = [frame.timestamp_s for frame in frames if frame is not None]
        assert timestamps[0] == pytest.approx(0.0, abs=0.02)
        assert timestamps[1] - timestamps[0] == pytest.approx(1.0 / 60.0, abs=0.02)
        assert timestamps[2] - timestamps[1] == pytest.approx(1.0 / 60.0, abs=0.02)
        assert frames[0].metadata["timestamp_source"] != "wall_clock"

    def test_video_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 5)
        with VideoBenchmarkSource(path) as src:
            frame = src.read()
            assert frame is not None

    def test_video_seek(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 20)
        src = VideoBenchmarkSource(path)
        src.open()
        result = src.seek(10)
        assert result
        frame = src.read()
        assert frame is not None
        assert frame.frame_index == 10
        src.release()


# ---------------------------------------------------------------------------
# F. LiveCameraSource (skip if no hardware)

class TestLiveCamera:
    def test_source_interface(self):
        src = LiveCameraSource(device_index=0)
        assert src.source_id == "camera:0"
        assert src.nominal_fps == 30.0

    def test_source_not_open_initially(self):
        src = LiveCameraSource(device_index=0)
        assert not src.is_open()


# ---------------------------------------------------------------------------
# G. BenchmarkEngine (synthetic pipeline)

class TestBenchmarkEngine:
    def test_run_frames(self):
        frames = _motion_frames(50)
        engine = BenchmarkEngine(
            detector=ClassicalBeaconDetector(),
            tracker=KalmanTracker(),
        )
        result = engine.run_frames(frames)
        assert result.frame_count == 50
        assert result.duration >= 0

    def test_run_frames_with_ground_truth(self):
        frames = _motion_frames(50)
        gt = SyntheticGroundTruthProvider()
        for f in frames:
            gt.add_point(GroundTruthPoint(frame_index=f.frame_index, timestamp_s=f.timestamp_s, target_x=320.0, target_y=240.0))
        engine = BenchmarkEngine(
            detector=ClassicalBeaconDetector(),
            tracker=KalmanTracker(),
            ground_truth=gt,
        )
        result = engine.run_frames(frames)
        assert result.tracking.frames_with_ground_truth == 50

    def test_run_frames_custom_session(self):
        frames = _motion_frames(20)
        session = BenchmarkSession(session_id="test123", source_name="unit_test")
        engine = BenchmarkEngine()
        result = engine.run_frames(frames, session=session)
        assert result.session.session_id == "test123"

    def test_profiler_available(self):
        engine = BenchmarkEngine()
        assert engine.get_profiler() is not None

    def test_collector_available(self):
        engine = BenchmarkEngine()
        assert engine.get_collector() is not None


# ---------------------------------------------------------------------------
# H. BenchmarkEngine (external video pipeline)

class TestBenchmarkEngineVideo:
    def _create_test_video(self, path: str, num_frames: int = 30) -> None:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (64, 48), isColor=False)
        rng = np.random.default_rng(42)
        for i in range(num_frames):
            img = rng.integers(0, 30, (48, 64), dtype=np.uint8)
            cx = 32 + int(10 * math.sin(2 * math.pi * i / num_frames))
            cv2.circle(img, (cx, 24), 4, 200, -1)
            writer.write(img)
        writer.release()

    def test_run_source(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_test_video(path, 30)
        with VideoBenchmarkSource(path) as src:
            engine = BenchmarkEngine()
            result = engine.run_source(src)
        assert result.frame_count == 30


# ---------------------------------------------------------------------------
# I. BenchmarkThresholds

class TestThresholds:
    def test_all_pass(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 0.5
        r.tracking.rmse_px = 3.0
        r.loss.loss_rate_percent = 1.0
        r.reacquisition.mean_s = 0.3
        r.performance.processing_fps = 100.0
        results = t.evaluate(r)
        assert all(rr.verdict == ThresholdVerdict.PASS for rr in results)

    def test_all_fail(self):
        t = BenchmarkThresholds()
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 10.0
        r.tracking.rmse_px = 50.0
        r.loss.loss_rate_percent = 20.0
        r.reacquisition.mean_s = 5.0
        r.performance.processing_fps = 5.0
        results = t.evaluate(r)
        assert all(rr.verdict == ThresholdVerdict.FAIL for rr in results)

    def test_custom_thresholds(self):
        t = BenchmarkThresholds(acquisition_max_s=5.0, tracking_error_max_px=20.0)
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 3.0
        r.tracking.rmse_px = 15.0
        results = t.evaluate(r)
        acq_result = [rr for rr in results if rr.name == "acquisition_time"][0]
        assert acq_result.verdict == ThresholdVerdict.PASS


# ---------------------------------------------------------------------------
# J. Export

class TestExport:
    def test_export_json(self):
        r = BenchmarkResult()
        r.frame_count = 100
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_json(r, path)
        with open(path) as f:
            data = json.load(f)
        assert "session" in data
        assert "metrics" in data
        assert "thresholds" in data

    def test_export_csv(self):
        frames = [
            FrameMetrics(frame_index=i, timestamp_s=i * 0.033, error_px=float(i))
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_csv(frames, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 11  # header + 10 rows

    def test_export_event_log(self):
        events = [{"type": "acquired", "timestamp_s": 0.5}, {"type": "lost", "timestamp_s": 1.0}]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_event_log(events, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows

    def test_export_json_creates_dirs(self):
        r = BenchmarkResult()
        path = tempfile.mkdtemp() + "/sub/dir/result.json"
        export_json(r, path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# K. Report

class TestReport:
    def test_html_report(self):
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 1.5
        r.tracking.rmse_px = 5.0
        r.loss.loss_rate_percent = 2.0
        r.reacquisition.mean_s = 0.5
        r.performance.processing_fps = 60.0
        r.loss.lock_retention_percent = 98.0
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        generate_html_report(r, path)
        content = Path(path).read_text()
        assert "SIH26169" in content
        assert "Acquisition" in content


# ---------------------------------------------------------------------------
# L. Plots

class TestPlots:
    def test_generate_plots(self):
        r = BenchmarkResult()
        frames = [
            FrameMetrics(frame_index=i, timestamp_s=i * 0.033, error_px=float(i % 10),
                         error_x=float(i % 5), error_y=float(i % 3),
                         track_x=100.0 + i, track_y=200.0 + i,
                         true_x=100.0, true_y=200.0,
                         processing_time_ms=10.0 + i % 5)
            for i in range(50)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            plots = generate_plots(r, frames, tmpdir)
            if find_spec("matplotlib") is None:
                assert len(plots) == 0
            else:
                assert len(plots) > 0
                for p in plots:
                    assert Path(p).exists()


# ---------------------------------------------------------------------------
# M. Experiment runner

class TestRunner:
    def test_run_scenario(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            vpath = f.name
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vpath, fourcc, 30.0, (64, 48), isColor=False)
        for i in range(20):
            img = np.zeros((48, 64), dtype=np.uint8)
            cv2.circle(img, (32, 24), 4, 200, -1)
            writer.write(img)
        writer.release()

        scenario = BenchmarkScenario(name="test", source_path=vpath, mode=BenchmarkMode.EXTERNAL_VIDEO)
        source = VideoBenchmarkSource(vpath)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_scenario(scenario, source, output_dir=tmpdir)
            assert result.frame_count == 20
            assert Path(tmpdir + "/benchmark_result.json").exists()
            assert Path(tmpdir + "/frame_metrics.csv").exists()

    def test_ci_regression(self):
        result = ci_regression_benchmark()
        assert result.frame_count == 150
        assert result.performance.throughput_fps is not None
        assert result.performance.throughput_fps > 0


# ---------------------------------------------------------------------------
# N. CI regression

class TestCIRegression:
    def test_deterministic(self):
        r1 = ci_regression_benchmark()
        r2 = ci_regression_benchmark()
        assert r1.frame_count == r2.frame_count

    def test_performance(self):
        result = ci_regression_benchmark()
        assert result.performance.throughput_fps is not None
        assert result.performance.throughput_fps > 0


# ---------------------------------------------------------------------------
# O. End-to-end synthetic benchmark

class TestEndToEndSynthetic:
    def test_full_pipeline(self):
        frames = _motion_frames(100, w=320, h=240)
        gt = SyntheticGroundTruthProvider()
        for f in frames:
            cx = 160 + int(48 * math.sin(2 * math.pi * f.frame_index / 100))
            cy = 120 + int(24 * math.cos(2 * math.pi * f.frame_index / 70))
            gt.add_point(GroundTruthPoint(
                frame_index=f.frame_index, timestamp_s=f.timestamp_s,
                target_x=float(cx), target_y=float(cy),
            ))

        engine = BenchmarkEngine(
            detector=ClassicalBeaconDetector(),
            tracker=KalmanTracker(),
            ground_truth=gt,
        )
        result = engine.run_frames(frames)
        assert result.frame_count == 100
        assert result.tracking.frames_with_ground_truth == 100
        assert result.tracking.rmse_px is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            export_json(result, f"{tmpdir}/result.json")
            export_csv(engine.get_collector().frames, f"{tmpdir}/frames.csv")
            generate_html_report(result, f"{tmpdir}/report.html")
            plots = generate_plots(result, engine.get_collector().frames, f"{tmpdir}/plots")
            assert Path(f"{tmpdir}/result.json").exists()
            assert len(plots) > 0


# ---------------------------------------------------------------------------
# P. End-to-end external video benchmark

class TestEndToEndVideo:
    def _create_video(self, path: str, n: int = 30) -> None:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (128, 96), isColor=False)
        rng = np.random.default_rng(42)
        for i in range(n):
            img = rng.integers(0, 30, (96, 128), dtype=np.uint8)
            cx = 64 + int(20 * math.sin(2 * math.pi * i / n))
            cv2.circle(img, (cx, 48), 6, 220, -1)
            writer.write(img)
        writer.release()

    def test_video_benchmark(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        self._create_video(path, 30)

        with VideoBenchmarkSource(path) as src:
            engine = BenchmarkEngine(
                detector=ClassicalBeaconDetector(),
                tracker=KalmanTracker(),
            )
            result = engine.run_source(src)

        assert result.frame_count == 30
        assert result.performance.processing_fps is not None


# ---------------------------------------------------------------------------
# Q. FPS independence

class TestFPSIndependence:
    @pytest.mark.parametrize("fps", [5, 10, 15, 24, 30, 60])
    def test_various_fps(self, fps: int):
        frames = _motion_frames(30, w=160, h=120, fps=float(fps))
        engine = BenchmarkEngine()
        result = engine.run_frames(frames)
        assert result.frame_count == 30
        assert result.source_fps is not None

    def test_irregular_timestamps(self):
        frames = []
        ts = 0.0
        for i in range(50):
            dt = 0.033 + 0.005 * math.sin(i)
            ts += dt
            frames.append(_beacon_frame(160, 120, 80, 60, ts=ts, idx=i))
        engine = BenchmarkEngine()
        result = engine.run_frames(frames)
        assert result.frame_count == 50


# ---------------------------------------------------------------------------
# R. Coordinate mapping

class TestCoordinateMapping:
    def test_video_to_frame_coordinates(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240), isColor=False)
        for i in range(5):
            img = np.zeros((240, 320), dtype=np.uint8)
            cv2.circle(img, (160, 120), 8, 200, -1)
            writer.write(img)
        writer.release()

        with VideoBenchmarkSource(path) as src:
            frame = src.read()
            assert frame is not None
            assert frame.width == 320
            assert frame.height == 240
            result = ClassicalBeaconDetector().detect(frame.image)
            if result.detected:
                assert 0 <= result.primary_detection.center_x <= 320
                assert 0 <= result.primary_detection.center_y <= 240


# ---------------------------------------------------------------------------
# S. Deterministic replay

class TestDeterminism:
    def test_same_seed_same_result(self):
        r1 = ci_regression_benchmark()
        r2 = ci_regression_benchmark()
        assert r1.frame_count == r2.frame_count
        assert r1.acquisition.stable_acquisition_s == r2.acquisition.stable_acquisition_s


# ---------------------------------------------------------------------------
# T. Performance budgets

class TestPerformanceBudget:
    def test_budget_violations(self):
        p = PerformanceProfiler()
        p.record_stage("perception", 100.0)
        p.record_stage("tracking", 5.0)
        violations = p.check_budget({"perception": 50.0, "tracking": 10.0})
        assert len(violations) == 1
        assert violations[0]["stage"] == "perception"


# ---------------------------------------------------------------------------
# U. SIH scorecard

class TestSIHScorecard:
    def test_scorecard_generation(self):
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 1.5
        r.tracking.rmse_px = 4.3
        r.tracking.p95_error_px = 8.1
        r.tracking.max_error_px = 12.2
        r.tracking.within_threshold_percent = 92.0
        r.loss.loss_rate_percent = 2.1
        r.loss.lock_retention_percent = 97.9
        r.reacquisition.mean_s = 0.46
        r.performance.processing_fps = 63.2
        r.threshold_results = BenchmarkThresholds().evaluate(r)
        table = r.summary_table()
        assert "4.30" in table
        assert "PASS" in table or "FAIL" in table or "NOT_EVALUABLE" in table


# ---------------------------------------------------------------------------
# U2. Threshold verdict integrity (migrated from retired SIH suite:
# NOT_EVAL semantics live in BenchmarkThresholds, not scenario labels)

class TestThresholdVerdicts:
    def test_clean_result_marks_reacquisition_not_evaluable_without_loss(self):
        r = BenchmarkResult()
        verdicts = {v.name: v.verdict
                    for v in BenchmarkThresholds().evaluate(r)}
        assert verdicts["reacquisition_time"] == ThresholdVerdict.NOT_EVALUABLE
        assert verdicts["tracking_error"] == ThresholdVerdict.NOT_EVALUABLE


# ---------------------------------------------------------------------------
# V. Edge cases

class TestEdgeCases:
    def test_empty_frames(self):
        engine = BenchmarkEngine()
        result = engine.run_frames([])
        assert result.frame_count == 0

    def test_single_frame(self):
        frames = [_beacon_frame(ts=0.0, idx=0)]
        engine = BenchmarkEngine()
        result = engine.run_frames(frames)
        assert result.frame_count == 1

    def test_no_ground_truth(self):
        frames = _motion_frames(20)
        engine = BenchmarkEngine(ground_truth=NullGroundTruthProvider())
        result = engine.run_frames(frames)
        assert result.tracking.frames_with_ground_truth == 0
        assert result.tracking.rmse_px is None

    def test_video_missing_file(self):
        src = VideoBenchmarkSource("/nonexistent.mp4")
        engine = BenchmarkEngine()
        try:
            result = engine.run_source(src)
            assert result.frame_count == 0
        except (FileNotFoundError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# W. Compositional

class TestCompositional:
    def test_all_export_formats(self):
        r = BenchmarkResult()
        r.acquisition.stable_acquisition_s = 1.0
        frames = [FrameMetrics(frame_index=i, timestamp_s=i * 0.033) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            export_json(r, f"{tmpdir}/result.json")
            export_csv(frames, f"{tmpdir}/frames.csv")
            export_event_log([{"type": "test"}], f"{tmpdir}/events.csv")
            generate_html_report(r, f"{tmpdir}/report.html")
            plots = generate_plots(r, frames, f"{tmpdir}/plots")
            assert Path(f"{tmpdir}/result.json").exists()
            assert Path(f"{tmpdir}/frames.csv").exists()
            assert Path(f"{tmpdir}/events.csv").exists()
            assert Path(f"{tmpdir}/report.html").exists()

    def test_engine_with_all_backends(self):
        frames = _motion_frames(30)
        for name, det in [("classical_bright_spot", ClassicalBeaconDetector())]:
            engine = BenchmarkEngine(detector=det, tracker=KalmanTracker())
            result = engine.run_frames(frames)
            assert result.detector_name == name


# ---------------------------------------------------------------------------
# Q. Closed-loop method benchmark (PS metrics)

class TestBenchmarkMethods:
    def test_all_five_methods_listed(self):
        from fsoc_tracker.benchmark.methods import METHOD_LABELS, METHOD_ORDER
        assert len(METHOD_ORDER) == 5
        assert set(METHOD_LABELS) == set(METHOD_ORDER)

    def test_real_artifacts_all_available(self):
        from fsoc_tracker.benchmark.methods import METHOD_ORDER, check_method_availability
        for method in METHOD_ORDER:
            availability = check_method_availability(method)
            assert availability.available, f"{method}: {availability.reason}"

    def test_missing_artifacts_unavailable(self, tmp_path, monkeypatch):
        from fsoc_tracker.benchmark.methods import check_method_availability
        monkeypatch.setenv("FSOC_BENCHMARK_MODEL_DIR", str(tmp_path))
        availability = check_method_availability("learned_temporal_learned_policy")
        assert not availability.available

    def test_load_refuses_unavailable(self, tmp_path, monkeypatch):
        from fsoc_tracker.benchmark.methods import (
            MethodUnavailableError,
            load_method_components,
        )
        monkeypatch.setenv("FSOC_BENCHMARK_MODEL_DIR", str(tmp_path))
        with pytest.raises(MethodUnavailableError):
            load_method_components("learned_temporal_learned_policy")

    def test_stale_policy_labels_unavailable(self, tmp_path, monkeypatch):
        import shutil

        from fsoc_tracker.ai.learned import LearnedFeatureClassifier
        from fsoc_tracker.benchmark.methods import check_method_availability
        shutil.copy(
            "artifacts/models/mission-v3/motion-v2.npz", tmp_path / "motion-v2.npz"
        )
        real = LearnedFeatureClassifier.load("artifacts/models/mission-v3/policy-v2.npz")
        stale = LearnedFeatureClassifier(["track", "hold"])
        stale._weights = real._weights[:, :2]
        stale.save(tmp_path / "policy-v2.npz")
        monkeypatch.setenv("FSOC_BENCHMARK_MODEL_DIR", str(tmp_path))
        availability = check_method_availability("learned_temporal_learned_policy")
        assert not availability.available
        assert "stale" in availability.reason


REQUIRED_RECORD_FIELDS = [
    "method", "scenario", "seed", "source", "duration_s", "fps",
    "latency_ms", "acquisition_s", "rmse_px", "mae_px", "p95_px",
    "max_error_px", "loss_rate_pct", "reacquisition_s",
    "lock_retention_pct", "fov_retention_pct", "search_duration_s",
    "link_downtime_s", "messages_delivered", "messages_total",
    "safety_events", "provenance", "repro_command",
    "log_json_path", "log_csv_path",
]


def _run_cfg(mode, frames=25, output_dir=None, **kwargs):
    from fsoc_tracker.benchmark.runner import BenchmarkRunConfig
    return BenchmarkRunConfig(
        mode=mode, world_profile="multi", seed=42, max_frames=frames,
        output_dir=output_dir or tempfile.mkdtemp(), **kwargs,
    )


class TestGuiRunner:
    def test_kalman_record_schema(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkRunner().run(_run_cfg(BenchmarkMode.KALMAN_EXPERT, output_dir=tmpdir))
            record = metrics.to_dict()
            for field in REQUIRED_RECORD_FIELDS:
                assert field in record, f"missing record field: {field}"
            assert record["method"] == "kalman_expert"
            assert record["scenario"] and record["seed"] == 42
            assert record["source"] == "simulation"
            assert Path(record["log_json_path"]).exists()
            assert Path(record["log_csv_path"]).exists()

    def test_methods_genuinely_differ(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            classical = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.CLASSICAL_PID, output_dir=tmpdir))
            kalman = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.KALMAN_EXPERT, output_dir=tmpdir))
            assert classical.provenance["tracker"] == "raw_detection"
            assert kalman.provenance["tracker"] == "kalman"
            assert classical.rmse_px != kalman.rmse_px

    def test_learned_modes_run_brain(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            learned = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.LEARNED_TEMPORAL_EXPERT, output_dir=tmpdir))
            assert learned.ai_action_histogram, "mission brain must act in learned mode"
            assert learned.prediction_rmse_px is not None
            policy = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.LEARNED_TEMPORAL_LEARNED_POLICY, output_dir=tmpdir))
            assert policy.ai_action_histogram

    def test_unavailable_method_raises_no_fabrication(self, tmp_path, monkeypatch):
        from fsoc_tracker.benchmark.methods import MethodUnavailableError
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        monkeypatch.setenv("FSOC_BENCHMARK_MODEL_DIR", str(tmp_path))
        with pytest.raises(MethodUnavailableError):
            BenchmarkRunner().run(_run_cfg(
                BenchmarkMode.LEARNED_TEMPORAL_LEARNED_POLICY, output_dir=str(tmp_path)))

    def test_deterministic_same_seed(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            first = BenchmarkRunner().run(_run_cfg(BenchmarkMode.KALMAN_EXPERT, output_dir=tmpdir))
            second = BenchmarkRunner().run(_run_cfg(BenchmarkMode.KALMAN_EXPERT, output_dir=tmpdir))
            assert first.rmse_px == second.rmse_px
            assert first.repro_command == second.repro_command

    def test_repro_command_roundtrips_tuning_flags(self):
        from fsoc_tracker.benchmark.runner import (
            BenchmarkRunConfig,
            BenchmarkMode,
            build_repro_command,
        )
        base = build_repro_command(BenchmarkRunConfig(
            mode=BenchmarkMode.KALMAN_EXPERT))
        assert "--assoc-identity" not in base
        full = build_repro_command(BenchmarkRunConfig(
            mode=BenchmarkMode.KALMAN_EXPERT, assoc_gate_px=60.0,
            assoc_method="mahalanobis", assoc_appearance=8.0,
            assoc_identity=40.0, identity_code="10110010",
            lead_compensation=True, lead_time_s=0.2))
        for flag in ("--assoc-gate-px 60.0", "--assoc-method mahalanobis",
                     "--assoc-appearance 8.0", "--assoc-identity 40.0",
                     "--identity-code 10110010", "--lead",
                     "--lead-time 0.2"):
            assert flag in full

    def test_full_ai_mission_smoke(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.FULL_AI_MISSION, frames=20, output_dir=tmpdir))
            assert metrics.frames_processed == 20
            assert metrics.provenance["harness"] == "tracking_pipeline_full_stack"
            assert Path(metrics.log_json_path).exists()

    def test_stop_request(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        runner = BenchmarkRunner()

        def stop_after_five(current: int, total: int) -> None:
            if current >= 5:
                runner.request_stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = runner.run(
                _run_cfg(BenchmarkMode.KALMAN_EXPERT, output_dir=tmpdir),
                progress_callback=stop_after_five,
            )
            assert metrics.completed is False
            assert metrics.frames_processed < 25

    def test_export_run_logs(self):
        from fsoc_tracker.benchmark.runner import BenchmarkMode, BenchmarkRunner
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkRunner().run(
                _run_cfg(BenchmarkMode.KALMAN_EXPERT, frames=10, output_dir=tmpdir))
            assert Path(metrics.log_json_path).exists()
            assert Path(metrics.log_csv_path).exists()
            data = json.loads(Path(metrics.log_json_path).read_text())
            assert data["method"] == "kalman_expert"


class TestMultiSeedAggregation:
    def test_aggregate_stats(self):
        from fsoc_tracker.cli.run_benchmark import _aggregate_seed_runs
        per_seed = [
            {"seed": 42, "acquisition_s": 0.1, "rmse_px": 1.0,
             "mae_px": 0.9, "p95_px": 2.0, "max_error_px": 3.0,
             "loss_rate_pct": 1.0, "reacquisition_s": None,
             "lock_retention_pct": 99.0, "fps": 300.0,
             "frames_processed": 200},
            {"seed": 43, "acquisition_s": 0.2, "rmse_px": 3.0,
             "mae_px": 2.0, "p95_px": 4.0, "max_error_px": 5.0,
             "loss_rate_pct": 2.0, "reacquisition_s": 0.5,
             "lock_retention_pct": 98.0, "fps": 320.0,
             "frames_processed": 200},
        ]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agg = _aggregate_seed_runs(
                "kalman_expert", 1, 200, tmp, per_seed)
            assert agg["seeds"] == [42, 43]
            assert agg["rmse_px"]["mean"] == pytest.approx(2.0)
            assert agg["rmse_px"]["n"] == 2
            assert agg["reacquisition_s"]["n"] == 1
            assert agg["frames_processed"] == 400
            import os
            assert os.path.exists(agg["aggregate_json_path"])

    def test_multi_seed_benchmark_runs(self):
        import tempfile

        from fsoc_tracker.cli.run_benchmark import run_sim_benchmark
        with tempfile.TemporaryDirectory() as tmp:
            result = run_sim_benchmark(
                "kalman_expert", "nominal", 42, 30, tmp, seeds=[42, 43])
        assert result["seeds"] == [42, 43]
        assert result["rmse_px"]["mean"] is not None
        assert result["frames_processed"] == 60

    def test_assoc_override_runs(self):
        import tempfile

        from fsoc_tracker.cli.run_benchmark import run_sim_benchmark
        with tempfile.TemporaryDirectory() as tmp:
            result = run_sim_benchmark(
                "kalman_expert", "nominal", 42, 10, tmp,
                assoc_gate_px=40.0, assoc_method="mahalanobis")
        assert result["frames_processed"] == 10


class TestReacquisitionFromReappearance:
    """Reacquisition measured from GT reappearance, not loss start."""

    def _fm(self, idx, ts, locked, tx=None):
        from fsoc_tracker.benchmark.models import FrameMetrics
        return FrameMetrics(frame_index=idx, timestamp_s=ts,
                            lock_status=locked, true_x=tx,
                            true_y=0.0 if tx is not None else None)

    def test_reappearance_metric(self):
        from fsoc_tracker.benchmark.collector import MetricsCollector
        c = MetricsCollector()
        # locked 0.0-0.2, outage 0.2-1.0 (no GT), reappear 1.0, relock 1.2
        frames = [self._fm(0, 0.0, True, 320.0), self._fm(1, 0.1, True, 321.0),
                  self._fm(2, 0.2, True, 322.0), self._fm(3, 0.3, False, None),
                  self._fm(30, 1.0, False, 350.0), self._fm(31, 1.1, False, 351.0),
                  self._fm(32, 1.2, True, 352.0)]
        for fm in frames:
            c.record_frame(fm)
        assert c._reacq_times and abs(c._reacq_times[0] - 0.9) < 1e-9
        assert c._reacq_from_reappearance
        assert abs(c._reacq_from_reappearance[0] - 0.2) < 1e-9


class TestDetectionMetrics:
    """Per-frame TP/FP/FN/TN classification in the collector."""

    def _fm(self, idx, ts, detected, err, tx=320.0):
        from fsoc_tracker.benchmark.models import FrameMetrics
        return FrameMetrics(frame_index=idx, timestamp_s=ts,
                            lock_status=detected, detected=detected,
                            true_x=tx, true_y=240.0, error_px=err)

    def test_precision_recall_perfect(self):
        from fsoc_tracker.benchmark.collector import MetricsCollector
        c = MetricsCollector()
        for i in range(10):
            c.record_frame(self._fm(i, i / 30, True, 2.0))
        r = c.build_result()
        assert r.detection.true_positives == 10
        assert r.detection.precision == 1.0
        assert r.detection.recall == 1.0

    def test_false_negatives_and_ghosts(self):
        from fsoc_tracker.benchmark.collector import MetricsCollector
        from fsoc_tracker.benchmark.models import FrameMetrics
        c = MetricsCollector()
        for i in range(5):
            c.record_frame(self._fm(i, i / 30, False, None))
        for i in range(5, 8):
            c.record_frame(FrameMetrics(
                frame_index=i, timestamp_s=i / 30, lock_status=False,
                detected=True, true_x=None, true_y=None, error_px=None))
        r = c.build_result()
        assert r.detection.false_negatives == 5
        assert r.detection.false_positives == 3
        assert r.detection.true_negatives == 0
        assert r.detection.recall == 0.0

    def test_wrong_target_is_fp(self):
        from fsoc_tracker.benchmark.collector import MetricsCollector
        c = MetricsCollector()
        c.record_frame(self._fm(0, 0.0, True, 50.0))
        r = c.build_result()
        assert r.detection.false_positives == 1
        assert r.detection.true_positives == 0
