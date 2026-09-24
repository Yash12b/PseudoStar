"""Benchmark data models.

All session, scenario, result, threshold, and per-frame metric models.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BenchmarkMode(str, Enum):
    SYNTHETIC = "synthetic"
    EXTERNAL_VIDEO = "external_video"
    LIVE_CAMERA = "live_camera"
    OFFLINE_DATASET = "offline_dataset"


class ThresholdVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class SourceStatus(str, Enum):
    OPENED = "opened"
    READING = "reading"
    EOF = "eof"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class BenchmarkThresholds:
    acquisition_max_s: float = 2.0
    tracking_error_max_px: float = 10.0
    target_loss_max_percent: float = 5.0
    reacquisition_max_s: float = 1.0
    minimum_processing_fps: float = 20.0

    def evaluate(self, result: BenchmarkResult) -> list[ThresholdResult]:
        results = []

        results.append(ThresholdResult(
            name="acquisition_time",
            threshold=f"<={self.acquisition_max_s}s",
            value=result.acquisition.stable_acquisition_s,
            actual=f"{result.acquisition.stable_acquisition_s:.3f}s" if result.acquisition.stable_acquisition_s is not None else "N/A",
            verdict=ThresholdVerdict.PASS
            if result.acquisition.stable_acquisition_s is not None and result.acquisition.stable_acquisition_s <= self.acquisition_max_s
            else ThresholdVerdict.NOT_EVALUABLE if result.acquisition.stable_acquisition_s is None
            else ThresholdVerdict.FAIL,
        ))

        has_gt = result.tracking.rmse_px is not None
        results.append(ThresholdResult(
            name="tracking_error",
            threshold=f"<={self.tracking_error_max_px}px",
            value=result.tracking.rmse_px,
            actual=f"{result.tracking.rmse_px:.2f}px" if result.tracking.rmse_px is not None else "N/A",
            verdict=ThresholdVerdict.PASS
            if has_gt and result.tracking.rmse_px <= self.tracking_error_max_px
            else ThresholdVerdict.NOT_EVALUABLE if not has_gt
            else ThresholdVerdict.FAIL,
        ))

        results.append(ThresholdResult(
            name="target_loss_rate",
            threshold=f"<{self.target_loss_max_percent}%",
            value=result.loss.loss_rate_percent,
            actual=f"{result.loss.loss_rate_percent:.1f}%" if result.loss.loss_rate_percent is not None else "N/A",
            verdict=ThresholdVerdict.PASS
            if result.loss.loss_rate_percent is not None and result.loss.loss_rate_percent < self.target_loss_max_percent
            else ThresholdVerdict.NOT_EVALUABLE if result.loss.loss_rate_percent is None
            else ThresholdVerdict.FAIL,
        ))

        results.append(ThresholdResult(
            name="reacquisition_time",
            threshold=f"<={self.reacquisition_max_s}s",
            value=result.reacquisition.mean_s,
            actual=f"{result.reacquisition.mean_s:.3f}s" if result.reacquisition.mean_s is not None else "N/A",
            verdict=ThresholdVerdict.PASS
            if result.reacquisition.mean_s is not None and result.reacquisition.mean_s <= self.reacquisition_max_s
            else ThresholdVerdict.NOT_EVALUABLE if result.reacquisition.mean_s is None
            else ThresholdVerdict.FAIL,
        ))

        results.append(ThresholdResult(
            name="processing_fps",
            threshold=f">={self.minimum_processing_fps} FPS",
            value=result.performance.processing_fps,
            actual=f"{result.performance.processing_fps:.1f}" if result.performance.processing_fps is not None else "N/A",
            verdict=ThresholdVerdict.PASS
            if result.performance.processing_fps is not None and result.performance.processing_fps >= self.minimum_processing_fps
            else ThresholdVerdict.FAIL,
        ))

        return results


@dataclass
class BenchmarkConfig:
    source: str = ""
    scenario: str = ""
    output_directory: str = "benchmark_output"
    save_frame_log: bool = True
    save_event_log: bool = True
    generate_report: bool = True
    generate_plots: bool = True
    ground_truth_source: str = ""
    thresholds: BenchmarkThresholds = field(default_factory=BenchmarkThresholds)
    realtime: bool = False
    drop_frames: bool = False
    start_frame: int = 0
    end_frame: int | None = None
    start_time_s: float | None = None
    duration_s: float | None = None
    detector_backend: str = "classical"
    tracker_backend: str = "kalman"
    controller_backend: str = "pid"
    disturbance_profile: str = "clear"
    random_seed: int = 42
    max_frames: int | None = None


@dataclass
class FrameMetrics:
    frame_index: int = 0
    timestamp_s: float = 0.0
    dt: float = 0.0
    track_id: int = -1
    source_fps: float | None = None
    processing_time_ms: float = 0.0
    perception_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    control_time_ms: float = 0.0
    disturbance_time_ms: float = 0.0

    detected: bool = False
    detection_confidence: float = 0.0
    detection_x: float = 0.0
    detection_y: float = 0.0
    candidate_count: int = 0

    track_x: float = 0.0
    track_y: float = 0.0
    track_state: str = ""
    lock_status: bool = False
    prediction_only: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    uncertainty_x: float = 0.0
    uncertainty_y: float = 0.0
    residual: float = 0.0

    pan_error_deg: float = 0.0
    tilt_error_deg: float = 0.0
    pan_command_deg_s: float = 0.0
    tilt_command_deg_s: float = 0.0
    pan_saturated: bool = False
    tilt_saturated: bool = False

    true_x: float | None = None
    true_y: float | None = None
    error_x: float | None = None
    error_y: float | None = None
    error_px: float | None = None
    squared_error: float | None = None

    disturbance_state: str = ""


@dataclass
class ThresholdResult:
    name: str = ""
    threshold: str = ""
    value: float | None = None
    actual: str = ""
    verdict: ThresholdVerdict = ThresholdVerdict.NOT_EVALUABLE


@dataclass
class SessionMetadata:
    session_id: str = ""
    source_type: str = ""
    source_name: str = ""
    source_fps: float | None = None
    processed_fps: float | None = None
    frame_count: int = 0
    duration_s: float = 0.0
    detector_backend: str = ""
    tracker_backend: str = ""
    controller_backend: str = ""
    disturbance_profile: str = ""
    model_version: str = ""
    random_seed: int = 42
    configuration_hash: str = ""
    status: str = "pending"
    start_time: str = ""
    end_time: str = ""
    notes: str = ""


@dataclass
class BenchmarkSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str = ""
    mode: BenchmarkMode = BenchmarkMode.SYNTHETIC
    source_name: str = ""
    source_fps: float | None = None
    processed_fps: float | None = None
    frame_count: int = 0
    duration_s: float = 0.0
    detector_name: str = "classical"
    tracker_name: str = "kalman"
    controller_name: str = "pid"
    disturbance_profile: str = "clear"
    model_version: str = ""
    random_seed: int = 42
    configuration_hash: str = ""
    status: str = "pending"
    notes: str = ""

    def to_metadata(self) -> SessionMetadata:
        return SessionMetadata(
            session_id=self.session_id,
            source_type=self.mode.value,
            source_name=self.source_name,
            source_fps=self.source_fps,
            processed_fps=self.processed_fps,
            frame_count=self.frame_count,
            duration_s=self.duration_s,
            detector_backend=self.detector_name,
            tracker_backend=self.tracker_name,
            controller_backend=self.controller_name,
            disturbance_profile=self.disturbance_profile,
            model_version=self.model_version,
            random_seed=self.random_seed,
            configuration_hash=self.configuration_hash,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            notes=self.notes,
        )


@dataclass
class BenchmarkScenario:
    name: str = ""
    description: str = ""
    mode: BenchmarkMode = BenchmarkMode.SYNTHETIC
    source_path: str = ""
    target_size_px: float = 10.0
    trajectory_type: str = "straight_line"
    atmosphere: str = "clear"
    noise: str = "none"
    jitter: str = "none"
    platform_motion: str = "none"
    expected_fps: float = 30.0
    run_duration_s: float = 10.0
    random_seed: int = 42
    ground_truth_source: str = ""
    detector_backend: str = "classical"
    tracker_config: dict[str, Any] = field(default_factory=dict)
    disturbance_config: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionMetrics:
    first_detection_s: float | None = None
    stable_acquisition_s: float | None = None
    acquisition_duration_s: float | None = None
    total_acquisitions: int = 0


@dataclass
class TrackingMetricsSummary:
    mean_error_px: float | None = None
    rmse_px: float | None = None
    rmse_x_px: float | None = None
    rmse_y_px: float | None = None
    mae_x_px: float | None = None
    mae_y_px: float | None = None
    max_error_px: float | None = None
    p50_error_px: float | None = None
    p90_error_px: float | None = None
    p95_error_px: float | None = None
    p99_error_px: float | None = None
    within_threshold_percent: float | None = None
    frames_with_ground_truth: int = 0
    frames_evaluated: int = 0


@dataclass
class DetectionMetrics:
    """Per-frame detection classification against ground truth.

    TP: detected, target visible, error within threshold.
    FP: detected while target absent, or error above threshold.
    FN: target visible but nothing detected.
    TN: target absent and nothing detected.
    Frames without ground truth are skipped (unjudgeable).
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float | None = None
    recall: float | None = None
    evaluated_frames: int = 0


@dataclass
class MOTMetrics:
    """Multi-object-tracking style scores for the single target.

    MOTA = 1 - (FN + FP + IDSW) / GT_frames (py-motmetrics definition;
    IDSW counts track-ID changes while the target stays visible).
    MOTP = mean matched distance over TP frames.
    IDF1 = 2TP / (2TP + FP + FN) (identity F1 for one target).
    """

    mota: float | None = None
    motp_px: float | None = None
    idf1: float | None = None
    id_switches: int = 0
    gt_frames: int = 0


@dataclass
class LossMetrics:
    loss_event_count: int = 0
    loss_rate_percent: float | None = None
    lock_retention_percent: float | None = None
    locked_frames: int = 0
    evaluable_frames: int = 0
    total_locked_time_s: float = 0.0
    total_evaluable_time_s: float = 0.0


@dataclass
class ReacquisitionMetrics:
    times: list[float] = field(default_factory=list)
    mean_s: float | None = None
    median_s: float | None = None
    max_s: float | None = None
    p95_s: float | None = None
    total_events: int = 0
    failed_reacquisitions: int = 0
    # Honest PS reading: time from target REAPPEARANCE (GT visible again)
    # to re-lock, excluding the outage itself (covered by `times`).
    times_from_reappearance: list[float] = field(default_factory=list)
    mean_from_reappearance_s: float | None = None


@dataclass
class LatencyMetrics:
    mean_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None
    total_ms: float = 0.0
    count: int = 0


@dataclass
class PerformanceMetrics:
    mean_processing_ms: float | None = None
    p95_processing_ms: float | None = None
    max_processing_ms: float | None = None
    processing_fps: float | None = None
    throughput_fps: float | None = None
    frames_processed: int = 0
    wall_time_s: float = 0.0


@dataclass
class PerceptionLatency:
    mean_ms: float | None = None
    p95_ms: float | None = None


@dataclass
class TrackingLatency:
    mean_ms: float | None = None
    p95_ms: float | None = None


@dataclass
class ControlLatency:
    mean_ms: float | None = None
    p95_ms: float | None = None


@dataclass
class PipelineLatency:
    perception: PerceptionLatency = field(default_factory=PerceptionLatency)
    tracking: TrackingLatency = field(default_factory=TrackingLatency)
    control: ControlLatency = field(default_factory=ControlLatency)
    total: LatencyMetrics = field(default_factory=LatencyMetrics)


@dataclass
class PerformanceBudget:
    max_preprocessing_ms: float = 5.0
    max_perception_ms: float = 50.0
    max_tracking_ms: float = 5.0
    max_control_ms: float = 5.0
    max_total_ms: float = 50.0
    violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    session: BenchmarkSession = field(default_factory=BenchmarkSession)
    duration: float = 0.0
    source_fps: float | None = None
    processed_fps: float | None = None
    frame_count: int = 0
    acquisition: AcquisitionMetrics = field(default_factory=AcquisitionMetrics)
    tracking: TrackingMetricsSummary = field(default_factory=TrackingMetricsSummary)
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    mot: MOTMetrics = field(default_factory=MOTMetrics)
    loss: LossMetrics = field(default_factory=LossMetrics)
    reacquisition: ReacquisitionMetrics = field(default_factory=ReacquisitionMetrics)
    latency: PipelineLatency = field(default_factory=PipelineLatency)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    budget: PerformanceBudget = field(default_factory=PerformanceBudget)
    threshold_results: list[ThresholdResult] = field(default_factory=list)
    detector_name: str = ""
    model_name: str = ""
    configuration_hash: str = ""
    notes: str = ""
    failure_categories: dict[str, int] = field(default_factory=dict)

    def summary_table(self) -> str:
        lines = [
            "+--------------------------------------+------------------+----------+",
            "| Metric                               | Result           | Verdict  |",
            "+--------------------------------------+------------------+----------+",
        ]

        def row(metric: str, value: str, verdict: str = "") -> str:
            return f"| {metric:<36s} | {value:<16s} | {verdict:<8s} |"

        acq = self.acquisition
        trk = self.tracking
        det = self.detection
        los = self.loss
        rea = self.reacquisition
        perf = self.performance

        lines.append(row("Acquisition time",
                         f"{acq.stable_acquisition_s:.3f}s" if acq.stable_acquisition_s is not None else "N/A"))
        lines.append(row("Tracking RMSE",
                         f"{trk.rmse_px:.2f}px" if trk.rmse_px is not None else "N/A"))
        lines.append(row("P95 error",
                         f"{trk.p95_error_px:.2f}px" if trk.p95_error_px is not None else "N/A"))
        lines.append(row("Max error",
                         f"{trk.max_error_px:.2f}px" if trk.max_error_px is not None else "N/A"))
        lines.append(row("Within threshold",
                         f"{trk.within_threshold_percent:.1f}%" if trk.within_threshold_percent is not None else "N/A"))
        lines.append(row("Target loss",
                         f"{los.loss_rate_percent:.1f}%" if los.loss_rate_percent is not None else "N/A"))
        lines.append(row("Precision",
                         f"{det.precision:.3f}" if det.precision is not None else "N/A"))
        lines.append(row("Recall",
                         f"{det.recall:.3f}" if det.recall is not None else "N/A"))
        mot = self.mot
        lines.append(row("MOTA",
                         f"{mot.mota:.3f}" if mot.mota is not None else "N/A"))
        lines.append(row("MOTP",
                         f"{mot.motp_px:.2f}px" if mot.motp_px is not None else "N/A"))
        lines.append(row("IDF1",
                         f"{mot.idf1:.3f}" if mot.idf1 is not None else "N/A"))
        lines.append(row("Lock retention",
                         f"{los.lock_retention_percent:.1f}%" if los.lock_retention_percent is not None else "N/A"))
        lines.append(row("Re-acquisition",
                         f"{rea.mean_s:.3f}s" if rea.mean_s is not None else "N/A"))
        lines.append(row("Processing FPS",
                         f"{perf.processing_fps:.1f}" if perf.processing_fps is not None else "N/A"))

        lines.append("+--------------------------------------+------------------+----------+")

        for t in self.threshold_results:
            verdict_str = t.verdict.value
            lines.append(row(f"  {t.name}", t.actual, verdict_str))

        lines.append("+--------------------------------------+------------------+----------+")
        return "\n".join(lines)
