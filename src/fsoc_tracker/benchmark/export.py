"""Benchmark result export — CSV, JSON, event log.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fsoc_tracker.benchmark.collector import FrameMetrics
from fsoc_tracker.benchmark.models import BenchmarkResult

if TYPE_CHECKING:
    from fsoc_tracker.benchmark.runner import BenchmarkMetrics


def export_json(result: BenchmarkResult, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "session": {
            "session_id": result.session.session_id,
            "start_time": result.session.start_time,
            "end_time": result.session.end_time,
            "mode": result.session.mode.value,
            "source_name": result.session.source_name,
            "source_fps": result.session.source_fps,
            "processed_fps": result.session.processed_fps,
            "frame_count": result.session.frame_count,
            "duration_s": result.session.duration_s,
            "detector": result.session.detector_name,
            "tracker": result.session.tracker_name,
            "controller": result.session.controller_name,
            "disturbance": result.session.disturbance_profile,
            "seed": result.session.random_seed,
            "status": result.session.status,
        },
        "metrics": {
            "duration_s": result.duration,
            "source_fps": result.source_fps,
            "processed_fps": result.processed_fps,
            "frame_count": result.frame_count,
            "acquisition": {
                "first_detection_s": result.acquisition.first_detection_s,
                "stable_acquisition_s": result.acquisition.stable_acquisition_s,
                "total_acquisitions": result.acquisition.total_acquisitions,
            },
            "tracking": {
                "mean_error_px": result.tracking.mean_error_px,
                "rmse_px": result.tracking.rmse_px,
                "rmse_x_px": result.tracking.rmse_x_px,
                "rmse_y_px": result.tracking.rmse_y_px,
                "mae_x_px": result.tracking.mae_x_px,
                "mae_y_px": result.tracking.mae_y_px,
                "max_error_px": result.tracking.max_error_px,
                "p50_error_px": result.tracking.p50_error_px,
                "p90_error_px": result.tracking.p90_error_px,
                "p95_error_px": result.tracking.p95_error_px,
                "p99_error_px": result.tracking.p99_error_px,
                "within_threshold_percent": result.tracking.within_threshold_percent,
                "frames_with_ground_truth": result.tracking.frames_with_ground_truth,
            },
            "loss": {
                "loss_event_count": result.loss.loss_event_count,
                "loss_rate_percent": result.loss.loss_rate_percent,
                "lock_retention_percent": result.loss.lock_retention_percent,
                "locked_frames": result.loss.locked_frames,
                "evaluable_frames": result.loss.evaluable_frames,
            },
            "detection": {
                "true_positives": result.detection.true_positives,
                "false_positives": result.detection.false_positives,
                "false_negatives": result.detection.false_negatives,
                "true_negatives": result.detection.true_negatives,
                "precision": result.detection.precision,
                "recall": result.detection.recall,
                "evaluated_frames": result.detection.evaluated_frames,
            },
            "mot": {
                "mota": result.mot.mota,
                "motp_px": result.mot.motp_px,
                "idf1": result.mot.idf1,
                "id_switches": result.mot.id_switches,
                "gt_frames": result.mot.gt_frames,
            },
            "reacquisition": {
                "mean_s": result.reacquisition.mean_s,
                "median_s": result.reacquisition.median_s,
                "max_s": result.reacquisition.max_s,
                "p95_s": result.reacquisition.p95_s,
                "total_events": result.reacquisition.total_events,
                "failed_reacquisitions": result.reacquisition.failed_reacquisitions,
                "times_from_reappearance": result.reacquisition.times_from_reappearance,
                "mean_from_reappearance_s": result.reacquisition.mean_from_reappearance_s,
            },
            "performance": {
                "mean_processing_ms": result.performance.mean_processing_ms,
                "p95_processing_ms": result.performance.p95_processing_ms,
                "max_processing_ms": result.performance.max_processing_ms,
                "processing_fps": result.performance.processing_fps,
                "throughput_fps": result.performance.throughput_fps,
                "frames_processed": result.performance.frames_processed,
                "wall_time_s": result.performance.wall_time_s,
            },
            "latency": {
                "perception_mean_ms": result.latency.perception.mean_ms,
                "perception_p95_ms": result.latency.perception.p95_ms,
                "tracking_mean_ms": result.latency.tracking.mean_ms,
                "tracking_p95_ms": result.latency.tracking.p95_ms,
                "control_mean_ms": result.latency.control.mean_ms,
                "control_p95_ms": result.latency.control.p95_ms,
                "total_mean_ms": result.latency.total.mean_ms,
                "total_p95_ms": result.latency.total.p95_ms,
            },
        },
        "thresholds": [
            {"name": t.name, "threshold": t.threshold, "actual": t.actual, "verdict": t.verdict.value}
            for t in result.threshold_results
        ],
        "detector": result.detector_name,
    }

    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    return str(p)


def export_csv(frames: list[FrameMetrics], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        return str(p)

    headers = [
        "timestamp_s", "frame_index", "dt", "source_fps", "processing_time_ms",
        "perception_time_ms", "tracking_time_ms",
        "detected", "detection_confidence", "detection_x", "detection_y", "candidate_count",
        "track_x", "track_y", "track_state", "lock_status", "prediction_only",
        "velocity_x", "velocity_y", "uncertainty_x", "uncertainty_y", "residual",
        "true_x", "true_y", "error_x", "error_y", "error_px", "squared_error",
        "pan_error_deg", "tilt_error_deg", "pan_command_deg_s", "tilt_command_deg_s",
        "disturbance_state",
    ]

    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for fm in frames:
            row = {h: getattr(fm, h, "") for h in headers}
            writer.writerow(row)

    return str(p)


def export_event_log(events: list[dict[str, Any]], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not events:
        return str(p)

    all_keys: set[str] = set()
    for e in events:
        all_keys.update(e.keys())
    headers = sorted(all_keys)

    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow(e)

    return str(p)


def export_run_json(
    metrics: BenchmarkMetrics | dict[str, Any],
    frame_rows: list[dict[str, Any]],
    output_dir: str,
) -> tuple[str, str]:
    """Write one benchmark run: summary record JSON + per-frame CSV.

    Returns (json_path, csv_path).
    """
    from fsoc_tracker.benchmark.runner import FRAME_LOG_COLUMNS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base = (
        f"run_{stamp}_{record.get('method', 'unknown')}"
        f"_world{record.get('scenario', '?')}_seed{record.get('seed', '?')}"
    )
    base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)
    json_path = out / f"{base}.json"
    csv_path = out / f"{base}_frames.csv"
    record["log_json_path"] = str(json_path)
    record["log_csv_path"] = str(csv_path)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    export_run_csv(frame_rows, str(csv_path), columns=list(FRAME_LOG_COLUMNS))
    return str(json_path), str(csv_path)


def export_run_csv(
    frame_rows: list[dict[str, Any]],
    path: str,
    columns: list[str] | None = None,
) -> str:
    """Write per-frame benchmark rows to CSV (automatic performance log)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = columns or sorted({k for row in frame_rows for k in row})
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in frame_rows:
            writer.writerow({h: row.get(h, "") for h in headers})
    return str(p)
