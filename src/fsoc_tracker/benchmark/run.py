"""Experiment runner and sweep utilities.

Supports single scenario runs, multi-scenario sweeps, and
deterministic CI regression benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fsoc_tracker.benchmark.engine import BenchmarkEngine
from fsoc_tracker.benchmark.export import export_csv, export_event_log, export_json
from fsoc_tracker.benchmark.ground_truth import (
    GroundTruthProvider,
    NullGroundTruthProvider,
)
from fsoc_tracker.benchmark.models import (
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkScenario,
    BenchmarkSession,
)
from fsoc_tracker.benchmark.plots import generate_plots
from fsoc_tracker.benchmark.report import generate_html_report
from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.perception.base import PerceptionEngine
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.tracking.tracker import KalmanTracker


def run_scenario(
    scenario: BenchmarkScenario,
    source: FrameSource,
    detector: PerceptionEngine | None = None,
    tracker: KalmanTracker | None = None,
    ground_truth: GroundTruthProvider | None = None,
    output_dir: str | None = None,
    generate_report: bool = True,
    generate_plot_files: bool = True,
    save_frame_log: bool = True,
) -> BenchmarkResult:
    det = detector or ClassicalBeaconDetector()
    trk = tracker or KalmanTracker()
    gt = ground_truth or NullGroundTruthProvider()

    engine = BenchmarkEngine(detector=det, tracker=trk, ground_truth=gt)

    session = BenchmarkSession(
        mode=BenchmarkMode.EXTERNAL_VIDEO if scenario.mode == BenchmarkMode.EXTERNAL_VIDEO else BenchmarkMode.SYNTHETIC,
        source_name=scenario.source_path or scenario.name,
        source_fps=scenario.expected_fps,
        detector_name=det.name,
        random_seed=scenario.random_seed,
        disturbance_profile=scenario.atmosphere,
    )

    if not source.is_open():
        source.open()
    result = engine.run_source(source, session=session)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        export_json(result, str(out / "benchmark_result.json"))
        if save_frame_log:
            export_csv(engine.get_collector().frames, str(out / "frame_metrics.csv"))
            export_event_log(engine.get_collector().events, str(out / "event_log.csv"))
        if generate_report:
            generate_html_report(result, str(out / "report.html"))
        if generate_plot_files:
            generate_plots(result, engine.get_collector().frames, str(out / "plots"))

    return result


def sweep(
    scenarios: list[BenchmarkScenario],
    sources: dict[str, FrameSource],
    detectors: dict[str, PerceptionEngine] | None = None,
    tracker: KalmanTracker | None = None,
    output_base: str = "benchmark_results",
) -> dict[str, BenchmarkResult]:
    dets = detectors or {"classical": ClassicalBeaconDetector()}
    trk = tracker or KalmanTracker()
    results: dict[str, BenchmarkResult] = {}

    for scenario in scenarios:
        for det_name, det in dets.items():
            run_name = f"{scenario.name}_{det_name}"
            source = sources.get(scenario.source_path)
            if source is None:
                continue

            out_dir = str(Path(output_base) / run_name)
            result = run_scenario(
                scenario=scenario,
                source=source,
                detector=det,
                tracker=trk,
                output_dir=out_dir,
            )
            results[run_name] = result

    return results


def ci_regression_benchmark() -> BenchmarkResult:
    """Quick deterministic benchmark for CI. No large models required."""
    from fsoc_tracker.core.models import ColorModel, Frame, SourceType

    det = ClassicalBeaconDetector()
    trk = KalmanTracker()

    rng = np.random.default_rng(42)
    frames: list[Frame] = []
    for i in range(150):
        ts = i / 30.0
        img = np.full((120, 160), 5.0, dtype=np.float64)
        cx = 80 + int(20 * np.sin(2 * np.pi * i / 150))
        cy = 60 + int(10 * np.cos(2 * np.pi * i / 100))
        for y in range(max(0, cy - 8), min(120, cy + 8)):
            for x in range(max(0, cx - 8), min(160, cx + 8)):
                d = (x - cx) ** 2 + (y - cy) ** 2
                img[y, x] += 200 * np.exp(-d / 18.0)
        img += rng.normal(0, 2, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(Frame(
            image=img, width=160, height=120, channels=1,
            color_model=ColorModel.GRAY, source_id="ci_regression",
            source_type=SourceType.SYNTHETIC,
            frame_index=i, timestamp_s=ts, nominal_fps=30.0,
        ))

    engine = BenchmarkEngine(detector=det, tracker=trk)
    session = BenchmarkSession(
        mode=BenchmarkMode.SYNTHETIC,
        source_name="ci_regression",
        source_fps=30.0,
        detector_name="classical",
        random_seed=42,
    )
    return engine.run_frames(frames, session=session)
