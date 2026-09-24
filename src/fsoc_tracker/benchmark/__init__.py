"""Benchmark engine for standardized evaluation.

Supports synthetic closed-loop, external video, live camera, and
offline dataset benchmark modes.
"""

from fsoc_tracker.benchmark.collector import FrameMetrics, MetricsCollector
from fsoc_tracker.benchmark.engine import BenchmarkEngine
from fsoc_tracker.benchmark.export import export_csv, export_json, export_run_csv, export_run_json
from fsoc_tracker.benchmark.ground_truth import (
    GroundTruthProvider,
    SidecarGroundTruthProvider,
    SyntheticGroundTruthProvider,
)
from fsoc_tracker.benchmark.live_source import LiveCameraSource
from fsoc_tracker.benchmark.methods import (
    METHOD_LABELS,
    METHOD_ORDER,
    MethodAvailability,
    MethodUnavailableError,
    check_method_availability,
    load_method_components,
)
from fsoc_tracker.benchmark.models import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkScenario,
    BenchmarkSession,
    BenchmarkThresholds,
    ThresholdResult,
    ThresholdVerdict,
)
from fsoc_tracker.benchmark.plots import generate_plots
from fsoc_tracker.benchmark.profiler import PerformanceProfiler
from fsoc_tracker.benchmark.report import generate_html_report
from fsoc_tracker.benchmark.runner import (
    BenchmarkMetrics,
    BenchmarkMode,
    BenchmarkRunConfig,
    BenchmarkRunner,
    build_repro_command,
)
from fsoc_tracker.benchmark.video_source import VideoBenchmarkSource

__all__ = [
    "BenchmarkConfig", "BenchmarkResult", "BenchmarkScenario",
    "BenchmarkSession", "BenchmarkThresholds", "FrameMetrics",
    "ThresholdResult", "ThresholdVerdict",
    "MetricsCollector", "PerformanceProfiler",
    "GroundTruthProvider", "SyntheticGroundTruthProvider",
    "SidecarGroundTruthProvider",
    "VideoBenchmarkSource", "LiveCameraSource",
    "BenchmarkEngine", "export_csv", "export_json",
    "export_run_csv", "export_run_json",
    "generate_html_report", "generate_plots",
    "BenchmarkMetrics", "BenchmarkMode", "BenchmarkRunConfig",
    "BenchmarkRunner", "build_repro_command",
    "METHOD_LABELS", "METHOD_ORDER",
    "MethodAvailability", "MethodUnavailableError",
    "check_method_availability", "load_method_components",
]
