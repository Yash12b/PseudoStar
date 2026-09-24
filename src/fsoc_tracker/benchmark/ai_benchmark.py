"""Closed-loop AI benchmark.

Runs all SIH scenarios with AI in the loop and compares performance
against classical-only baseline.

Metrics: acquisition time, tracking error, loss events, reacquisition time,
processing FPS, mean error, P95 error.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.pipeline.sources import SimulationSource


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario run."""
    scenario_name: str
    trajectory_type: str
    disturbance: str
    frame_count: int = 0
    duration_s: float = 0.0
    acquisition_time_s: float | None = None
    loss_events: int = 0
    reacquisition_times: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    processing_times_ms: list[float] = field(default_factory=list)
    ai_times_ms: list[float] = field(default_factory=list)
    ai_decisions: dict[str, int] = field(default_factory=dict)

    @property
    def mean_error_px(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0

    @property
    def p95_error_px(self) -> float:
        return float(np.percentile(self.errors, 95)) if self.errors else 0.0

    @property
    def processing_fps(self) -> float:
        mean_ms = np.mean(self.processing_times_ms) if self.processing_times_ms else 100.0
        return 1000.0 / max(mean_ms, 0.001)

    @property
    def mean_reacquisition_s(self) -> float:
        return float(np.mean(self.reacquisition_times)) if self.reacquisition_times else 0.0

    def summary(self) -> str:
        acq = f"{self.acquisition_time_s:.2f}s" if self.acquisition_time_s is not None else "N/A"
        reacq = f"{self.mean_reacquisition_s:.2f}s" if self.reacquisition_times else "N/A"
        return (
            f"{self.scenario_name}: "
            f"frames={self.frame_count} "
            f"acq={acq} "
            f"loss={self.loss_events} "
            f"reacq={reacq} "
            f"err={self.mean_error_px:.1f}px "
            f"p95={self.p95_error_px:.1f}px "
            f"fps={self.processing_fps:.1f}"
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark result comparing AI vs classical."""
    classical: list[ScenarioMetrics] = field(default_factory=list)
    ai_enhanced: list[ScenarioMetrics] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        lines = ["=== AI Benchmark Results ===", ""]
        lines.append("Classical baseline:")
        for m in self.classical:
            lines.append(f"  {m.summary()}")
        lines.append("")
        lines.append("AI-enhanced:")
        for m in self.ai_enhanced:
            lines.append(f"  {m.summary()}")
        lines.append("")

        if self.classical and self.ai_enhanced:
            c_mean_err = np.mean([m.mean_error_px for m in self.classical])
            a_mean_err = np.mean([m.mean_error_px for m in self.ai_enhanced])
            c_loss = sum(m.loss_events for m in self.classical)
            a_loss = sum(m.loss_events for m in self.ai_enhanced)
            c_fps = np.mean([m.processing_fps for m in self.classical])
            a_fps = np.mean([m.processing_fps for m in self.ai_enhanced])
            lines.append("Comparison:")
            lines.append(f"  Mean error: {c_mean_err:.1f}px -> {a_mean_err:.1f}px")
            lines.append(f"  Total losses: {c_loss} -> {a_loss}")
            lines.append(f"  Mean FPS: {c_fps:.1f} -> {a_fps:.1f}")

        return "\n".join(lines)


def _run_scenario(
    pipeline: TrackingPipeline,
    scenario_name: str,
    trajectory_type: str,
    disturbance: str,
    seed: int,
    duration_s: float = 10.0,
    dt: float = 1.0 / 30.0,
    target_size: float = 10.0,
) -> ScenarioMetrics:
    """Run a single scenario and collect metrics."""
    from fsoc_tracker.disturbances.config import get_preset_config
    from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
    from fsoc_tracker.simulation.camera.camera import VirtualCamera
    from fsoc_tracker.simulation.camera.state import CameraState
    from fsoc_tracker.simulation.engine import SimulationEngine
    from fsoc_tracker.simulation.sensor.config import SensorConfig
    from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
    from fsoc_tracker.simulation.world import WorldConfig

    traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2}
    if trajectory_type == "circular":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius": 0.5, "angular_speed_rad_s": 0.3}
    elif trajectory_type == "figure_8":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3}
    elif trajectory_type == "sinusoidal":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3, "freq_x": 0.3, "freq_y": 0.2}

    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=seed)
    engine = SimulationEngine(wc)
    engine.add_target(trajectory_type=trajectory_type, trajectory_params=traj_params)

    cam_state = CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=640, height=480,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
    )
    camera = VirtualCamera(cam_state)
    sc = SensorConfig(width=640, height=480, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    dist_cfg = get_preset_config(disturbance)
    disturbance_pipe = DisturbancePipeline(dist_cfg) if disturbance != "clear" else None

    source = SimulationSource(engine, camera, sensor, disturbance_pipe, dt)

    pipeline.reset()
    pipeline.set_source(source)
    pipeline.start()

    metrics = ScenarioMetrics(
        scenario_name=scenario_name,
        trajectory_type=trajectory_type,
        disturbance=disturbance,
    )
    max_frames = int(duration_s / dt)

    try:
        while metrics.frame_count < max_frames:
            result = pipeline.step()
            if result is None:
                break
            metrics.frame_count += 1
            metrics.processing_times_ms.append(result.total_ms)
            if result.ai_ms > 0:
                metrics.ai_times_ms.append(result.ai_ms)
            if result.ai_decision is not None:
                action = result.ai_decision.action.value
                metrics.ai_decisions[action] = metrics.ai_decisions.get(action, 0) + 1
            if result.error_px is not None:
                metrics.errors.append(result.error_px)
            if result.tracking is not None:
                if metrics.acquisition_time_s is None and result.tracking.state.name in ("TRACKING", "REACQUIRING"):
                    metrics.acquisition_time_s = result.timestamp_s
                if result.tracking.state.name == "SEARCHING" and metrics.frame_count > 10:
                    metrics.loss_events += 1
    except Exception:
        pass

    pipeline.stop()
    metrics.duration_s = metrics.frame_count * dt
    return metrics


def run_benchmark(
    duration_s: float = 5.0,
    seed: int = 42,
    output_dir: str = "runs/ai-benchmark",
) -> BenchmarkResult:
    """Run full benchmark comparing classical vs AI-enhanced pipeline."""
    from fsoc_tracker.ai.adaptive_roi import AdaptiveROI
    from fsoc_tracker.ai.failure_predictor import FailurePredictor
    from fsoc_tracker.ai.mission import AIMissionBrain
    from fsoc_tracker.control.adaptive import AdaptiveController
    from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
    from fsoc_tracker.tracking.search import SearchController

    scenarios = [
        ("SL_clear", "straight_line", "clear"),
        ("SL_light", "straight_line", "light"),
        ("CIRC_clear", "circular", "clear"),
        ("CIRC_light", "circular", "light"),
        ("F8_clear", "figure_8", "clear"),
        ("F8_light", "figure_8", "light"),
        ("SIN_clear", "sinusoidal", "clear"),
        ("SIN_light", "sinusoidal", "light"),
    ]

    classical_results = []
    ai_results = []

    print("[BENCHMARK] Running classical baseline...")
    for name, traj, dist in scenarios:
        pipeline = TrackingPipeline()
        metrics = _run_scenario(pipeline, name, traj, dist, seed, duration_s)
        classical_results.append(metrics)
        print(f"  {metrics.summary()}")

    print("[BENCHMARK] Running AI-enhanced pipeline...")
    for name, traj, dist in scenarios:
        pipeline = TrackingPipeline(
            ai_brain=AIMissionBrain(),
            adaptive_roi=AdaptiveROI(),
            failure_predictor=FailurePredictor(),
            adaptive_kalman=AdaptiveKalmanManager(),
            adaptive_controller=AdaptiveController(),
            search_controller=SearchController(),
        )
        metrics = _run_scenario(pipeline, f"AI_{name}", traj, dist, seed, duration_s)
        ai_results.append(metrics)
        print(f"  {metrics.summary()}")

    result = BenchmarkResult(
        classical=classical_results,
        ai_enhanced=ai_results,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_summary.txt").write_text(result.summary(), encoding="utf-8")

    report = {
        "timestamp": result.timestamp,
        "classical": [
            {
                "name": m.scenario_name,
                "mean_error_px": m.mean_error_px,
                "p95_error_px": m.p95_error_px,
                "loss_events": m.loss_events,
                "processing_fps": m.processing_fps,
                "acquisition_time_s": m.acquisition_time_s,
                "frame_count": m.frame_count,
            }
            for m in classical_results
        ],
        "ai_enhanced": [
            {
                "name": m.scenario_name,
                "mean_error_px": m.mean_error_px,
                "p95_error_px": m.p95_error_px,
                "loss_events": m.loss_events,
                "processing_fps": m.processing_fps,
                "acquisition_time_s": m.acquisition_time_s,
                "frame_count": m.frame_count,
                "ai_decisions": m.ai_decisions,
            }
            for m in ai_results
        ],
    }
    (out / "benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\n{result.summary()}")
    print(f"\nResults saved to {out}")
    return result
