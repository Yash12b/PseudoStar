"""Headless simulation runner.

Usage:
    python -m fsoc_tracker.cli.run_simulation [--trajectory straight_line] [--duration 10] [--dt 0.033]
"""

from __future__ import annotations

import argparse
import sys
import time

from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.disturbances.config import get_preset_config
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.pipeline.eval import EvalSink
from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.pipeline.session import SessionController
from fsoc_tracker.pipeline.sources import SimulationSource
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.simulation.world_builder import make_experiment_world
from fsoc_tracker.tracking.tracker import KalmanTracker


def run_simulation(
    trajectory: str = "straight_line",
    duration_s: float = 10.0,
    dt: float = 1.0 / 30.0,
    seed: int = 42,
    target_size: float = 10.0,
    disturbance: str = "clear",
    output_dir: str = "runs",
    verbose: bool = False,
) -> dict:
    """Run a headless simulation and return results."""
    print(f"[SIM] trajectory={trajectory} duration={duration_s}s dt={dt:.4f}s seed={seed}")
    print(f"[SIM] disturbance={disturbance} target_size={target_size}px")

    scenario = make_experiment_world(
        motion=trajectory, seed=seed, target_size_px=target_size,
        disturbance=disturbance,
    )
    wc = WorldConfig(
        width=scenario.world_width, height=scenario.world_height,
        random_seed=seed,
    )
    engine = SimulationEngine(wc)
    engine.load_scenario(scenario)

    cam_state = CameraState(
        horizontal_fov_deg=scenario.terminal.hfov_deg,
        vertical_fov_deg=scenario.terminal.vfov_deg,
        width=640, height=480,
        max_pan_speed_deg_s=scenario.terminal.max_pan_speed_deg_s,
        max_tilt_speed_deg_s=scenario.terminal.max_tilt_speed_deg_s,
        position_x=scenario.terminal.x,
        position_y=scenario.terminal.y,
        position_z=scenario.terminal.z,
        pan_deg=scenario.terminal.yaw_deg,
        tilt_deg=scenario.terminal.pitch_deg,
    )
    camera = VirtualCamera(cam_state)

    sc = SensorConfig(width=640, height=480, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    dist_cfg = get_preset_config(disturbance)
    disturbance_pipe = DisturbancePipeline(dist_cfg) if disturbance != "clear" else None

    # Build pipeline
    perception_cfg = PerceptionConfig(
        threshold_mode="global",
        threshold_value=20.0,
    )
    source = SimulationSource(
        engine, camera, sensor, disturbance_pipe, dt, eval_sink=EvalSink(),
    )
    pipeline = TrackingPipeline(
        perception=ClassicalBeaconDetector(perception_cfg),
        tracker=KalmanTracker(),
        controller=CoarsePointingController(),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)

    session = SessionController(pipeline, output_dir=output_dir)
    session.configure(
        source_type="simulation",
        source_info=f"trajectory={trajectory}",
        disturbance_profile=disturbance,
        random_seed=seed,
    )

    # Run
    session.start()
    frame_count = 0
    max_frames = int(duration_s / dt)
    t_wall_start = time.monotonic()

    while session.state.value == "RUNNING" and frame_count < max_frames:
        result = session.step()
        if result is None:
            break
        frame_count += 1
        if verbose and frame_count % 30 == 0:
            err_str = f"err={result.error_px:.1f}px" if result.error_px is not None else "err=N/A"
            print(
                f"  frame={frame_count:5d} t={result.timestamp_s:7.2f}s "
                f"track={result.tracking.state.name if result.tracking else '?'} "
                f"{err_str}"
            )

    final = session.stop()
    wall_time = time.monotonic() - t_wall_start

    print("\n" + "=" * 60)
    print(final.summary())
    print(f"Wall-clock time: {wall_time:.2f}s")
    print(f"Artifacts: {final.artifacts_dir}")
    print("=" * 60)

    return {
        "session_id": final.metadata.session_id,
        "frame_count": final.frame_count,
        "duration_s": final.duration_s,
        "processing_fps": final.processing_fps,
        "mean_processing_ms": final.mean_processing_ms,
        "acquisition_time_s": final.acquisition_time_s,
        "loss_events": final.loss_events,
        "errors": final.errors,
        "artifacts_dir": final.artifacts_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FSOC Headless Simulation Runner")
    parser.add_argument("-t", "--trajectory", default="straight_line",
                        choices=["straight_line", "circular", "figure_8", "random", "spiral", "sinusoidal"])
    parser.add_argument("-d", "--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=float, default=10.0)
    parser.add_argument("--disturbance", default="clear",
                        choices=["clear", "light", "moderate", "severe", "extreme"])
    parser.add_argument("--output", default="runs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    result = run_simulation(
        trajectory=args.trajectory,
        duration_s=args.duration,
        dt=args.dt,
        seed=args.seed,
        target_size=args.target_size,
        disturbance=args.disturbance,
        output_dir=args.output,
        verbose=args.verbose,
    )
    sys.exit(0 if result["frame_count"] > 0 else 1)


if __name__ == "__main__":
    main()
