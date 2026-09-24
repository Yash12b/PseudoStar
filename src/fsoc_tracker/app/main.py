"""FSOC Tracker entry point.

Run with: ``python -m fsoc_tracker``

Smoke test exercises the real TrackingPipeline:
    SimulationSource -> Perception -> Tracking -> Control -> Camera
"""

from __future__ import annotations

import argparse
import sys

from fsoc_tracker.config.settings import load_config
from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
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
from fsoc_tracker.utils.logging import get_logger, setup_logging


def _run_smoke_test(config_path: str | None = None) -> int:
    """Execute a real pipeline smoke test.

    Returns:
        0 on success, 1 on failure.
    """
    config = load_config(config_path)

    setup_logging(
        log_level=config.app.log_level,
        log_dir=config.app.log_dir,
        session_id=config.app.session_id,
    )
    logger = get_logger("smoke_test")

    logger.info("=== FSOC Tracker Smoke Test ===")
    logger.info("Session: %s", config.app.session_id)
    logger.info("Camera: %dx%d @ %.1f Hz", config.camera.width, config.camera.height, config.camera.update_rate_hz)

    # --- Build simulation ---
    scenario = make_experiment_world(motion="straight_line", seed=42)
    wc = WorldConfig(width=scenario.world_width, height=scenario.world_height, random_seed=42)
    engine = SimulationEngine(wc)
    engine.load_scenario(scenario)

    cam_state = CameraState(
        horizontal_fov_deg=config.camera.horizontal_fov_deg,
        vertical_fov_deg=config.camera.vertical_fov_deg,
        width=config.camera.width,
        height=config.camera.height,
        max_pan_speed_deg_s=config.control.max_pan_speed_deg_s,
        max_tilt_speed_deg_s=config.control.max_tilt_speed_deg_s,
        position_x=scenario.terminal.x,
        position_y=scenario.terminal.y,
        position_z=scenario.terminal.z,
        pan_deg=scenario.terminal.yaw_deg,
        tilt_deg=scenario.terminal.pitch_deg,
    )
    camera = VirtualCamera(cam_state)

    sc = SensorConfig(
        width=config.camera.width,
        height=config.camera.height,
        beacon_default_size_px=config.target.size_px,
    )
    sensor = VirtualSensorRenderer(sc)

    dt = 1.0 / config.camera.update_rate_hz
    source = SimulationSource(engine, camera, sensor, None, dt, eval_sink=EvalSink())

    # --- Build real pipeline ---
    perception_cfg = PerceptionConfig(
        threshold_mode="global",
        threshold_value=20.0,
    )
    pipeline = TrackingPipeline(
        perception=ClassicalBeaconDetector(perception_cfg),
        tracker=KalmanTracker(),
        controller=CoarsePointingController(),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)

    session = SessionController(pipeline, output_dir="runs", auto_save=False)
    session.configure(source_type="simulation", source_info="smoke_test")

    # --- Run ---
    session.start()
    frame_count = 0
    max_frames = int(config.camera.update_rate_hz * 5)  # 5 seconds max

    try:
        while session.state.value == "RUNNING" and frame_count < max_frames:
            result = session.step()
            if result is None:
                break
            frame_count += 1
            if frame_count % 15 == 0:
                track_state = result.tracking.state.name if result.tracking else "?"
                err = f"{result.error_px:.1f}px" if result.error_px is not None else "N/A"
                logger.info(
                    "Frame %d | dt=%.4fs | track=%s | err=%s | perce=%.1fms track=%.1fms ctrl=%.1fms",
                    frame_count, result.dt, track_state, err,
                    result.perception_ms, result.tracking_ms, result.control_ms,
                )
    except Exception:
        logger.exception("Smoke test failed")
        return 1
    finally:
        session.stop()

    # --- Summary ---
    ps = pipeline.state
    logger.info("=== Smoke Test Summary ===")
    logger.info("Frames processed: %d", ps.frame_count)
    if ps.processing_times_ms:
        import statistics
        mean_ms = statistics.mean(ps.processing_times_ms)
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        logger.info("Mean latency: %.1fms (%.1f FPS)", mean_ms, fps)
    if ps.acquisition_time_s is not None:
        logger.info("Acquisition time: %.3fs", ps.acquisition_time_s)
    logger.info("Loss events: %d", ps.loss_events)
    logger.info("Session: %s", config.app.session_id)
    logger.info("Smoke test PASSED.")
    return 0


def main() -> None:
    """Launch the desktop application by default.

    Headless smoke testing remains available explicitly with ``--smoke`` so
    the packaged application is double-clickable and never requires a shell.
    """
    parser = argparse.ArgumentParser(
        prog="fsoc-tracker",
        description="FSOC Tracker - AI-Based Virtual Camera Tracking System",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: built-in defaults)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI (the default)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the headless pipeline smoke test instead of the GUI",
    )
    args = parser.parse_args()

    if args.smoke:
        sys.exit(_run_smoke_test(config_path=args.config))

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for the desktop application. "
            "Install the desktop extra with: pip install 'fsoc-tracker[desktop]'"
        ) from exc

    from fsoc_tracker.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
