"""Generate the 5-minute closed-loop application demo video.

Runs a genuine closed-loop tracking run (sim render -> detect ->
centroid -> Kalman -> PID -> camera) and writes every rendered frame
with the square-bracket tracking reticle, velocity vector, and
telemetry bar burned in. All overlays show measured per-frame
quantities — nothing staged.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")

from fsoc_tracker.control.controller import (
    CameraActuator,
    CoarsePointingController,
)
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.tracker import KalmanTracker

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30.0, 300
DT = 1.0 / FPS
OUT = "artifacts/demo/demo_app_5min.mp4"


def _draw_brackets(frame, cx, cy, half, color) -> None:
    bl = max(half // 3, 6)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, y0 = cx + sx * half, cy + sy * half
            cv2.line(frame, (x0, y0), (x0 - sx * bl, y0), color, 2,
                     cv2.LINE_AA)
            cv2.line(frame, (x0, y0), (x0, y0 - sy * bl), color, 2,
                     cv2.LINE_AA)
    ch = max(half // 2, 5)
    cv2.line(frame, (cx - ch, cy), (cx + ch, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - ch), (cx, cy + ch), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)


def main() -> None:
    import os
    os.makedirs("artifacts/demo", exist_ok=True)

    engine = SimulationEngine(WorldConfig(width=2000.0, height=2000.0,
                                          random_seed=42))
    engine.add_target(
        trajectory_type="sinusoidal",
        trajectory_params={"cx": 1000.0, "cy": 1000.0, "cz": 600.0,
                           "amplitude_x": 22.0, "amplitude_y": 18.0,
                           "amplitude_z": 12.0, "freq_x": 0.08,
                           "freq_y": 0.05, "freq_z": 0.03,
                           "phase_x_rad": 0.0, "phase_y_rad": 1.2,
                           "phase_z_rad": 0.7},
    )
    camera = VirtualCamera(CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=WIDTH, height=HEIGHT,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0))
    camera.set_target_pan_tilt(1.0, 0.0)
    camera.update(1.0)  # start misaligned: genuine search -> acquire
    sensor = VirtualSensorRenderer(SensorConfig(width=WIDTH, height=HEIGHT))
    detector = ClassicalBeaconDetector()
    tracker = KalmanTracker()
    controller = CoarsePointingController()
    actuator = CameraActuator()

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (WIDTH, HEIGHT))
    sim_time, lat = 0.0, 0.0
    n_frames = int(FPS * SECONDS)
    t_wall = time.perf_counter()
    for i in range(n_frames):
        t0 = time.perf_counter()
        engine.step(DT)
        rendered = sensor.render(
            camera, engine.get_state().get_active_targets(), sim_time, i)
        det = detector.detect(rendered.image, sim_time, i)
        dets = (det.detections if det.primary_detection
                and det.primary_detection.detected else [])
        trk = tracker.update(dets, sim_time)
        cmd, _ = controller.compute(trk, camera.intrinsics, DT, sim_time)
        actuator.apply_command(camera, cmd, DT)
        lat = 0.9 * lat + 0.1 * (time.perf_counter() - t0) * 1000.0

        frame = rendered.image
        frame = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                 if frame.ndim == 2 else frame.copy())
        state_name = trk.state.name
        locked = bool(getattr(trk, "locked", False))
        color = ((57, 229, 140) if locked
                 else (84, 180, 255) if state_name in ("TRACKING", "ACQUIRING")
                 else (60, 60, 220))

        # Estimate reticle (measured tracker output)
        ex, ey = int(trk.estimated_x), int(trk.estimated_y)
        if 0 <= ex < WIDTH and 0 <= ey < HEIGHT:
            _draw_brackets(frame, ex, ey, 22, color)
            # Velocity vector (measured Kalman velocity, 0.5 s ahead)
            vx, vy = float(trk.velocity_x), float(trk.velocity_y)
            px2, py2 = int(ex + 0.5 * vx), int(ey + 0.5 * vy)
            cv2.arrowedLine(frame, (ex, ey), (px2, py2), (34, 153, 210), 1,
                            cv2.LINE_AA, tipLength=0.25)

        # Raw detection ring (measured detector output)
        if det.primary_detection and det.primary_detection.detected:
            cx = int(det.primary_detection.center_x)
            cy = int(det.primary_detection.center_y)
            cv2.circle(frame, (cx, cy), 8, (255, 180, 84), 1, cv2.LINE_AA)

        # Camera center
        cv2.drawMarker(frame, (WIDTH // 2, HEIGHT // 2), (200, 200, 200),
                       cv2.MARKER_CROSS, 16, 1)

        # Telemetry bar (measured values only)
        bar = np.zeros((34, WIDTH, 3), dtype=np.uint8)
        if det.primary_detection and det.primary_detection.detected:
            dcx = det.primary_detection.center_x
            dcy = det.primary_detection.center_y
            err = ((ex - dcx) ** 2 + (ey - dcy) ** 2) ** 0.5
            txt = (f"{state_name} lock={int(locked)} centroid={dcx:.1f},{dcy:.1f} "
                   f"err={err:.1f}px pan={camera.state.pan_deg:+.2f} "
                   f"tilt={camera.state.tilt_deg:+.2f} fps={1.0 / max(lat / 1000.0, 1e-6):.0f}")
        else:
            txt = (f"{state_name} lock={int(locked)} centroid=-- err=-- "
                   f"pan={camera.state.pan_deg:+.2f} "
                   f"tilt={camera.state.tilt_deg:+.2f} fps={1.0 / max(lat / 1000.0, 1e-6):.0f}")
        cv2.putText(bar, txt[:76], (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)
        out.write(np.vstack([frame[:HEIGHT - 34], bar]))
        sim_time += DT
        if i % (FPS * 60) == 0:
            print(f"  {i / n_frames * 100:5.1f}%  frame {i}/{n_frames}")
    out.release()
    print(f"\nWrote {OUT} ({n_frames} frames, {SECONDS}s, "
          f"rendered in {time.perf_counter() - t_wall:.1f}s)")


if __name__ == "__main__":
    main()
