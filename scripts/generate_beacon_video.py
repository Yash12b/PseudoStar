"""Generate a 5-minute 3D beacon video for FSOC tracker testing.

Raw rendered beacon on dark background — no overlays. Feed directly
into the tracker to test detection and closed-loop tracking.
"""
import sys
import time

import cv2

sys.path.insert(0, "src")

from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig

WIDTH, HEIGHT, FPS = 640, 480, 30.0
DT = 1.0 / FPS
DURATION_S = 300
OUT = "artifacts/demo/beacon_5min_3d.mp4"


def main() -> None:
    import os
    os.makedirs("artifacts/demo", exist_ok=True)

    world_cfg = WorldConfig(width=2000.0, height=2000.0, depth=2000.0,
                            random_seed=7)
    engine = SimulationEngine(world_cfg)

    engine.add_target(
        trajectory_type="sinusoidal",
        trajectory_params={
            "cx": 1000.0, "cy": 1000.0, "cz": 600.0,
            "amplitude_x": 22.0,
            "amplitude_y": 18.0,
            "amplitude_z": 12.0,
            "freq_x": 0.08,
            "freq_y": 0.05,
            "freq_z": 0.03,
            "phase_x_rad": 0.0,
            "phase_y_rad": 1.2,
            "phase_z_rad": 0.7,
        },
    )

    camera = VirtualCamera(CameraState(
        horizontal_fov_deg=4.0,
        vertical_fov_deg=3.0,
        width=WIDTH, height=HEIGHT,
        position_x=1000.0, position_y=1000.0, position_z=50.0,
        pan_deg=0.0, tilt_deg=0.0,
        max_pan_speed_deg_s=5.0,
        max_tilt_speed_deg_s=5.0,
    ))

    sensor = VirtualSensorRenderer(SensorConfig(
        width=WIDTH, height=HEIGHT,
        background_level=5,
        beacon_peak_intensity=255.0,
        psf_sigma_px=2.5,
    ))

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (WIDTH, HEIGHT))
    n_frames = int(FPS * DURATION_S)
    sim_time = 0.0
    t_wall = time.perf_counter()

    for i in range(n_frames):
        engine.step(DT)
        rendered = sensor.render(
            camera, engine.get_state().get_active_targets(), sim_time, i)
        frame = rendered.image
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
        sim_time += DT
        if i % (FPS * 60) == 0:
            print(f"  {i / n_frames * 100:5.1f}%  frame {i}/{n_frames}")

    out.release()
    wall = time.perf_counter() - t_wall
    print(f"\nWrote {OUT}  ({n_frames} frames, {DURATION_S}s, "
          f"rendered in {wall:.1f}s)")


if __name__ == "__main__":
    main()
