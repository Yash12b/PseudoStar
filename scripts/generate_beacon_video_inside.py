"""Generate a 5-minute video: beacon in 3D random motion, always in frame.

Raw sensor output — no overlays. The beacon performs a projection-guarded
3D random walk: every proposed step is projected through the fixed camera
and accepted only if the beacon stays fully inside the frame, so the
beacon is visible in 100% of frames while moving randomly in X/Y/Z.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")

from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.target import WorldTargetState

WIDTH, HEIGHT, FPS = 640, 480, 30.0
DT = 1.0 / FPS
DURATION_S = 300
OUT = "artifacts/demo/beacon_5min_inside.mp4"

MARGIN_X = 40
MARGIN_Y = 30
Z_MIN, Z_MAX = 500.0, 700.0
STEP_XY = 1.5
STEP_Z = 1.0
SEED = 21


def _inside(camera: VirtualCamera, x: float, y: float, z: float) -> bool:
    proj = camera.project_world_point((x, y, z))
    if proj.depth <= 0:
        return False
    return (MARGIN_X <= proj.pixel_x <= WIDTH - MARGIN_X
            and MARGIN_Y <= proj.pixel_y <= HEIGHT - MARGIN_Y
            and Z_MIN <= z <= Z_MAX)


def main() -> None:
    import os
    os.makedirs("artifacts/demo", exist_ok=True)

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

    rng = np.random.default_rng(SEED)
    x, y, z = 1000.0, 1000.0, 600.0
    target = WorldTargetState(target_id=0, brightness=1.0)

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (WIDTH, HEIGHT))
    n_frames = int(FPS * DURATION_S)
    sim_time = 0.0
    visible_frames = 0
    t_wall = time.perf_counter()

    for i in range(n_frames):
        # Propose random 3D steps; keep the first one that stays in frame.
        for _ in range(20):
            cx = x + float(rng.uniform(-STEP_XY, STEP_XY))
            cy = y + float(rng.uniform(-STEP_XY, STEP_XY))
            cz = z + float(rng.uniform(-STEP_Z, STEP_Z))
            if _inside(camera, cx, cy, cz):
                x, y, z = cx, cy, cz
                break

        target.x, target.y, target.z = x, y, z
        rendered = sensor.render(camera, [target], sim_time, i)
        gt = rendered.ground_truths[0] if rendered.ground_truths else None
        if gt is not None and gt.target_visible:
            visible_frames += 1

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
    print(f"Beacon visible: {visible_frames}/{n_frames} "
          f"({100.0 * visible_frames / n_frames:.1f}%)")


if __name__ == "__main__":
    main()
