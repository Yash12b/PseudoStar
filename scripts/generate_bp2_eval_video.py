"""Generate a 60 s noisy evaluation video with ground-truth sidecar.

Beacon stays in frame on a sinusoidal path; Gaussian sigma=20 plus
10% salt-and-pepper noise applied per frame. Writes MP4 + gt.csv
(frame_index, timestamp_s, true_x, true_y, visible) for BP2 scoring.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")

from fsoc_tracker.disturbances.config import DisturbanceConfig, NoiseConfig
from fsoc_tracker.disturbances.context import DisturbanceContext
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30.0, 60
DT = 1.0 / FPS
OUT = "artifacts/demo/bp2_noisy_eval.mp4"
GT = "artifacts/demo/bp2_noisy_eval_gt.csv"


def main() -> None:
    import os
    os.makedirs("artifacts/demo", exist_ok=True)

    engine = SimulationEngine(WorldConfig(random_seed=11))
    engine.add_target(
        trajectory_type="sinusoidal",
        trajectory_params={"cx": 1000.0, "cy": 1000.0, "cz": 600.0,
                           "amplitude_x": 14.0, "amplitude_y": 10.0,
                           "freq_x": 0.06, "freq_y": 0.045,
                           "phase_x_rad": 0.0, "phase_y_rad": 0.8},
    )
    camera = VirtualCamera(CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=WIDTH, height=HEIGHT,
        position_x=1000.0, position_y=1000.0, position_z=50.0,
        pan_deg=0.0, tilt_deg=0.0,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0))
    sensor = VirtualSensorRenderer(SensorConfig(width=WIDTH, height=HEIGHT))
    dist = DisturbancePipeline(DisturbanceConfig(
        enabled=True,
        noise=NoiseConfig(enabled=True, gaussian_sigma=20.0,
                          salt_pepper_density=0.10)))

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (WIDTH, HEIGHT))
    n = int(FPS * SECONDS)
    rows = ["frame_index,timestamp_s,true_x,true_y,visible"]
    sim_time = 0.0
    for i in range(n):
        engine.step(DT)
        rendered = sensor.render(
            camera, engine.get_state().get_active_targets(), sim_time, i)
        frame = np.ascontiguousarray(rendered.image.astype(np.uint8))
        ctx = DisturbanceContext(
            timestamp_s=sim_time, dt=DT, frame_index=i,
            image_width=WIDTH, image_height=HEIGHT, random_seed=i)
        frame = np.ascontiguousarray(
            np.clip(dist.apply_to_image(frame, ctx), 0, 255).astype(np.uint8))
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        gt = rendered.ground_truths[0] if rendered.ground_truths else None
        if gt is not None and gt.target_visible:
            rows.append(f"{i},{sim_time:.4f},{gt.target_pixel_x:.2f},"
                        f"{gt.target_pixel_y:.2f},1")
        else:
            rows.append(f"{i},{sim_time:.4f},,,0")
        sim_time += DT
    out.release()
    with open(GT, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Wrote {OUT} + {GT} ({n} frames)")


if __name__ == "__main__":
    main()
