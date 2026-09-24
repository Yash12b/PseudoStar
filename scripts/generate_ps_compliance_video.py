"""Generate a 4-minute PS-compliance test video.

One labeled 15 s segment per problem-statement parameter group, all at
640x480 @30 fps with a fixed 4x3 deg camera at screen centre:
  P3/P4/P5/P7/P10  baseline (resolution, FOV, rate, spot, default size)
  P10              beacon size 5 px and 20 px
  P9               shape circular and gaussian spot (square = baseline)
  P12              motion: circular, figure-8, random, spiral, sinusoidal
  P8               multi-target (3 beacons)
  P21              gaussian noise s=20, salt&pepper 10%, poisson+lowlight,
                   fog+rain+jitter +-20px
  P18/P19          FOV exit and re-entry (loss / re-acquisition)

Feed the output to the tracker in VIDEO mode: every pixel is raw
sensor output with no overlays of any kind (segment identity lives in
this script's SEGMENTS table and the manual's time map — burned-in
text would hand the tracker free static targets).
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")

from fsoc_tracker.disturbances.config import (
    AtmosphereConfig,
    AtmosphereMode,
    DisturbanceConfig,
    NoiseConfig,
)
from fsoc_tracker.disturbances.context import DisturbanceContext
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import BeaconShape, SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig

WIDTH, HEIGHT, FPS = 640, 480, 30.0
DT = 1.0 / FPS
SEG_S = 15
BAR_H = 34
OUT = "artifacts/demo/ps_compliance_4min.mp4"

BASE_CENTER = {"cx": 1000.0, "cy": 1000.0, "cz": 600.0}
BASE_STRAIGHT = {"x0": 1000.0, "y0": 1000.0, "z0": 600.0,
                 "vx": 0.5, "vy": 0.2}
BASE_BOUNDED_RANDOM = {"x0": 1000.0, "y0": 1000.0, "z0": 600.0,
                       "step_size": 1.0, "step_duration": 0.15,
                       "x_min": 980.0, "x_max": 1020.0,
                       "y_min": 985.0, "y_max": 1015.0,
                       "z_min": 500.0, "z_max": 700.0, "seed": 7}


def _engine(motion, params, n_extra=0):
    eng = SimulationEngine(WorldConfig(width=2000.0, height=2000.0,
                                       random_seed=7))
    eng.add_target(trajectory_type=motion, trajectory_params=params)
    for k in range(n_extra):
        eng.add_target(
            trajectory_type="circular",
            trajectory_params={"cx": 1000.0 + (k + 1) * 8.0, "cy": 1000.0,
                               "cz": 600.0 + k * 40.0, "radius": 6.0,
                               "angular_speed_rad_s": 0.4 + 0.2 * k})
    return eng


def _sensor(size=10.0, shape="spot"):
    if shape == "circular":
        cfg = SensorConfig(width=WIDTH, height=HEIGHT,
                           beacon_default_size_px=size,
                           beacon_shape=BeaconShape.CIRCULAR,
                           beacon_soft_edges=False)
    elif shape == "square":
        cfg = SensorConfig(width=WIDTH, height=HEIGHT,
                           beacon_default_size_px=size,
                           beacon_shape=BeaconShape.SQUARE,
                           beacon_soft_edges=False)
    else:
        cfg = SensorConfig(width=WIDTH, height=HEIGHT,
                           beacon_default_size_px=size,
                           beacon_soft_edges=True)
    return VirtualSensorRenderer(cfg)


def _dist_noise_gauss():
    return DisturbancePipeline(DisturbanceConfig(
        enabled=True, noise=NoiseConfig(enabled=True, gaussian_sigma=20.0)))


def _dist_noise_sp():
    return DisturbancePipeline(DisturbanceConfig(
        enabled=True, noise=NoiseConfig(enabled=True,
                                        salt_pepper_density=0.10)))


def _dist_noise_pois_lowlight():
    return DisturbancePipeline(DisturbanceConfig(
        enabled=True,
        noise=NoiseConfig(enabled=True, poisson_enabled=True,
                          poisson_scale=2.0),
        atmosphere=AtmosphereConfig(enabled=True, mode=AtmosphereMode.LOW_LIGHT,
                                    low_light_factor=0.6)))


def _dist_fog_rain():
    return DisturbancePipeline(DisturbanceConfig(
        enabled=True,
        atmosphere=AtmosphereConfig(enabled=True, mode=AtmosphereMode.FOG,
                                    fog_strength=0.5, rain_density=0.5)))


SEGMENTS = [
    ("P3/P4/P5/P7/P10 BASE 640x480 4x3deg 30fps 10px",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(10.0, "square"), None, 0)),
    ("P10 SIZE 5px (min)",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(5.0, "square"), None, 0)),
    ("P10 SIZE 20px (max)",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(20.0, "square"), None, 0)),
    ("P9 SHAPE circular",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(10.0, "circular"), None, 0)),
    ("P9 SHAPE gaussian spot",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(10.0, "spot"), None, 0)),
    ("P12 MOTION circular",
     lambda: (_engine("circular", dict(BASE_CENTER, radius=12.0,
                                       angular_speed_rad_s=0.5)),
              _sensor(), None, 0)),
    ("P12 MOTION figure-8",
     lambda: (_engine("figure_8", dict(BASE_CENTER, amplitude_x=14.0,
                                       amplitude_y=9.0)),
              _sensor(), None, 0)),
    ("P12 MOTION random",
     lambda: (_engine("random_walk", dict(BASE_BOUNDED_RANDOM)),
              _sensor(), None, 0)),
    ("P12 MOTION spiral",
     lambda: (_engine("spiral", dict(BASE_CENTER, radius_start=3.0,
                                     radius_growth=0.6,
                                     angular_speed_rad_s=0.6)),
              _sensor(), None, 0)),
    ("P12 MOTION sinusoidal",
     lambda: (_engine("sinusoidal", dict(BASE_CENTER, amplitude_x=16.0,
                                         amplitude_y=11.0, freq_x=0.08,
                                         freq_y=0.05)),
              _sensor(), None, 0)),
    ("P8 MULTI-TARGET x3",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT),
                      n_extra=2),
              _sensor(), None, 0)),
    ("P21 NOISE gaussian sigma=20",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(), _dist_noise_gauss(), 0)),
    ("P21 NOISE salt&pepper 10pct",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(), _dist_noise_sp(), 0)),
    ("P21 NOISE poisson + LOW LIGHT",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(), _dist_noise_pois_lowlight(), 0)),
    ("P21 FOG+RAIN+JITTER +-20px",
     lambda: (_engine("straight_line", dict(BASE_STRAIGHT)),
              _sensor(), _dist_fog_rain(), 20)),
    ("P18/P19 FOV EXIT + REACQUIRE",
     lambda: (_engine("sinusoidal", dict(BASE_CENTER, amplitude_x=30.0,
                                         amplitude_y=22.0, freq_x=0.08,
                                         freq_y=0.05)),
              _sensor(), None, 0)),
]


def main() -> None:
    import os
    os.makedirs("artifacts/demo", exist_ok=True)

    camera = VirtualCamera(CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=WIDTH, height=HEIGHT,
        position_x=1000.0, position_y=1000.0, position_z=50.0,
        pan_deg=0.0, tilt_deg=0.0,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0))

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (WIDTH, HEIGHT))
    n_seg_frames = int(FPS * SEG_S)
    sim_time, fi = 0.0, 0
    rng = np.random.default_rng(11)
    t_wall = time.perf_counter()

    for si, (label, build) in enumerate(SEGMENTS):
        eng, sensor, dist, jitter_px = build()
        for i in range(n_seg_frames):
            eng.step(DT)
            rendered = sensor.render(
                camera, eng.get_state().get_active_targets(), sim_time, fi)
            frame = rendered.image
            if dist is not None:
                ctx = DisturbanceContext(
                    timestamp_s=sim_time, dt=DT, frame_index=fi,
                    image_width=WIDTH, image_height=HEIGHT, random_seed=fi)
                frame = dist.apply_to_image(
                    np.ascontiguousarray(frame.astype(np.uint8)), ctx)
            if jitter_px:
                dx = int(rng.integers(-jitter_px, jitter_px + 1))
                dy = int(rng.integers(-jitter_px, jitter_px + 1))
                frame = np.roll(np.roll(frame, dy, axis=0), dx, axis=1)
            frame = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
            frame = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                     if frame.ndim == 2 else frame.copy())
            out.write(frame)
            sim_time += DT
            fi += 1
        print(f"  seg {si + 1}/{len(SEGMENTS)} done: {label[:40]}")

    out.release()
    total_s = len(SEGMENTS) * SEG_S
    print(f"\nWrote {OUT} ({fi} frames, {total_s}s = {total_s / 60:.1f} min, "
          f"rendered in {time.perf_counter() - t_wall:.1f}s)")


if __name__ == "__main__":
    main()
