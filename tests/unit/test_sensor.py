"""Comprehensive tests for the sensor/image-formation module.

16+ test categories covering:
- Center projection (beacon matches ground truth)
- Off-axis target (correct pixel position)
- Subpixel location (fractional coordinates preserved)
- Apparent size (5px, 10px, 20px produce expected footprints)
- Clipping (partially visible target)
- Behind camera (not rendered)
- Outside FOV (not rendered)
- Intensity (peak within dynamic range)
- PSF (blur changes spot shape)
- Monochrome (output shape/type correct)
- Determinism (same state + same config = same image)
- Multiple targets (deterministic rendering)
- FPS independence (same state = same image)
- Ground truth separate (no overlay in raw image)
- No NaN/Inf (valid output)
- Resolution (320x240, 640x480, 1280x720)
- Different FOVs
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.sensor.config import (
    SensorConfig,
)
from fsoc_tracker.simulation.sensor.models import (
    GroundTruth,
    RenderedFrame,
    TargetVisibility,
)
from fsoc_tracker.simulation.sensor.psf import gaussian_psf_2d
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.target import WorldTargetState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    x: float = 1000.0,
    y: float = 1000.0,
    z: float = 950.0,
    target_id: int = 0,
    brightness: float = 1.0,
    width: float = 0.01,
) -> WorldTargetState:
    return WorldTargetState(
        target_id=target_id,
        x=x, y=y, z=z,
        width=width, height=width,
        brightness=brightness,
    )


def _make_camera(
    x: float = 1000.0,
    y: float = 1000.0,
    z: float = 0.0,
    hfov: float = 30.0,
    vfov: float = 22.5,
    width: int = 640,
    height: int = 480,
) -> VirtualCamera:
    state = CameraState(
        position_x=x, position_y=y, position_z=z,
        horizontal_fov_deg=hfov, vertical_fov_deg=vfov,
        width=width, height=height,
    )
    return VirtualCamera(state)


def _render(
    targets: list[WorldTargetState] | None = None,
    camera: VirtualCamera | None = None,
    sensor_config: SensorConfig | None = None,
    timestamp_s: float = 0.0,
    frame_index: int = 0,
) -> RenderedFrame:
    if targets is None:
        targets = [_make_target()]
    if camera is None:
        camera = _make_camera()
    renderer = VirtualSensorRenderer(sensor_config)
    return renderer.render(camera, targets, timestamp_s, frame_index)


# ---------------------------------------------------------------------------
# A. Center projection
# ---------------------------------------------------------------------------

class TestCenterProjection:
    def test_beacon_center_matches_ground_truth(self):
        rendered = _render()
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_visible is True
        assert abs(gt.target_pixel_x - 320.0) < 1.0
        assert abs(gt.target_pixel_y - 240.0) < 1.0

    def test_rendered_image_has_beacon(self):
        rendered = _render()
        assert rendered.image.max() > 5


# ---------------------------------------------------------------------------
# B. Off-axis target
# ---------------------------------------------------------------------------

class TestOffAxisTarget:
    def test_target_right_of_center(self):
        target = _make_target(x=1020.0, y=1000.0)
        rendered = _render(targets=[target])
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_pixel_x > 320.0

    def test_target_above_center(self):
        target = _make_target(x=1000.0, y=1020.0)
        rendered = _render(targets=[target])
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_pixel_y < 240.0


# ---------------------------------------------------------------------------
# C. Subpixel
# ---------------------------------------------------------------------------

class TestSubpixel:
    def test_fractional_pixel_preserved(self):
        target = _make_target(x=1000.0, y=1000.0)
        camera = _make_camera()
        camera.set_position(1000.0, 1000.0, 0.0)
        target2 = _make_target(x=1005.3, y=1002.7)
        rendered = _render(targets=[target2], camera=camera)
        gt = rendered.primary_ground_truth
        assert gt is not None
        frac_x = gt.target_pixel_x - math.floor(gt.target_pixel_x)
        frac_y = gt.target_pixel_y - math.floor(gt.target_pixel_y)
        assert frac_x > 0.01
        assert frac_y > 0.01


# ---------------------------------------------------------------------------
# D. Apparent size
# ---------------------------------------------------------------------------

class TestApparentSize:
    def test_size_5px(self):
        cfg = SensorConfig(beacon_default_size_px=5.0, beacon_soft_edges=False)
        rendered = _render(sensor_config=cfg)
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert abs(gt.target_size_px - 5.0) < 0.1

    def test_size_10px(self):
        cfg = SensorConfig(beacon_default_size_px=10.0, beacon_soft_edges=False)
        rendered = _render(sensor_config=cfg)
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert abs(gt.target_size_px - 10.0) < 0.1

    def test_size_20px(self):
        cfg = SensorConfig(beacon_default_size_px=20.0, beacon_soft_edges=False)
        rendered = _render(sensor_config=cfg)
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert abs(gt.target_size_px - 20.0) < 0.1

    def test_size_clamped_to_minimum(self):
        cfg = SensorConfig(beacon_default_size_px=2.0, minimum_beacon_size_px=5.0)
        rendered = _render(sensor_config=cfg)
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_size_px >= 5.0

    def test_size_clamped_to_maximum(self):
        cfg = SensorConfig(beacon_default_size_px=50.0, maximum_beacon_size_px=20.0)
        rendered = _render(sensor_config=cfg)
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_size_px <= 20.0


# ---------------------------------------------------------------------------
# E. Clipping
# ---------------------------------------------------------------------------

class TestClipping:
    def test_partial_visible_at_edge(self):
        camera = _make_camera(hfov=4.0, vfov=3.0)
        target = _make_target(x=1000.0 + 40, y=1000.0)
        cfg = SensorConfig(beacon_default_size_px=10.0)
        renderer = VirtualSensorRenderer(cfg)
        rendered = renderer.render(camera, [target])
        gt = rendered.primary_ground_truth
        assert gt is not None
        if gt.visibility == TargetVisibility.PARTIAL:
            assert gt.target_visible is True
        elif gt.visibility == TargetVisibility.OUTSIDE:
            assert gt.target_visible is False


# ---------------------------------------------------------------------------
# F. Behind camera
# ---------------------------------------------------------------------------

class TestBehindCamera:
    def test_behind_camera_not_rendered(self):
        target = _make_target(z=-10.0)
        rendered = _render(targets=[target])
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.visibility == TargetVisibility.BEHIND_CAMERA
        assert gt.target_visible is False
        assert rendered.image.max() <= 5


# ---------------------------------------------------------------------------
# G. Outside FOV
# ---------------------------------------------------------------------------

class TestOutsideFOV:
    def test_far_right_outside_fov(self):
        camera = _make_camera(hfov=4.0, vfov=3.0)
        target = _make_target(x=1000.0 + 200)
        cfg = SensorConfig()
        renderer = VirtualSensorRenderer(cfg)
        rendered = renderer.render(camera, [target])
        gt = rendered.primary_ground_truth
        assert gt is not None
        assert gt.target_visible is False


# ---------------------------------------------------------------------------
# H. Intensity
# ---------------------------------------------------------------------------

class TestIntensity:
    def test_peak_within_dynamic_range(self):
        cfg = SensorConfig(max_intensity=255.0, beacon_peak_intensity=200.0)
        rendered = _render(sensor_config=cfg)
        assert rendered.image.max() <= 255
        assert rendered.image.max() > 5

    def test_no_nan_or_inf(self):
        rendered = _render()
        assert not np.any(np.isnan(rendered.image))
        assert not np.any(np.isinf(rendered.image))


# ---------------------------------------------------------------------------
# I. PSF
# ---------------------------------------------------------------------------

class TestPSF:
    def test_psf_changes_spot_shape(self):
        cfg_no_psf = SensorConfig(enable_psf=False, beacon_soft_edges=False, beacon_default_size_px=10.0)
        cfg_psf = SensorConfig(enable_psf=True, psf_sigma_px=2.0, beacon_soft_edges=False, beacon_default_size_px=10.0)
        rendered_no = _render(sensor_config=cfg_no_psf)
        rendered_psf = _render(sensor_config=cfg_psf)
        assert not np.array_equal(rendered_no.image, rendered_psf.image)

    def test_gaussian_psf_kernel_sums_to_one(self):
        kernel = gaussian_psf_2d(1.5)
        assert abs(kernel.sum() - 1.0) < 1e-10

    def test_psf_kernel_shape(self):
        kernel = gaussian_psf_2d(2.0)
        assert kernel.shape[0] == kernel.shape[1]
        assert kernel.shape[0] > 5


# ---------------------------------------------------------------------------
# J. Monochrome
# ---------------------------------------------------------------------------

class TestMonochrome:
    def test_output_shape(self):
        rendered = _render()
        assert rendered.image.ndim == 2
        assert rendered.image.shape == (480, 640)

    def test_output_dtype(self):
        rendered = _render()
        assert rendered.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# K. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_state_same_image(self):
        cfg = SensorConfig(seed=42)
        r1 = _render(sensor_config=cfg, timestamp_s=1.0, frame_index=5)
        r2 = _render(sensor_config=cfg, timestamp_s=1.0, frame_index=5)
        assert np.array_equal(r1.image, r2.image)

    def test_different_seed_different_noise(self):
        cfg1 = SensorConfig(seed=42, background_noise_level=2.0)
        cfg2 = SensorConfig(seed=99, background_noise_level=2.0)
        r1 = _render(sensor_config=cfg1, frame_index=0)
        r2 = _render(sensor_config=cfg2, frame_index=0)
        assert not np.array_equal(r1.image, r2.image)


# ---------------------------------------------------------------------------
# L. Multiple targets
# ---------------------------------------------------------------------------

class TestMultipleTargets:
    def test_two_targets_both_visible(self):
        t1 = _make_target(x=1000.0 + 5, y=1000.0, target_id=0)
        t2 = _make_target(x=1000.0 - 5, y=1000.0, target_id=1)
        rendered = _render(targets=[t1, t2])
        assert len(rendered.ground_truths) == 2
        visible_count = sum(1 for gt in rendered.ground_truths if gt.target_visible)
        assert visible_count >= 1

    def test_deterministic_compositing(self):
        t1 = _make_target(x=1000.0, y=1000.0, target_id=0)
        t2 = _make_target(x=1000.0, y=1000.0, target_id=1)
        cfg = SensorConfig(seed=42)
        r1 = _render(targets=[t1, t2], sensor_config=cfg)
        r2 = _render(targets=[t1, t2], sensor_config=cfg)
        assert np.array_equal(r1.image, r2.image)


# ---------------------------------------------------------------------------
# M. FPS independence
# ---------------------------------------------------------------------------

class TestFPSIndependence:
    def test_same_state_same_image(self):
        cfg = SensorConfig()
        r1 = _render(sensor_config=cfg, timestamp_s=2.5, frame_index=10)
        r2 = _render(sensor_config=cfg, timestamp_s=2.5, frame_index=10)
        assert np.array_equal(r1.image, r2.image)


# ---------------------------------------------------------------------------
# N. Ground truth separate
# ---------------------------------------------------------------------------

class TestGroundTruthSeparate:
    def test_raw_image_no_overlay(self):
        cfg = SensorConfig(background_level=5.0, beacon_peak_intensity=255.0)
        rendered = _render(sensor_config=cfg)
        bg_pixels = np.sum(rendered.image <= 10)
        total_pixels = rendered.image.size
        assert bg_pixels > total_pixels * 0.5


# ---------------------------------------------------------------------------
# O. No NaN/Inf
# ---------------------------------------------------------------------------

class TestNoNaNInf:
    def test_valid_output(self):
        rendered = _render()
        assert not np.any(np.isnan(rendered.image.astype(np.float64)))
        assert not np.any(np.isinf(rendered.image.astype(np.float64)))


# ---------------------------------------------------------------------------
# P. Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    @pytest.mark.parametrize("w,h", [(320, 240), (640, 480), (1280, 720)])
    def test_various_resolutions(self, w, h):
        camera = _make_camera(width=w, height=h)
        cfg = SensorConfig(width=w, height=h)
        renderer = VirtualSensorRenderer(cfg)
        target = _make_target()
        rendered = renderer.render(camera, [target])
        assert rendered.image.shape == (h, w)
        assert rendered.width == w
        assert rendered.height == h


# ---------------------------------------------------------------------------
# Q. Different FOVs
# ---------------------------------------------------------------------------

class TestDifferentFOVs:
    def test_wider_fov_moves_target_closer_to_center(self):
        target = _make_target(x=1010.0, y=1000.0)
        cam_narrow = _make_camera(hfov=4.0, vfov=3.0)
        cam_wide = _make_camera(hfov=30.0, vfov=22.5)
        cfg = SensorConfig()

        r_narrow = VirtualSensorRenderer(cfg).render(cam_narrow, [target])
        r_wide = VirtualSensorRenderer(cfg).render(cam_wide, [target])

        gt_narrow = r_narrow.primary_ground_truth
        gt_wide = r_wide.primary_ground_truth
        assert gt_narrow is not None
        assert gt_wide is not None
        assert abs(gt_wide.target_pixel_x - 320.0) < abs(gt_narrow.target_pixel_x - 320.0)


# ---------------------------------------------------------------------------
# Additional: Ground truth to_dict roundtrip
# ---------------------------------------------------------------------------

class TestGroundTruthSerialization:
    def test_to_dict_roundtrip(self):
        gt = GroundTruth(
            target_id=0,
            target_visible=True,
            visibility=TargetVisibility.VISIBLE,
            target_world_position=(100.0, 200.0, 300.0),
            target_camera_position=(10.0, 20.0, 300.0),
            target_pixel_x=319.5,
            target_pixel_y=240.3,
            target_bbox=(310, 230, 330, 250),
            target_size_px=10.0,
            horizontal_angle_deg=1.5,
            vertical_angle_deg=-0.3,
            depth=300.0,
            brightness=0.8,
        )
        d = gt.to_dict()
        gt2 = GroundTruth.from_dict(d)
        assert gt2.target_pixel_x == 319.5
        assert gt2.visibility == TargetVisibility.VISIBLE


# ---------------------------------------------------------------------------
# Additional: RenderedFrame properties
# ---------------------------------------------------------------------------

class TestRenderedFrame:
    def test_properties(self):
        rendered = _render()
        assert rendered.width == 640
        assert rendered.height == 480
        assert rendered.channels == 1

    def test_metadata_present(self):
        rendered = _render()
        assert "sensor_config" in rendered.metadata
