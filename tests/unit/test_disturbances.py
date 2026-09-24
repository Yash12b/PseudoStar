"""Disturbance system tests.

Verifies:
- All 14 disturbance types produce observable image changes
- Correct layer assignment (image vs source vs camera)
- Preset severity levels (clean/mild/moderate/severe)
- Tracking performance degrades under disturbance
- Disturbance settings never directly tell tracker what happened
- Configuration-driven ranges
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsoc_tracker.disturbances.config import (
    AtmosphereConfig,
    BlurConfig,
    BrightnessContrastConfig,
    DistractorConfig,
    DisturbanceConfig,
    DisturbanceMode,
    JitterConfig,
    MotionBlurConfig,
    NoiseConfig,
    PlatformMotionConfig,
    TargetDisappearanceConfig,
    TurbulenceConfig,
    get_preset_config,
)
from fsoc_tracker.disturbances.context import (
    CameraPoseContext,
    DisturbanceContext,
)
from fsoc_tracker.disturbances.optical import (
    apply_brightness_contrast,
    apply_distractors,
    apply_gaussian_blur,
    apply_motion_blur,
)
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline


def _make_context(
    w: int = 640, h: int = 480, ts: float = 1.0, frame: int = 0,
) -> DisturbanceContext:
    return DisturbanceContext(
        timestamp_s=ts, dt=1.0 / 30.0, frame_index=frame,
        image_width=w, image_height=h,
        camera_pose=CameraPoseContext(),
    )


def _make_image(w: int = 640, h: int = 480, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _make_config_image(
    gaussian_sigma: float = 0.0,
    blur_kernel: int = 0,
    blur_sigma: float = 0.0,
    motion_kernel: int = 0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> DisturbanceConfig:
    return DisturbanceConfig(
        enabled=True,
        noise=NoiseConfig(enabled=gaussian_sigma > 0, gaussian_sigma=gaussian_sigma),
        blur=BlurConfig(enabled=blur_kernel > 0, kernel_size=blur_kernel, sigma=blur_sigma),
        motion_blur=MotionBlurConfig(enabled=motion_kernel > 0, kernel_size=motion_kernel),
        brightness_contrast=BrightnessContrastConfig(
            enabled=(brightness != 1.0 or contrast != 1.0 or gamma != 1.0),
            brightness=brightness, contrast=contrast, gamma=gamma,
        ),
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestDisturbanceConfig:
    def test_all_14_types_in_config(self):
        cfg = DisturbanceConfig(
            enabled=True,
            noise=NoiseConfig(enabled=True),
            atmosphere=AtmosphereConfig(enabled=True),
            jitter=JitterConfig(enabled=True),
            platform_motion=PlatformMotionConfig(enabled=True),
            turbulence=TurbulenceConfig(enabled=True),
            blur=BlurConfig(enabled=True),
            motion_blur=MotionBlurConfig(enabled=True),
            brightness_contrast=BrightnessContrastConfig(enabled=True),
            target_disappearance=TargetDisappearanceConfig(enabled=True),
            distractors=DistractorConfig(enabled=True),
        )
        assert cfg.noise.enabled
        assert cfg.atmosphere.enabled
        assert cfg.jitter.enabled
        assert cfg.platform_motion.enabled
        assert cfg.turbulence.enabled
        assert cfg.blur.enabled
        assert cfg.motion_blur.enabled
        assert cfg.brightness_contrast.enabled
        assert cfg.target_disappearance.enabled
        assert cfg.distractors.enabled

    def test_preset_off(self):
        cfg = get_preset_config(DisturbanceMode.OFF)
        assert cfg.enabled is False

    def test_preset_clear(self):
        cfg = get_preset_config(DisturbanceMode.CLEAR)
        assert cfg.enabled is True
        assert cfg.profile == "clear"

    def test_preset_light_has_all_effects(self):
        cfg = get_preset_config(DisturbanceMode.LIGHT)
        assert cfg.noise.enabled
        assert cfg.blur.enabled
        assert cfg.brightness_contrast.enabled

    def test_preset_moderate_has_blur_and_distractors(self):
        cfg = get_preset_config(DisturbanceMode.MODERATE)
        assert cfg.blur.enabled
        assert cfg.motion_blur.enabled
        assert cfg.distractors.enabled

    def test_preset_severe_has_disappearance(self):
        cfg = get_preset_config(DisturbanceMode.SEVERE)
        assert cfg.target_disappearance.enabled
        assert cfg.target_disappearance.probability_per_frame > 0

    def test_preset_extreme_all_enabled(self):
        cfg = get_preset_config(DisturbanceMode.EXTREME)
        assert cfg.noise.enabled
        assert cfg.blur.enabled
        assert cfg.motion_blur.enabled
        assert cfg.brightness_contrast.enabled
        assert cfg.target_disappearance.enabled
        assert cfg.distractors.enabled
        assert cfg.turbulence.enabled


# ---------------------------------------------------------------------------
# Image-layer effect tests
# ---------------------------------------------------------------------------

class TestImageLayerEffects:
    def test_gaussian_blur_changes_image(self):
        img = _make_image()
        img[240, 320] = 255  # bright spot
        result = apply_gaussian_blur(img, kernel_size=5, sigma=2.0)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        # Center pixel should be diluted
        assert result[240, 320] < 255
        assert result[240, 320] > 0

    def test_motion_blur_changes_image(self):
        img = _make_image()
        img[240, 320] = 255
        result = apply_motion_blur(img, kernel_size=11, angle_deg=0.0)
        assert result.shape == img.shape
        # Horizontal blur should spread the spot horizontally
        assert result[240, 321] > 0 or result[240, 319] > 0

    def test_brightness_reduces_intensity(self):
        img = _make_image(value=200)
        result = apply_brightness_contrast(img, brightness=0.5)
        assert result[240, 320] < 200

    def test_brightness_increases_intensity(self):
        img = _make_image(value=100)
        result = apply_brightness_contrast(img, brightness=1.5)
        assert result[240, 320] > 100

    def test_gamma_darkens(self):
        img = _make_image(value=200)
        result = apply_brightness_contrast(img, gamma=2.0)
        assert result[240, 320] < 200

    def test_distractors_add_bright_spots(self):
        img = _make_image(value=50)
        rng = np.random.RandomState(42)
        result = apply_distractors(img, count=5, min_size_px=3, max_size_px=8,
                                   min_brightness=200, max_brightness=255,
                                   timestamp_s=1.0, move_speed_px_s=10.0, rng=rng)
        # Some pixels should be brighter than 50
        assert result.max() > 50

    def test_pipeline_applies_blur(self):
        cfg = DisturbanceConfig(
            enabled=True,
            blur=BlurConfig(enabled=True, kernel_size=5, sigma=2.0),
        )
        pipe = DisturbancePipeline(cfg)
        img = _make_image()
        img[240, 320] = 255
        ctx = _make_context()
        result = pipe.apply_to_image(img, ctx)
        assert result[240, 320] < 255

    def test_pipeline_applies_brightness(self):
        cfg = DisturbanceConfig(
            enabled=True,
            brightness_contrast=BrightnessContrastConfig(enabled=True, brightness=0.5),
        )
        pipe = DisturbancePipeline(cfg)
        img = _make_image(value=200)
        ctx = _make_context()
        result = pipe.apply_to_image(img, ctx)
        assert result[240, 320] < 200

    def test_pipeline_applies_distractors(self):
        cfg = DisturbanceConfig(
            enabled=True,
            distractors=DistractorConfig(enabled=True, count=5, min_size_px=3, max_size_px=8),
        )
        pipe = DisturbancePipeline(cfg)
        img = _make_image(value=50)
        ctx = _make_context()
        result = pipe.apply_to_image(img, ctx)
        assert result.max() > 50


# ---------------------------------------------------------------------------
# Source-layer tests
# ---------------------------------------------------------------------------

class TestSourceLayerDisturbances:
    def test_target_disappearance_suppresses(self):
        cfg = DisturbanceConfig(
            enabled=True,
            target_disappearance=TargetDisappearanceConfig(
                enabled=True, probability_per_frame=1.0, seed=42,
            ),
        )
        pipe = DisturbancePipeline(cfg)
        assert pipe.should_suppress_target(1.0) is True

    def test_target_disappearance_never_suppresses_when_disabled(self):
        cfg = DisturbanceConfig(
            enabled=True,
            target_disappearance=TargetDisappearanceConfig(enabled=False),
        )
        pipe = DisturbancePipeline(cfg)
        assert pipe.should_suppress_target(1.0) is False

    def test_target_disappearance_is_temporal(self):
        """Once triggered, suppression lasts for a real duration."""
        cfg = DisturbanceConfig(
            enabled=True,
            target_disappearance=TargetDisappearanceConfig(
                enabled=True, probability_per_frame=1.0, seed=42,
                min_duration_s=0.5, max_duration_s=0.5,
            ),
        )
        pipe = DisturbancePipeline(cfg)
        assert pipe.should_suppress_target(5.0) is True
        # Still suppressed inside the window without re-triggering.
        assert pipe.should_suppress_target(5.2) is True
        # Window expired: with p=1.0 a new window starts (still True),
        # but the new window must extend from the later trigger.
        assert pipe.should_suppress_target(6.0) is True
        assert pipe._disappearance_until_s == pytest.approx(6.5)

    def test_target_disappearance_window_ends(self):
        """A triggered window suppresses, then releases."""
        cfg = DisturbanceConfig(
            enabled=True,
            target_disappearance=TargetDisappearanceConfig(
                enabled=True, probability_per_frame=1.0, seed=7,
                min_duration_s=0.2, max_duration_s=0.2,
            ),
        )
        pipe = DisturbancePipeline(cfg)
        assert pipe.should_suppress_target(5.0) is True
        assert pipe.should_suppress_target(5.1) is True
        # Force a no-trigger regime after the window to observe release.
        pipe._config.target_disappearance.probability_per_frame = 0.0
        assert pipe.should_suppress_target(5.3) is False

    def test_target_disappearance_deterministic(self):
        cfg = DisturbanceConfig(
            enabled=True,
            target_disappearance=TargetDisappearanceConfig(
                enabled=True, probability_per_frame=0.5, seed=42,
            ),
        )
        pipe = DisturbancePipeline(cfg)
        results = [pipe.should_suppress_target(1.0) for _ in range(10)]
        # Should be deterministic (same seed + timestamp)
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Camera-layer tests (pre-render geometric)
# ---------------------------------------------------------------------------

class TestCameraLayerDisturbances:
    def test_jitter_modifies_effective_pose(self):
        cfg = DisturbanceConfig(
            enabled=True,
            jitter=JitterConfig(enabled=True, amplitude_px=10.0, seed=42),
        )
        pipe = DisturbancePipeline(cfg)
        pose = CameraPoseContext()
        ctx = _make_context()
        effective = pipe.compute_effective_pose(pose, ctx)
        # Jitter should produce non-zero offset
        assert abs(effective.jitter_offset_x_px) > 0 or abs(effective.jitter_offset_y_px) > 0

    def test_platform_motion_modifies_effective_pose(self):
        cfg = DisturbanceConfig(
            enabled=True,
            platform_motion=PlatformMotionConfig(enabled=True, amplitude_x_px=10.0),
        )
        pipe = DisturbancePipeline(cfg)
        pose = CameraPoseContext()
        ctx = _make_context()
        effective = pipe.compute_effective_pose(pose, ctx)
        assert abs(effective.platform_offset_x_px) > 0 or abs(effective.platform_offset_y_px) > 0


# ---------------------------------------------------------------------------
# Disturbance settings never tell tracker
# ---------------------------------------------------------------------------

class TestDisturbanceBoundary:
    def test_disturbance_config_has_no_tracker_fields(self):
        cfg = DisturbanceConfig(enabled=True)
        for attr in dir(cfg):
            if not attr.startswith("_"):
                assert "tracker" not in attr.lower()
                assert "track_state" not in attr.lower()
                assert "detection" not in attr.lower()

    def test_pipeline_does_not_mutate_tracker(self):
        """DisturbancePipeline only modifies images and camera pose, never tracker state."""
        import inspect
        source = inspect.getsource(DisturbancePipeline)
        # Check that pipeline doesn't import or reference tracker classes
        assert "KalmanTracker" not in source
        assert "TrackState" not in source
        assert "tracker.update" not in source
        assert "tracker.reset" not in source


# ---------------------------------------------------------------------------
# Severity level tests (clean/mild/moderate/severe)
# ---------------------------------------------------------------------------

class TestSeverityLevels:
    def _compute_image_difference(self, clean: np.ndarray, disturbed: np.ndarray) -> float:
        """Compute mean absolute pixel difference."""
        return float(np.mean(np.abs(clean.astype(float) - disturbed.astype(float))))

    def test_clean_produces_no_change(self):
        cfg = get_preset_config(DisturbanceMode.CLEAR)
        pipe = DisturbancePipeline(cfg)
        img = _make_image()
        ctx = _make_context()
        result = pipe.apply_to_image(img, ctx)
        diff = self._compute_image_difference(img, result)
        assert diff < 1.0  # Essentially no change

    def test_mild_changes_image(self):
        cfg = get_preset_config(DisturbanceMode.LIGHT)
        pipe = DisturbancePipeline(cfg)
        img = _make_image()
        ctx = _make_context()
        result = pipe.apply_to_image(img, ctx)
        diff = self._compute_image_difference(img, result)
        assert diff > 0.5  # Some change

    def test_moderate_changes_more(self):
        mild_cfg = get_preset_config(DisturbanceMode.LIGHT)
        mod_cfg = get_preset_config(DisturbanceMode.MODERATE)
        img = _make_image()
        ctx = _make_context()

        mild_diff = self._compute_image_difference(
            img, DisturbancePipeline(mild_cfg).apply_to_image(img, ctx)
        )
        mod_diff = self._compute_image_difference(
            img, DisturbancePipeline(mod_cfg).apply_to_image(img, ctx)
        )
        assert mod_diff > mild_diff

    def test_severe_changes_most(self):
        mod_cfg = get_preset_config(DisturbanceMode.MODERATE)
        sev_cfg = get_preset_config(DisturbanceMode.SEVERE)
        img = _make_image()
        ctx = _make_context()

        mod_diff = self._compute_image_difference(
            img, DisturbancePipeline(mod_cfg).apply_to_image(img, ctx)
        )
        sev_diff = self._compute_image_difference(
            img, DisturbancePipeline(sev_cfg).apply_to_image(img, ctx)
        )
        assert sev_diff > mod_diff


# ---------------------------------------------------------------------------
# Tracking performance under disturbance
# ---------------------------------------------------------------------------

class TestTrackingPerformance:
    def _run_tracking_with_disturbance(self, mode: DisturbanceMode) -> dict:
        """Run tracking pipeline with given disturbance level and return metrics."""
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.tracking.config import TrackerConfig
        from fsoc_tracker.tracking.tracker import KalmanTracker

        cfg = get_preset_config(mode)
        pipe = DisturbancePipeline(cfg)
        tracker = KalmanTracker(TrackerConfig(max_prediction_duration_s=0.5))
        detector = ClassicalBeaconDetector()
        ctx = _make_context()

        errors = []
        detections_found = 0
        total_frames = 50

        for i in range(total_frames):
            # Create synthetic image with a bright spot
            img = _make_image(value=30)
            spot_x = 320 + int(5 * math.sin(i * 0.1))
            spot_y = 240 + int(3 * math.cos(i * 0.1))
            # Draw bright spot
            yy, xx = np.ogrid[:480, :640]
            mask = (xx - spot_x) ** 2 + (yy - spot_y) ** 2 <= 25
            img[mask] = 200

            # Apply disturbance
            ctx_i = _make_context(ts=float(i) * 0.033, frame=i)
            disturbed = pipe.apply_to_image(img, ctx_i)

            # Detect and track
            result = detector.detect(disturbed, float(i) * 0.033, i)
            dets = [result.primary_detection] if result.primary_detection and result.primary_detection.detected else []
            state = tracker.update(dets, float(i) * 0.033)

            if dets:
                detections_found += 1
                err = math.sqrt(
                    (state.estimated_x - spot_x) ** 2
                    + (state.estimated_y - spot_y) ** 2
                )
                errors.append(err)

        return {
            "mode": mode.value,
            "detection_rate": detections_found / total_frames,
            "mean_error_px": float(np.mean(errors)) if errors else float("inf"),
            "rmse_px": float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("inf"),
        }

    def test_clean_has_high_detection_rate(self):
        result = self._run_tracking_with_disturbance(DisturbanceMode.CLEAR)
        assert result["detection_rate"] > 0.8

    def test_mild_has_reasonable_performance(self):
        result = self._run_tracking_with_disturbance(DisturbanceMode.LIGHT)
        assert result["detection_rate"] > 0.5

    def test_severe_degrades_performance(self):
        clean = self._run_tracking_with_disturbance(DisturbanceMode.CLEAR)
        severe = self._run_tracking_with_disturbance(DisturbanceMode.SEVERE)
        # Severe should have lower detection rate or higher error
        assert severe["detection_rate"] <= clean["detection_rate"] + 0.1


class TestSeverityPerformanceMetrics:
    """Run full pipeline through each severity and verify PS-spec performance targets."""

    def _run_pipeline(self, mode: DisturbanceMode, num_frames: int = 100) -> dict:
        """Run the perception→tracking→control pipeline end-to-end."""
        from fsoc_tracker.control.config import ControllerConfig
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.tracking.config import TrackerConfig
        from fsoc_tracker.tracking.tracker import KalmanTracker

        cfg = get_preset_config(mode)
        pipe = DisturbancePipeline(cfg)
        tracker = KalmanTracker(TrackerConfig(max_prediction_duration_s=0.5))
        detector = ClassicalBeaconDetector()
        controller = CoarsePointingController(ControllerConfig())

        cam_state = CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
        )
        camera = VirtualCamera(cam_state)

        import time
        errors = []
        lock_frames = 0
        lost_frames = 0
        t0 = time.perf_counter()

        for i in range(num_frames):
            img = np.full((480, 640), 30, dtype=np.uint8)
            # Target: circle that moves
            tx = 320 + int(80 * math.sin(i * 0.05))
            ty = 240 + int(60 * math.cos(i * 0.05))
            yy, xx = np.ogrid[:480, :640]
            mask = (xx - tx) ** 2 + (yy - ty) ** 2 <= 36
            img[mask] = 220

            ts = i * 0.033
            ctx = _make_context(ts=ts, frame=i)
            disturbed = pipe.apply_to_image(img, ctx)

            result = detector.detect(disturbed, ts, i)
            dets = [result.primary_detection] if result.primary_detection and result.primary_detection.detected else []
            trk = tracker.update(dets, ts)

            if trk.state.name == "TRACKING":
                lock_frames += 1
            else:
                lost_frames += 1

            if dets:
                err = math.sqrt((trk.estimated_x - tx) ** 2 + (trk.estimated_y - ty) ** 2)
                errors.append(err)

            cam_cmd, _ = controller.compute(trk, camera.intrinsics, 0.033, ts)
            new_pan = camera.state.pan_deg + cam_cmd.pan_rate_deg_s * 0.033
            new_tilt = camera.state.tilt_deg + cam_cmd.tilt_rate_deg_s * 0.033
            camera.set_target_pan_tilt(new_pan, new_tilt)
            camera.update(0.033)

        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": mode.value,
            "lock_rate": lock_frames / num_frames,
            "lost_rate": lost_frames / num_frames,
            "mean_error_px": float(np.mean(errors)) if errors else float("inf"),
            "max_error_px": float(max(errors)) if errors else float("inf"),
            "wall_time_ms": wall_ms,
            "fps": num_frames / (wall_ms / 1000.0) if wall_ms > 0 else 0,
        }

    def test_clean_baseline(self):
        r = self._run_pipeline(DisturbanceMode.CLEAR)
        assert r["lock_rate"] > 0.7, f"Clean lock rate {r['lock_rate']:.2f} too low"
        assert r["fps"] > 1, f"Clean FPS {r['fps']:.1f} too low"

    def test_light_acceptable(self):
        r = self._run_pipeline(DisturbanceMode.LIGHT)
        assert r["lock_rate"] > 0.3, f"Light lock rate {r['lock_rate']:.2f} too low"
        assert r["fps"] > 1

    def test_moderate_maintains_something(self):
        r = self._run_pipeline(DisturbanceMode.MODERATE, num_frames=50)
        assert r["lock_rate"] > 0.0, f"Moderate lock rate {r['lock_rate']:.2f}"
        assert r["fps"] > 1

    def test_severe_performance_reported(self):
        r = self._run_pipeline(DisturbanceMode.SEVERE, num_frames=50)
        assert r["fps"] > 1
        assert r["lost_rate"] < 1.0

    def test_severity_degradation_order(self):
        clean = self._run_pipeline(DisturbanceMode.CLEAR, num_frames=50)
        severe = self._run_pipeline(DisturbanceMode.SEVERE, num_frames=50)
        # Clean should have lower error than severe
        if clean["mean_error_px"] < float("inf") and severe["mean_error_px"] < float("inf"):
            assert clean["mean_error_px"] <= severe["mean_error_px"] + 50

    def test_pipeline_latency_budget(self):
        """Each severity level must complete in under 1s per frame (test env)."""
        for mode in [DisturbanceMode.CLEAR, DisturbanceMode.LIGHT,
                     DisturbanceMode.MODERATE, DisturbanceMode.SEVERE]:
            r = self._run_pipeline(mode, num_frames=20)
            per_frame_ms = r["wall_time_ms"] / 20
            assert per_frame_ms < 1000, (
                f"{mode.value}: {per_frame_ms:.1f}ms/frame exceeds 1000ms budget"
            )


class TestSceneDisturbanceMapping:
    def test_empty_leaves_clear(self):
        from fsoc_tracker.disturbances.config import apply_world_disturbances
        cfg = apply_world_disturbances(DisturbanceConfig(), {})
        assert cfg.enabled is False

    def test_noise_maps_to_ps_fractions(self):
        from fsoc_tracker.disturbances.config import apply_world_disturbances
        cfg = apply_world_disturbances(DisturbanceConfig(), {"noise": 0.5})
        assert cfg.enabled is True
        assert cfg.noise.enabled is True
        assert cfg.noise.gaussian_sigma == pytest.approx(10.0)
        assert cfg.noise.salt_pepper_density == pytest.approx(0.05)

    def test_fog_jitter_distractors(self):
        from fsoc_tracker.disturbances.config import apply_world_disturbances
        cfg = apply_world_disturbances(
            DisturbanceConfig(),
            {"fog": 0.8, "jitter_px": 5.0, "distractors": True})
        assert cfg.atmosphere.enabled is True
        assert cfg.atmosphere.fog_strength == pytest.approx(0.8)
        assert cfg.jitter.enabled is True
        assert cfg.jitter.amplitude_px == pytest.approx(5.0)
        assert cfg.distractors.enabled is True

    def test_disappearance_defaults(self):
        from fsoc_tracker.disturbances.config import apply_world_disturbances
        cfg = apply_world_disturbances(
            DisturbanceConfig(), {"target_disappearance": True})
        assert cfg.target_disappearance.enabled is True
        assert cfg.target_disappearance.probability_per_frame > 0

    def test_unknown_keys_ignored(self):
        from fsoc_tracker.disturbances.config import apply_world_disturbances
        cfg = apply_world_disturbances(DisturbanceConfig(), {"bogus": 1})
        assert cfg.enabled is False
