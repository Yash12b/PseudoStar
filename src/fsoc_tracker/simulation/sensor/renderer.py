"""VirtualSensorRenderer: converts 3D world state into a sensor image.

The renderer obtains projected coordinates from the Stage-3 camera model,
then forms the beacon image using the sensor/optical model.

Architecture:
    Camera geometry (projection)  -->  Sensor image formation

The renderer does NOT duplicate projection math.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import ProjectionResult
from fsoc_tracker.simulation.sensor.beacon import deposit_beacon
from fsoc_tracker.simulation.sensor.config import SensorConfig, SizeMode
from fsoc_tracker.simulation.sensor.models import (
    GroundTruth,
    RenderedFrame,
    TargetVisibility,
)
from fsoc_tracker.simulation.sensor.psf import gaussian_psf_2d
from fsoc_tracker.simulation.target import WorldTargetState


class VirtualSensorRenderer:
    """Renders 3D world targets into a 2D sensor image.

    Usage::

        renderer = VirtualSensorRenderer(sensor_config)
        rendered = renderer.render(camera, world_targets, timestamp_s)
        image = rendered.image  # uint8 grayscale for perception
        gt = rendered.ground_truths  # evaluation metadata only
    """

    def __init__(self, config: SensorConfig | None = None) -> None:
        self._config = config or SensorConfig()
        self._psf_kernel: np.ndarray | None = None
        if self._config.enable_psf and self._config.psf_sigma_px > 0:
            self._psf_kernel = gaussian_psf_2d(self._config.psf_sigma_px)

    @property
    def config(self) -> SensorConfig:
        return self._config

    def render(
        self,
        camera: VirtualCamera,
        targets: list[WorldTargetState],
        timestamp_s: float = 0.0,
        frame_index: int = 0,
    ) -> RenderedFrame:
        """Render all targets into a sensor image.

        Args:
            camera: VirtualCamera with current pose.
            targets: List of world-space targets to render.
            timestamp_s: Current simulation timestamp.
            frame_index: Sequential frame index.

        Returns:
            RenderedFrame with raw sensor image and ground truth metadata.
        """
        cfg = self._config
        image = np.full(
            (cfg.height, cfg.width),
            cfg.background_level,
            dtype=np.float64,
        )

        if cfg.background_noise_level > 0:
            rng = np.random.RandomState(cfg.seed + frame_index)
            noise = rng.normal(0, cfg.background_noise_level, image.shape)
            image = image + noise

        ground_truths: list[GroundTruth] = []

        for target in targets:
            gt = self._render_target(image, camera, target, cfg, frame_index)
            ground_truths.append(gt)

        if cfg.enable_psf and self._psf_kernel is not None:
            image = _apply_psf_localized(image, self._psf_kernel)

        image = np.clip(image, 0, cfg.max_intensity)
        image = image.astype(np.uint8)

        return RenderedFrame(
            image=image,
            ground_truths=ground_truths,
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            metadata={
                "sensor_config": {
                    "width": cfg.width,
                    "height": cfg.height,
                    "channels": cfg.channels,
                    "color_mode": cfg.color_mode.value,
                },
            },
        )

    def _render_target(
        self,
        image: np.ndarray,
        camera: VirtualCamera,
        target: WorldTargetState,
        cfg: SensorConfig,
        frame_index: int = 0,
    ) -> GroundTruth:
        """Render a single target and return its ground truth."""
        world_pos = (target.x, target.y, target.z)
        proj = camera.project_world_point(world_pos)

        # Temporal identity code (ON/OFF keying): "" = steady ON.
        from fsoc_tracker.tracking.identity import code_bit_at
        code_on = code_bit_at(getattr(target, "code", "") or "",
                              frame_index, cfg.code_frames_per_bit)
        eff_brightness = float(target.brightness) * code_on

        gt = GroundTruth(
            target_id=target.target_id,
            target_world_position=world_pos,
            target_camera_position=(proj.camera_x, proj.camera_y, proj.camera_z),
            depth=proj.depth,
            brightness=target.brightness,
        )

        if proj.depth <= 0:
            gt.visibility = TargetVisibility.BEHIND_CAMERA
            gt.target_visible = False
            return gt

        gt.target_pixel_x = proj.pixel_x
        gt.target_pixel_y = proj.pixel_y
        gt.horizontal_angle_deg = proj.horizontal_angle_deg
        gt.vertical_angle_deg = proj.vertical_angle_deg

        size_px = self._compute_apparent_size(cfg, proj, target)

        if not proj.visible:
            overlap = self._check_partial_overlap(proj.pixel_x, proj.pixel_y, size_px, cfg)
            if overlap:
                gt.visibility = TargetVisibility.PARTIAL
                gt.target_visible = True
                gt.target_size_px = size_px
                gt.target_bbox = self._compute_bbox(proj.pixel_x, proj.pixel_y, size_px)
                if eff_brightness > 0:
                    deposit_beacon(image, proj.pixel_x, proj.pixel_y, size_px, eff_brightness * cfg.beacon_peak_intensity, cfg,
                                   shape_override=target.shape)
            else:
                gt.visibility = TargetVisibility.OUTSIDE
                gt.target_visible = False
        else:
            gt.visibility = TargetVisibility.VISIBLE
            gt.target_visible = True
            gt.target_size_px = size_px
            gt.target_bbox = self._compute_bbox(proj.pixel_x, proj.pixel_y, size_px)
            if eff_brightness > 0:
                deposit_beacon(image, proj.pixel_x, proj.pixel_y, size_px, eff_brightness * cfg.beacon_peak_intensity, cfg,
                               shape_override=target.shape)

        return gt

    def _compute_apparent_size(
        self,
        cfg: SensorConfig,
        proj: ProjectionResult,
        target: WorldTargetState,
    ) -> float:
        """Compute apparent beacon size in pixels.

        MODE A (FIXED): Returns configured default size.
        MODE B (DISTANCE_BASED): Interface for future physical model.
        """
        if cfg.size_mode == SizeMode.FIXED:
            if target.size_px is not None and target.size_px > 0:
                return cfg.clamped_beacon_size(target.size_px)
            return cfg.clamped_beacon_size(cfg.beacon_default_size_px)

        angular_size_rad = np.arctan2(target.width, proj.depth)
        focal_length_px = (proj.camera_x / proj.pixel_x if proj.pixel_x != 0 else cfg.width / 2.0)
        apparent = 2.0 * focal_length_px * np.tan(angular_size_rad / 2.0)
        return cfg.clamped_beacon_size(apparent)

    def _check_partial_overlap(
        self,
        cx: float,
        cy: float,
        size_px: float,
        cfg: SensorConfig,
    ) -> bool:
        """Check if a partially-outside target overlaps the image."""
        half = size_px / 2.0
        x1 = cx - half
        x2 = cx + half
        y1 = cy - half
        y2 = cy + half
        return x2 > 0 and x1 < cfg.width and y2 > 0 and y1 < cfg.height

    def _compute_bbox(
        self,
        cx: float,
        cy: float,
        size_px: float,
    ) -> tuple[int, int, int, int]:
        """Compute integer bounding box (x1, y1, x2, y2) clamped to image."""
        half = size_px / 2.0
        x1 = int(np.floor(cx - half))
        y1 = int(np.floor(cy - half))
        x2 = int(np.ceil(cx + half))
        y2 = int(np.ceil(cy + half))
        return (x1, y1, x2, y2)


def render_debug_view(
    rendered: RenderedFrame,
    show_center: bool = True,
    show_bbox: bool = True,
    show_crosshair: bool = True,
) -> np.ndarray:
    """Create a debug visualization copy (never contaminates raw image).

    Args:
        rendered: The RenderedFrame to visualize.
        show_center: Draw target center marker.
        show_bbox: Draw bounding box.
        show_crosshair: Draw image center crosshair.

    Returns:
        BGR numpy array for OpenCV display.
    """
    img = rendered.image
    if img.ndim == 2:
        vis = np.stack([img, img, img], axis=-1)
    else:
        vis = img.copy()

    vis = vis.astype(np.uint8)
    h, w = vis.shape[:2]

    if show_crosshair:
        cx, cy = w // 2, h // 2
        vis[max(0, cy - 5):min(h, cy + 5), max(0, cx - 1):min(w, cx + 1)] = [0, 255, 0]
        vis[max(0, cy - 1):min(h, cy + 1), max(0, cx - 5):min(w, cx + 5)] = [0, 255, 0]

    for gt in rendered.ground_truths:
        if not gt.target_visible:
            continue

        px = int(round(gt.target_pixel_x))
        py = int(round(gt.target_pixel_y))

        if show_center:
            r = 3
            vis[max(0, py - r):min(h, py + r + 1), max(0, px - r):min(w, px + r + 1)] = [0, 0, 255]

        if show_bbox:
            x1, y1, x2, y2 = gt.target_bbox
            x1c = max(0, min(w - 1, x1))
            x2c = max(0, min(w, x2))
            y1c = max(0, min(h - 1, y1))
            y2c = max(0, min(h, y2))
            vis[y1c, x1c:x2c] = [255, 255, 0]
            vis[y2c - 1, x1c:x2c] = [255, 255, 0]
            vis[y1c:y2c, x1c] = [255, 255, 0]
            vis[y1c:y2c, x2c - 1] = [255, 255, 0]

    return vis


def _apply_psf_localized(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply PSF only to pixels above background level (efficient)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    mask = image > 0.1
    if not np.any(mask):
        return image

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    result = image.copy()

    ys, xs = np.where(mask)
    for py, px in zip(ys, xs, strict=True):
        patch = padded[py:py + kh, px:px + kw]
        result[py, px] = np.sum(patch * kernel)

    return result
