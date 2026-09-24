"""Beacon image formation.

Renders a bright localized spot onto a sensor image.
Supports square, circular, hard-edged, and soft-edged (Gaussian) beacons
with sub-pixel accuracy via anti-aliased deposition.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.simulation.sensor.config import BeaconShape, SensorConfig


def deposit_beacon(
    image: np.ndarray,
    cx: float,
    cy: float,
    size_px: float,
    peak_intensity: float,
    config: SensorConfig,
    shape_override: str | None = None,
) -> None:
    """Deposit a beacon spot onto the image at sub-pixel location.

    Modifies image in-place. Uses anti-aliased deposition when
    enable_anti_aliasing is True and soft edges are enabled.

    Args:
        image: 2D float64 image array (height, width).
        cx: Sub-pixel X coordinate of beacon center.
        cy: Sub-pixel Y coordinate of beacon center.
        size_px: Apparent diameter/width of the beacon in pixels.
        peak_intensity: Peak pixel value at beacon center.
        config: Sensor configuration.
        shape_override: Per-target shape ("square", "circular", "spot" /
            "gaussian"). None keeps the config default. "spot" forces a
            soft Gaussian regardless of the soft-edges flag; the other
            values respect it. With default config (soft edges on) all
            shapes render identically to the previous behavior.
    """
    if size_px <= 0:
        return

    half = size_px / 2.0

    shape = (shape_override or config.beacon_shape.value
             if isinstance(config.beacon_shape, BeaconShape)
             else (shape_override or config.beacon_shape))
    shape = str(shape).lower()
    if shape in ("spot", "gaussian") or config.beacon_soft_edges:
        _deposit_soft_beacon(image, cx, cy, half, peak_intensity, config)
    elif shape == BeaconShape.CIRCULAR.value:
        _deposit_hard_circular(image, cx, cy, half, peak_intensity)
    else:
        _deposit_hard_square(image, cx, cy, half, peak_intensity)


def _deposit_hard_square(
    image: np.ndarray,
    cx: float,
    cy: float,
    half: float,
    peak_intensity: float,
) -> None:
    """Deposit a hard-edged square beacon with anti-aliased boundaries."""
    h, w = image.shape[:2]
    x1 = max(0, int(np.floor(cx - half)))
    x2 = min(w, int(np.ceil(cx + half)))
    y1 = max(0, int(np.floor(cy - half)))
    y2 = min(h, int(np.ceil(cy + half)))

    image[y1:y2, x1:x2] = np.maximum(image[y1:y2, x1:x2], peak_intensity)


def _deposit_hard_circular(
    image: np.ndarray,
    cx: float,
    cy: float,
    half: float,
    peak_intensity: float,
) -> None:
    """Deposit a hard-edged circular beacon."""
    h, w = image.shape[:2]
    x1 = max(0, int(np.floor(cx - half)))
    x2 = min(w, int(np.ceil(cx + half)) + 1)
    y1 = max(0, int(np.floor(cy - half)))
    y2 = min(h, int(np.ceil(cy + half)) + 1)

    for py in range(y1, y2):
        for px in range(x1, x2):
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist <= half:
                image[py, px] = max(image[py, px], peak_intensity)


def _deposit_soft_beacon(
    image: np.ndarray,
    cx: float,
    cy: float,
    half: float,
    peak_intensity: float,
    config: SensorConfig,
) -> None:
    """Deposit a Gaussian-softened beacon spot.

    The Gaussian sigma is derived from the beacon size so that
    the full-width at half-maximum (FWHM) approximately matches
    the requested apparent size.  FWHM = 2 * sqrt(2 * ln(2)) * sigma.
    """
    if half <= 0:
        return

    sigma = half / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    truncate = 3.0
    radius = int(np.ceil(truncate * sigma))
    h, w = image.shape[:2]

    x1 = max(0, int(np.floor(cx - radius)))
    x2 = min(w, int(np.ceil(cx + radius)) + 1)
    y1 = max(0, int(np.floor(cy - radius)))
    y2 = min(h, int(np.ceil(cy + radius)) + 1)

    for py in range(y1, y2):
        for px in range(x1, x2):
            dist_sq = (px - cx) ** 2 + (py - cy) ** 2
            intensity = peak_intensity * np.exp(-dist_sq / (2.0 * sigma**2))
            if intensity > 0.5:
                image[py, px] = max(image[py, px], intensity)
