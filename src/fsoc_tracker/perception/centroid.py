"""Subpixel centroid estimation.

Provides geometric and intensity-weighted centroid methods
with robust fallback for edge cases.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.perception.config import CentroidMethod


def compute_centroid(
    image: np.ndarray,
    mask: np.ndarray,
    method: CentroidMethod = CentroidMethod.INTENSITY_WEIGHTED,
) -> tuple[float, float]:
    """Compute the centroid of a candidate region.

    Args:
        image: Grayscale image.
        mask: Binary mask for the candidate.
        method: Centroid estimation method.

    Returns:
        (centroid_x, centroid_y) in pixel coordinates.
    """
    if mask is None or not np.any(mask):
        return (0.0, 0.0)

    # Crop to the mask bbox (+1 px margin): identical arithmetic on the
    # same pixel set, without full-frame copies per candidate.
    m_full = mask.astype(bool)
    ys0, xs0 = np.where(m_full)
    H, W = image.shape[:2]
    cx1 = max(0, int(np.min(xs0)) - 1)
    cy1 = max(0, int(np.min(ys0)) - 1)
    cx2 = min(W, int(np.max(xs0)) + 2)
    cy2 = min(H, int(np.max(ys0)) + 2)

    img = image.astype(np.float64)[cy1:cy2, cx1:cx2]
    m = m_full[cy1:cy2, cx1:cx2]

    if method == CentroidMethod.INTENSITY_WEIGHTED:
        result = _weighted_centroid(img, m)
        if result is not None:
            return (result[0] + cx1, result[1] + cy1)

    gx, gy = _geometric_centroid(m)
    return (gx + cx1, gy + cy1)


def _weighted_centroid(image: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    """Compute intensity-weighted centroid.

    x_c = sum(w_i * x_i) / sum(w_i)
    y_c = sum(w_i * y_i) / sum(w_i)

    Falls back to None if weights sum to zero or invalid.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    weights = image[mask]

    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    weights = np.maximum(weights, 0.0)
    total_weight = np.sum(weights)

    if total_weight <= 0 or not np.isfinite(total_weight):
        return None

    cx = float(np.sum(xs * weights) / total_weight)
    cy = float(np.sum(ys * weights) / total_weight)

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    return (cx, cy)


def _geometric_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute simple geometric centroid (mean of pixel coordinates)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))
