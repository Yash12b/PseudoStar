"""Candidate region generation from preprocessed images.

Extracts bright-spot candidate regions using thresholding and
connected component analysis.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.perception.config import PerceptionConfig, ThresholdMode
from fsoc_tracker.perception.models import CandidateFeatures


def generate_candidates(
    image: np.ndarray,
    background_level: float,
    config: PerceptionConfig,
    max_candidates: int | None = None,
) -> list[np.ndarray]:
    """Generate binary candidate masks from the preprocessed image.

    Args:
        image: Preprocessed grayscale image.
        background_level: Estimated background intensity.
        config: Perception configuration.
        max_candidates: Cap on returned regions (largest areas first).
            Defaults to ``config.max_candidates``. Pass 0/None... note
            None means "use config", while a positive int overrides it.

    Returns:
        List of binary masks, one per candidate region.
    """
    if image is None or image.size == 0:
        return []

    img = image.astype(np.float64)
    mask = _threshold(img, background_level, config)

    if mask is None or not np.any(mask):
        return []

    if config.morphology_enabled:
        mask = _clean_mask(mask, config)

    cap = config.max_candidates if max_candidates is None else max_candidates

    # Use the statistics-producing OpenCV call directly. The previous
    # implementation first computed connected components and then computed
    # them again with ``connectedComponentsWithStats`` on every frame.
    try:
        import cv2
        num_labels2, labels2, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas: list[tuple[float, int]] = []
        for label_id in range(1, num_labels2):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if config.min_candidate_area <= area <= config.max_candidate_area:
                areas.append((area, label_id))
        # Largest-first + cap: genuine beacons outrank noise specks, and
        # per-candidate feature extraction stays bounded under floods.
        areas.sort(key=lambda t: t[0], reverse=True)
        candidates = [(labels2 == label_id) for _, label_id in areas[:cap]]
        return candidates
    except ImportError:
        labels, num_labels = _connected_components(mask)
        if num_labels == 0:
            return []
    except cv2.error:
        labels, num_labels = _connected_components(mask)
        if num_labels == 0:
            return []

    candidates = []
    for label_id in range(1, num_labels + 1):
        region_mask = (labels == label_id)
        area = float(np.sum(region_mask))
        if config.min_candidate_area <= area <= config.max_candidate_area:
            candidates.append((area, region_mask))
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in candidates[:cap]]


def _threshold(
    img: np.ndarray,
    background_level: float,
    config: PerceptionConfig,
) -> np.ndarray | None:
    """Apply thresholding to create a binary mask."""
    if config.threshold_mode == ThresholdMode.GLOBAL:
        return (img > config.threshold_value).astype(np.uint8) * 255

    elif config.threshold_mode == ThresholdMode.ADAPTIVE:
        try:
            import cv2
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            block = config.adaptive_block_size
            if block % 2 == 0:
                block += 1
            return cv2.adaptiveThreshold(
                img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, config.adaptive_constant,
            )
        except ImportError:
            return (img > config.threshold_value).astype(np.uint8) * 255

    elif config.threshold_mode == ThresholdMode.PERCENTILE:
        dynamic_range = float(np.max(img) - background_level)
        if dynamic_range < 1.0:
            return None
        threshold = background_level + dynamic_range * (config.percentile_value / 100.0)
        return (img > threshold).astype(np.uint8) * 255

    return None


def _clean_mask(mask: np.ndarray, config: PerceptionConfig) -> np.ndarray:
    """Apply morphological cleaning to the mask."""
    try:
        import cv2
        ksize = config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        pass
    return mask


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Find connected components in a binary mask."""
    try:
        import cv2
        num_labels, labels = cv2.connectedComponents(mask)
        return labels, num_labels
    except ImportError:
        raise RuntimeError(
            "OpenCV is required for connected components "
            "(opencv-python is a mandatory dependency).") from None


def extract_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> CandidateFeatures:
    """Extract features from a candidate region.

    Args:
        image: Grayscale image (uint8 or float).
        mask: Binary mask for this candidate (bool or uint8).

    Returns:
        CandidateFeatures with all computed properties.
    """
    m = mask.astype(bool)
    if not np.any(m):
        return CandidateFeatures()

    ys, xs = np.where(m)
    area = float(len(ys))

    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))

    # Crop to the candidate bbox first: identical arithmetic on the same
    # pixel set, but perimeter/erosion/centroid no longer scan the full
    # frame per candidate (the noise-flood bottleneck). A 1 px margin
    # keeps edge erosion results identical to full-frame computation.
    H, W = image.shape[:2]
    cx1, cy1 = max(0, x_min - 1), max(0, y_min - 1)
    cx2, cy2 = min(W, x_max + 2), min(H, y_max + 2)
    img = image.astype(np.float64)[cy1:cy2, cx1:cx2]
    mc = m[cy1:cy2, cx1:cx2]

    cys, cxs = np.where(mc)
    centroid_x = float(np.mean(cxs)) + cx1
    centroid_y = float(np.mean(cys)) + cy1

    bbox = (x_min, y_min, x_max, y_max)

    width = float(x_max - x_min + 1)
    height = float(y_max - y_min + 1)
    aspect_ratio = width / height if height > 0 else 1.0

    perimeter = _compute_perimeter(mc)

    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0

    candidate_pixels = img[mc]
    mean_intensity = float(np.mean(candidate_pixels)) if len(candidate_pixels) > 0 else 0.0
    max_intensity = float(np.max(candidate_pixels)) if len(candidate_pixels) > 0 else 0.0
    integrated_intensity = float(np.sum(candidate_pixels))

    local_contrast = _compute_local_contrast(image.astype(np.float64), m, bbox)

    return CandidateFeatures(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area=area,
        bbox=bbox,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        perimeter=perimeter,
        circularity=circularity,
        mean_intensity=mean_intensity,
        max_intensity=max_intensity,
        integrated_intensity=integrated_intensity,
        local_contrast=local_contrast,
    )


def _compute_perimeter(mask: np.ndarray) -> float:
    """Compute the perimeter of a binary mask using edge detection."""
    try:
        import cv2
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel)
        edge = mask.astype(np.uint8) - eroded
        return float(np.sum(edge > 0))
    except ImportError:
        m = mask.astype(bool)
        padded = np.pad(m, 1, mode="constant", constant_values=False)
        edge_count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(padded, dy, axis=0), dx, axis=1)[1:-1, 1:-1]
                edge_count += np.sum(m & ~shifted)
        return float(edge_count)


def _compute_local_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 5,
) -> float:
    """Compute local contrast around the candidate region."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w, x2 + margin + 1)
    y2m = min(h, y2 + margin + 1)

    region = image[y1m:y2m, x1m:x2m].astype(np.float64)
    border_mask = np.ones(region.shape, dtype=bool)
    local_h, local_w = y2 - y1 + 1, x2 - x1 + 1
    oy = y1 - y1m
    ox = x1 - x1m
    border_mask[oy:oy + local_h, ox:ox + local_w] = False

    border_pixels = region[border_mask]
    if len(border_pixels) == 0:
        return 0.0

    candidate_pixels = image[mask].astype(np.float64)
    if len(candidate_pixels) == 0:
        return 0.0

    return float(np.mean(candidate_pixels) - np.mean(border_pixels))
