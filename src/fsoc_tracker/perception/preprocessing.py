"""Preprocessing pipeline for perception.

Modular, configurable image preprocessing stages.
Each stage can be enabled/disabled independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsoc_tracker.perception.config import PerceptionConfig


@dataclass
class PreprocessingResult:
    """Result of preprocessing a frame."""

    image: np.ndarray
    background_level: float = 0.0
    noise_estimate: float = 0.0
    contrast: float = 0.0


def estimate_background(image: np.ndarray, method: str = "percentile", percentile: float = 10.0) -> float:
    """Estimate background intensity level."""
    if method == "percentile":
        return float(np.percentile(image, percentile))
    elif method == "mean":
        return float(np.mean(image))
    elif method == "median":
        return float(np.median(image))
    return float(np.percentile(image, percentile))


def estimate_noise(image: np.ndarray, background: float) -> float:
    """Estimate noise level from background region."""
    diff = np.abs(image.astype(np.float64) - background)
    return float(np.median(diff))


def normalize_contrast(image: np.ndarray, background: float, target_range: float = 200.0) -> np.ndarray:
    """Normalize image contrast relative to background, subtracting background."""
    img = image.astype(np.float64)
    max_val = np.max(img)
    if max_val <= background:
        return np.zeros_like(img, dtype=np.uint8)
    dynamic_range = max_val - background
    if dynamic_range < 1.0:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = (img - background) * (target_range / dynamic_range)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur if sigma > 0."""
    if sigma <= 0:
        return image.copy()
    try:
        import cv2
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    except ImportError:
        return image.copy()


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter for noise reduction."""
    try:
        import cv2
        return cv2.medianBlur(image, kernel_size)
    except ImportError:
        return image.copy()


def preprocess_frame(
    image: np.ndarray,
    config: PerceptionConfig,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on a frame.

    Args:
        image: Input image (uint8 grayscale or BGR).
        config: Perception configuration.

    Returns:
        PreprocessingResult with processed image and statistics.
    """
    if image is None or image.size == 0:
        return PreprocessingResult(image=np.zeros((1, 1), dtype=np.uint8))

    img = image.copy()
    if img.ndim == 3:
        try:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ImportError:
            img = img[:, :, 0]

    bg = estimate_background(img, config.background_estimate_method, config.background_percentile)
    noise = estimate_noise(img, bg)
    contrast = float(np.max(img.astype(np.float64)) - bg)

    if config.denoise_enabled and noise > 1.0:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = median_filter(img, 3)

    if config.blur_sigma > 0:
        img = gaussian_blur(img, config.blur_sigma)

    if config.morphology_enabled:
        try:
            import cv2
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (config.morphology_kernel_size, config.morphology_kernel_size),
            )
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        except ImportError:
            pass

    if config.normalize_contrast_enabled:
        img = normalize_contrast(img, bg)
        # Normalization remaps background to ~0, so the pre-normalization
        # background estimate no longer describes the image. Re-estimate
        # on the normalized pixels (threshold math depends on it).
        bg = estimate_background(img, config.background_estimate_method,
                                 config.background_percentile)

    return PreprocessingResult(
        image=img,
        background_level=bg,
        noise_estimate=noise,
        contrast=contrast,
    )
