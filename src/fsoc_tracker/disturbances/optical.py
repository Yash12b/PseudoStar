"""Image-layer optical disturbances: blur, motion blur, brightness/contrast.

These effects degrade the rendered image post-render, simulating
optical and sensor degradation in the camera pipeline.
"""

from __future__ import annotations

import math

import numpy as np


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to image.

    Args:
        image: (H, W) or (H, W, 3) uint8 image
        kernel_size: kernel size (forced to odd)
        sigma: Gaussian standard deviation

    Returns:
        Blurred image (same shape, same dtype)
    """
    if sigma <= 0 or kernel_size <= 1:
        return image

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    half = k // 2

    # Build 1D Gaussian kernel
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel_1d = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()

    # Separate 2D convolution: horizontal then vertical
    if image.ndim == 3:
        # Apply to each channel
        result = np.empty_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            channel = image[:, :, c].astype(np.float64)
            # Horizontal pass
            tmp = np.apply_along_axis(
                lambda row: np.convolve(row, kernel_1d, mode='same'), 1, channel
            )
            # Vertical pass
            tmp = np.apply_along_axis(
                lambda col: np.convolve(col, kernel_1d, mode='same'), 0, tmp
            )
            result[:, :, c] = tmp
    else:
        channel = image.astype(np.float64)
        tmp = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_1d, mode='same'), 1, channel
        )
        result = np.apply_along_axis(
            lambda col: np.convolve(col, kernel_1d, mode='same'), 0, tmp
        )

    return np.clip(result, 0, 255).astype(image.dtype)


def apply_motion_blur(image: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
    """Apply directional motion blur to image.

    Args:
        image: (H, W) or (H, W, 3) uint8 image
        kernel_size: blur kernel length (forced to odd)
        angle_deg: direction angle in degrees (0=horizontal, 90=vertical)

    Returns:
        Motion-blurred image (same shape, same dtype)
    """
    if kernel_size <= 1:
        return image

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    half = k // 2

    # Build motion blur kernel (line along angle)
    kernel = np.zeros((k, k), dtype=np.float64)
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    for i in range(-half, half + 1):
        x = int(round(i * cos_a)) + half
        y = int(round(i * sin_a)) + half
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    # Apply convolution
    if image.ndim == 3:
        result = np.empty_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = _convolve2d(image[:, :, c].astype(np.float64), kernel)
    else:
        result = _convolve2d(image.astype(np.float64), kernel)

    return np.clip(result, 0, 255).astype(image.dtype)


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply brightness, contrast, and gamma adjustment.

    Args:
        image: (H, W) or (H, W, 3) uint8 image
        brightness: brightness multiplier (1.0=normal)
        contrast: contrast multiplier (1.0=normal)
        gamma: gamma correction (1.0=linear)

    Returns:
        Adjusted image (same shape, same dtype)
    """
    img = image.astype(np.float64) / 255.0

    # Apply brightness
    img = img * brightness

    # Apply contrast (around mean)
    mean = img.mean()
    img = mean + (img - mean) * contrast

    # Apply gamma
    if gamma != 1.0:
        img = np.power(np.clip(img, 0, 1), gamma)

    return (np.clip(img, 0, 1) * 255).astype(image.dtype)


def apply_distractors(
    image: np.ndarray,
    count: int,
    min_size_px: float,
    max_size_px: float,
    min_brightness: float,
    max_brightness: float,
    timestamp_s: float,
    move_speed_px_s: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Add false target distractors to the image.

    Distractors are bright circular spots that move slowly.

    Args:
        image: (H, W) or (H, W, 3) uint8 image
        count: number of distractors
        min_size_px, max_size_px: radius range
        min_brightness, max_brightness: intensity range
        timestamp_s: current time (for motion)
        move_speed_px_s: movement speed
        rng: random state

    Returns:
        Image with distractors added
    """
    if count <= 0:
        return image

    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(count):
        # Deterministic positions based on seed + index + time
        base_x = rng.uniform(50, w - 50)
        base_y = rng.uniform(50, h - 50)
        # Slow movement
        phase = rng.uniform(0, 2 * math.pi)
        dx = move_speed_px_s * math.sin(timestamp_s * 0.5 + phase)
        dy = move_speed_px_s * math.cos(timestamp_s * 0.3 + phase * 0.7)
        cx = int(np.clip(base_x + dx, 0, w - 1))
        cy = int(np.clip(base_y + dy, 0, h - 1))
        radius = rng.uniform(min_size_px, max_size_px)
        brightness = rng.uniform(min_brightness, max_brightness)

        # Draw filled circle
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        if result.ndim == 3:
            for c in range(result.shape[2]):
                channel = result[:, :, c].astype(np.float64)
                channel[mask] = np.clip(brightness, 0, 255)
                result[:, :, c] = channel.astype(result.dtype)
        else:
            chan = result.astype(np.float64)
            chan[mask] = np.clip(brightness, 0, 255)
            result = chan.astype(result.dtype)

    return result


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution (valid border handling)."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image, dtype=np.float64)

    for i in range(ih):
        for j in range(iw):
            patch = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(patch * kernel)

    return result
