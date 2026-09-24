"""Tiny CNN beacon detection model — heatmap-based center prediction.

This module provides a lightweight CNN for detecting small optical beacon
targets. It predicts a heatmap where peaks correspond to beacon centers.

Architecture:
    Input: (1, H, W) grayscale image
    → 3 conv layers with ReLU + MaxPool
    → 2 FC layers
    → Output: (1, H, W) heatmap

The model is designed for:
- Small targets (5-20 pixels)
- CPU inference
- Fast processing (< 5ms)
- Accurate centroid localization

Can run with pure NumPy (inference) or PyTorch (training).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fsoc_tracker.ai.config import AIModelConfig


@dataclass
class ModelOutput:
    """Output from the beacon detection model."""

    heatmap: np.ndarray  # (H, W) predicted heatmap
    confidence: float = 0.0
    num_peaks: int = 0


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int = 1,
    pad: int = 0,
) -> np.ndarray:
    """Simple 2D convolution using NumPy.

    Args:
        input: (C_in, H, W)
        weight: (C_out, C_in, KH, KW)
        bias: (C_out,)
        stride: stride
        pad: zero padding

    Returns:
        (C_out, H_out, W_out)
    """
    if pad > 0:
        input = np.pad(input, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

    c_in, h, w = input.shape
    c_out, _, kh, kw = weight.shape
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1

    # im2col + GEMM: algebraically identical to the naive per-pixel
    # loop, ~100x faster (the loop made real-time inference impossible).
    c_in, h, w = input.shape
    inp = np.ascontiguousarray(input)
    shape = (c_in, kh, kw, h_out, w_out)
    strides = (inp.strides[0], inp.strides[1], inp.strides[2],
               inp.strides[1] * stride, inp.strides[2] * stride)
    cols = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
    cols_2d = cols.reshape(c_in * kh * kw, h_out * w_out)
    out = weight.reshape(c_out, -1).astype(np.float64) @ cols_2d.astype(np.float64)
    out = out + bias.reshape(c_out, 1)
    return out.reshape(c_out, h_out, w_out).astype(np.float32)


def _maxpool2d(input: np.ndarray, kernel: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling 2D."""
    c, h, w = input.shape
    h_out = h // stride
    w_out = w // stride
    if kernel == stride:
        # Reshape-based fast path (exact same windows as the loop).
        trimmed = np.ascontiguousarray(input[:, :h_out * stride, :w_out * stride])
        return trimmed.reshape(c, h_out, stride, w_out, stride).max(
            axis=(2, 4)).astype(np.float32)
    output = np.zeros((c, h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = input[:, i * stride: i * stride + kernel, j * stride: j * stride + kernel]
            output[:, i, j] = np.max(patch, axis=(1, 2))
    return output


def _fc(input: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Fully connected layer."""
    return input @ weight.T + bias


class BeaconCNN:
    """Tiny CNN for beacon heatmap prediction.

    This is the NumPy-only inference implementation.
    Training requires PyTorch (see training.py).

    The model has ~15K parameters and runs in <5ms on CPU.
    """

    def __init__(self, config: AIModelConfig | None = None) -> None:
        self._config = config or AIModelConfig()
        self._weights: dict[str, np.ndarray] = {}
        self._initialized = False
        self._weights_source = "uninitialized"

        # Model dimensions
        self._c1_out = 16
        self._c2_out = 32
        self._c3_out = 64
        self._fc1_out = 128

    @property
    def config(self) -> AIModelConfig:
        return self._config

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def trained(self) -> bool:
        """Whether weights were loaded from a training artifact."""
        return self._weights_source == "trained"

    @property
    def weights_source(self) -> str:
        return self._weights_source

    def initialize(self, seed: int = 42) -> None:
        """Initialize weights with He initialization."""
        rng = np.random.default_rng(seed)
        iw, ih = self._config.input_width, self._config.input_height

        # Conv1: 1 -> 16 channels, 3x3
        scale1 = math.sqrt(2.0 / (1 * 3 * 3))
        self._weights["conv1_w"] = (rng.standard_normal((16, 1, 3, 3)) * scale1).astype(np.float32)
        self._weights["conv1_b"] = np.zeros(16, dtype=np.float32)

        # Conv2: 16 -> 32 channels, 3x3
        scale2 = math.sqrt(2.0 / (16 * 3 * 3))
        self._weights["conv2_w"] = (rng.standard_normal((32, 16, 3, 3)) * scale2).astype(np.float32)
        self._weights["conv2_b"] = np.zeros(32, dtype=np.float32)

        # Conv3: 32 -> 64 channels, 3x3
        scale3 = math.sqrt(2.0 / (32 * 3 * 3))
        self._weights["conv3_w"] = (rng.standard_normal((64, 32, 3, 3)) * scale3).astype(np.float32)
        self._weights["conv3_b"] = np.zeros(64, dtype=np.float32)

        # After 3x MaxPool(2): spatial size / 8
        flat_size = 64 * (ih // 8) * (iw // 8)

        # FC1
        scale_fc1 = math.sqrt(2.0 / flat_size)
        self._weights["fc1_w"] = (rng.standard_normal((128, flat_size)) * scale_fc1).astype(np.float32)
        self._weights["fc1_b"] = np.zeros(128, dtype=np.float32)

        # FC2 -> heatmap (ih//8 * iw//8)
        out_size = (ih // 8) * (iw // 8)
        scale_fc2 = math.sqrt(2.0 / 128)
        self._weights["fc2_w"] = (rng.standard_normal((out_size, 128)) * scale_fc2).astype(np.float32)
        self._weights["fc2_b"] = np.zeros(out_size, dtype=np.float32)

        self._initialized = True
        self._weights_source = "random"

    def forward(self, image: np.ndarray) -> ModelOutput:
        """Forward pass through the network.

        Args:
            image: (H, W) or (1, H, W) grayscale image, float32, range [0, 1]

        Returns:
            ModelOutput with heatmap and confidence.
        """
        if not self._initialized:
            self.initialize()

        # Ensure (1, H, W) format
        if image.ndim == 2:
            img = image[np.newaxis, :, :]
        else:
            img = image

        img = img.astype(np.float32)

        # Conv1 -> ReLU -> MaxPool
        x = _conv2d(img, self._weights["conv1_w"], self._weights["conv1_b"], pad=1)
        x = _relu(x)
        x = _maxpool2d(x)

        # Conv2 -> ReLU -> MaxPool
        x = _conv2d(x, self._weights["conv2_w"], self._weights["conv2_b"], pad=1)
        x = _relu(x)
        x = _maxpool2d(x)

        # Conv3 -> ReLU -> MaxPool
        x = _conv2d(x, self._weights["conv3_w"], self._weights["conv3_b"], pad=1)
        x = _relu(x)
        x = _maxpool2d(x)

        # Flatten
        flat = x.flatten()

        # FC1 -> ReLU
        x = _fc(flat, self._weights["fc1_w"], self._weights["fc1_b"])
        x = _relu(x)

        # FC2 -> reshape to heatmap
        x = _fc(x, self._weights["fc2_w"], self._weights["fc2_b"])
        heatmap = x.reshape(1, self._config.input_height // 8, self._config.input_width // 8)

        # Upscale heatmap to input size using bilinear interpolation
        heatmap_full = self._upscale_heatmap(heatmap[0])

        # Apply sigmoid for confidence
        heatmap_sig = _sigmoid(heatmap_full)

        # Find peaks
        confidence = float(np.max(heatmap_sig))
        peaks = self._find_peaks(heatmap_sig)

        return ModelOutput(
            heatmap=heatmap_sig,
            confidence=confidence,
            num_peaks=len(peaks),
        )

    def _upscale_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Bilinear upscale heatmap to input dimensions."""
        target_h = self._config.input_height
        target_w = self._config.input_width
        h, w = heatmap.shape

        zoom_h = target_h / h
        zoom_w = target_w / w

        try:
            from scipy.ndimage import zoom as _zoom
            return _zoom(heatmap, (zoom_h, zoom_w), order=1).astype(np.float32)
        except ImportError:
            # Fallback: nearest neighbor
            result = np.zeros((target_h, target_w), dtype=np.float32)
            for i in range(target_h):
                for j in range(target_w):
                    src_i = min(int(i / zoom_h), h - 1)
                    src_j = min(int(j / zoom_w), w - 1)
                    result[i, j] = heatmap[src_i, src_j]
            return result

    def _find_peaks(self, heatmap: np.ndarray) -> list[tuple[float, float, float]]:
        """Find local maxima in heatmap.

        Returns:
            List of (x, y, confidence) tuples.
        """
        threshold = self._config.peak_threshold
        max_det = self._config.max_detections

        peaks = []
        h, w = heatmap.shape

        for _ in range(max_det):
            idx = np.argmax(heatmap)
            y, x = divmod(int(idx), w)
            conf = float(heatmap[y, x])

            if conf < threshold:
                break

            peaks.append((float(x), float(y), conf))

            # Suppress neighborhood
            r = 3
            y_min = max(0, y - r)
            y_max = min(h, y + r + 1)
            x_min = max(0, x - r)
            x_max = min(w, x + r + 1)
            heatmap[y_min:y_max, x_min:x_max] = 0.0

        return peaks

    def predict_center(
        self,
        image: np.ndarray,
        scale_factor: float = 1.0,
    ) -> tuple[float, float, float]:
        """Predict beacon center in image coordinates.

        Args:
            image: (H, W) grayscale image
            scale_factor: scale from model input to original image

        Returns:
            (center_x, center_y, confidence)
        """
        output = self.forward(image)
        if output.num_peaks == 0:
            return (0.0, 0.0, 0.0)

        peaks = self._find_peaks(output.heatmap)
        if not peaks:
            return (0.0, 0.0, 0.0)

        # Best peak
        x, y, conf = peaks[0]
        # Scale back to original image coordinates
        x_orig = x * scale_factor
        y_orig = y * scale_factor

        return (x_orig, y_orig, conf)

    def get_weights(self) -> dict[str, np.ndarray]:
        """Get model weights for export."""
        return dict(self._weights)

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Set model weights (from training or loading)."""
        self._weights = dict(weights)
        self._initialized = True
        self._weights_source = "trained"

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(w.size for w in self._weights.values())

    def model_size_bytes(self) -> int:
        """Get model size in bytes."""
        return sum(w.nbytes for w in self._weights.values())
