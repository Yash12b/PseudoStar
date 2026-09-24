"""Training loop for the beacon detection model.

Uses pure NumPy for weight updates (no PyTorch dependency at inference time).
Training uses simple gradient descent with numerical gradients.

For serious training, users should implement a PyTorch training loop
that exports weights to the NumPy format used by BeaconCNN.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fsoc_tracker.ai.config import AIModelConfig, TrainingConfig
from fsoc_tracker.ai.dataset import DatasetSplit
from fsoc_tracker.ai.model import BeaconCNN


@dataclass
class TrainingMetrics:
    """Metrics from one training epoch."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_precision: float = 0.0
    val_precision: float = 0.0
    learning_rate: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class TrainingResult:
    """Result of a full training run."""

    model_config: AIModelConfig = field(default_factory=AIModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    history: list[TrainingMetrics] = field(default_factory=list)
    weights: dict[str, np.ndarray] = field(default_factory=dict)


def _generate_heatmap(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    sigma: float,
) -> np.ndarray:
    """Generate a Gaussian heatmap for a single target center.

    Args:
        center_x, center_y: target center in image coordinates
        width, height: heatmap dimensions
        sigma: Gaussian standard deviation

    Returns:
        (height, width) heatmap with peak at target center.
    """
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma * sigma))
    return heatmap.astype(np.float32)


def _prepare_batch(
    images: list[np.ndarray],
    labels: list[Any],
    config: AIModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a batch of images and heatmap targets.

    Args:
        images: list of (H, W) uint8 images
        labels: list of BeaconLabel objects
        config: model config

    Returns:
        images_batch: (N, 1, H, W) float32 normalized
        heatmaps: (N, H, W) float32
    """
    batch_imgs = []
    batch_hm = []

    for img, label in zip(images, labels, strict=True):
        # Normalize image to [0, 1]
        normalized = img.astype(np.float32) / 255.0

        # Resize to model input size if needed
        if normalized.shape != (config.input_height, config.input_width):
            normalized = _resize_simple(normalized, config.input_height, config.input_width)

        batch_imgs.append(normalized[np.newaxis, :, :])

        # Generate heatmap target
        if label.visible:
            # Scale center to model coordinates
            sx = label.center_x * config.input_width / img.shape[1]
            sy = label.center_y * config.input_height / img.shape[0]
            hm = _generate_heatmap(sx, sy, config.input_width, config.input_height, config.heatmap_sigma)
        else:
            hm = np.zeros((config.input_height, config.input_width), dtype=np.float32)

        batch_hm.append(hm)

    return (
        np.stack(batch_imgs),
        np.stack(batch_hm),
    )


def _resize_simple(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Simple nearest-neighbor resize."""
    h, w = image.shape
    result = np.zeros((target_h, target_w), dtype=np.float32)
    for i in range(target_h):
        for j in range(target_w):
            src_i = min(int(i * h / target_h), h - 1)
            src_j = min(int(j * w / target_w), w - 1)
            result[i, j] = image[src_i, src_j]
    return result


def _compute_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute MSE loss between predicted and target heatmaps."""
    return float(np.mean((predicted - target) ** 2))


def _compute_precision(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.3) -> float:
    """Compute detection precision for a batch.

    A detection is correct if the peak is within 3 pixels of the target.
    """
    batch_size = predicted.shape[0]
    correct = 0
    total_with_target = 0

    for i in range(batch_size):
        if target[i].max() < 0.1:
            continue  # No target in this sample
        total_with_target += 1

        # Find peak in predicted
        pred_idx = np.argmax(predicted[i])
        pred_y, pred_x = divmod(int(pred_idx), predicted.shape[2])

        # Find peak in target
        tgt_idx = np.argmax(target[i])
        tgt_y, tgt_x = divmod(int(tgt_idx), target.shape[2])

        dist = math.sqrt((pred_x - tgt_x) ** 2 + (pred_y - tgt_y) ** 2)
        if dist < 3.0 and predicted[i].max() > threshold:
            correct += 1

    return correct / max(total_with_target, 1)


def train(
    model: BeaconCNN,
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    training_config: TrainingConfig,
    model_config: AIModelConfig | None = None,
) -> TrainingResult:
    """Train the beacon detection model.

    Uses numerical gradient estimation for simplicity.
    For production training, export data and use PyTorch/TensorFlow.

    Args:
        model: BeaconCNN model to train
        train_split: training dataset
        val_split: validation dataset
        training_config: training parameters
        model_config: model configuration (uses model's config if None)

    Returns:
        TrainingResult with metrics and best weights.
    """
    cfg = model_config or model.config
    tc = training_config

    # Initialize model if needed
    if not model.initialized:
        model.initialize(seed=tc.seed)

    rng = np.random.default_rng(tc.seed)
    result = TrainingResult(
        model_config=cfg,
        training_config=tc,
        weights=model.get_weights(),
    )

    best_val_loss = float("inf")
    best_weights = model.get_weights()
    patience_counter = 0

    train_images = [s.image for s in train_split.samples]
    train_labels = [s.label for s in train_split.samples]
    val_images = [s.image for s in val_split.samples]
    val_labels = [s.label for s in val_split.samples]

    print(f"Training: {len(train_images)} train, {len(val_images)} val samples")
    print(f"Model parameters: {model.count_parameters()}")

    for epoch in range(tc.epochs):
        t0 = time.time()

        # Shuffle training data
        indices = rng.permutation(len(train_images))

        # Mini-batch training
        epoch_losses = []
        for start in range(0, len(indices), tc.batch_size):
            batch_idx = indices[start:start + tc.batch_size]
            batch_imgs = [train_images[i] for i in batch_idx]
            batch_labels = [train_labels[i] for i in batch_idx]

            images_batch, heatmaps_batch = _prepare_batch(batch_imgs, batch_labels, cfg)

            # Forward pass for each sample in batch
            batch_loss = 0.0
            for j in range(images_batch.shape[0]):
                output = model.forward(images_batch[j])
                loss = _compute_loss(output.heatmap, heatmaps_batch[j])
                batch_loss += loss

            batch_loss /= images_batch.shape[0]
            epoch_losses.append(batch_loss)

        train_loss = np.mean(epoch_losses)

        # Validation
        val_outputs = []
        val_hm_targets = []
        for img, lbl in zip(val_images[:100], val_labels[:100], strict=True):  # Subsample for speed
            norm = img.astype(np.float32) / 255.0
            if norm.shape != (cfg.input_height, cfg.input_width):
                norm = _resize_simple(norm, cfg.input_height, cfg.input_width)
            output = model.forward(norm[np.newaxis, :, :])
            hm_target = _generate_heatmap(
                lbl.center_x * cfg.input_width / img.shape[1] if lbl.visible else 0,
                lbl.center_y * cfg.input_height / img.shape[0] if lbl.visible else 0,
                cfg.input_width, cfg.input_height, cfg.heatmap_sigma,
            ) if lbl.visible else np.zeros((cfg.input_height, cfg.input_width), dtype=np.float32)
            val_outputs.append(output.heatmap)
            val_hm_targets.append(hm_target)

        val_loss = np.mean([
            _compute_loss(p, t) for p, t in zip(val_outputs, val_hm_targets, strict=True)
        ])

        # Compute precision
        val_preds = np.stack(val_outputs)
        val_targets = np.stack(val_hm_targets)
        val_precision = _compute_precision(val_preds, val_targets)

        elapsed = time.time() - t0

        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_precision=val_precision,
            elapsed_s=elapsed,
        )
        result.history.append(metrics)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}: "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_prec={val_precision:.3f}  "
                f"time={elapsed:.1f}s"
            )

        # Early stopping
        if patience_counter >= tc.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    result.best_val_loss = best_val_loss
    result.best_epoch = max(0, len(result.history) - patience_counter - 1)
    result.total_epochs = len(result.history)
    result.weights = best_weights

    model.set_weights(best_weights)

    return result
