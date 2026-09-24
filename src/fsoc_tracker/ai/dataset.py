"""Synthetic dataset generation for AI beacon detection training.

Generates training data from the virtual simulator with controlled
diversity in target size, position, brightness, and disturbances.

Ground truth is used ONLY for dataset generation, never at runtime.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_tracker.ai.config import DatasetConfig
from fsoc_tracker.disturbances.config import (
    AtmosphereConfig,
    DisturbanceConfig,
    NoiseConfig,
)
from fsoc_tracker.disturbances.context import DisturbanceContext
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline


@dataclass
class BeaconLabel:
    """Label for a single training sample."""

    visible: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_width: int = 0
    bbox_height: int = 0
    target_size_px: float = 0.0
    brightness: float = 0.0
    confidence: float = 1.0
    has_distractors: bool = False
    num_distractors: int = 0
    disturbance_types: list[str] = field(default_factory=list)
    category: str = "positive"


@dataclass
class DatasetSample:
    """A single training/validation/test sample."""

    image: np.ndarray
    label: BeaconLabel
    sample_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """A collection of samples for one split."""

    samples: list[DatasetSample] = field(default_factory=list)
    split_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_visible(self) -> int:
        return sum(1 for s in self.samples if s.label.visible)

    @property
    def num_negative(self) -> int:
        return sum(1 for s in self.samples if not s.label.visible)

    def statistics(self) -> dict[str, Any]:
        """Return reproducible distribution statistics for this split."""
        sizes = [
            round(float(sample.label.target_size_px), 3)
            for sample in self.samples
            if sample.label.visible
        ]
        disturbance_counts = Counter(
            disturbance
            for sample in self.samples
            for disturbance in sample.label.disturbance_types
        )
        return {
            "split": self.split_name,
            "samples": self.num_samples,
            "visible": self.num_visible,
            "negative": self.num_negative,
            "positive_ratio": self.num_visible / max(self.num_samples, 1),
            "categories": dict(Counter(sample.label.category for sample in self.samples)),
            "target_sizes": dict(Counter(str(size) for size in sizes)),
            "disturbances": dict(disturbance_counts),
            "image_shape": (
                list(self.samples[0].image.shape) if self.samples else None
            ),
            "config": self.config,
        }


def _deposit_beacon_simple(
    image: np.ndarray,
    cx: float,
    cy: float,
    size_px: float,
    brightness: float,
    rng: np.random.Generator,
) -> None:
    """Deposit a beacon spot into the image (simple Gaussian model).

    Args:
        image: (H, W) float64 image to modify in-place
        cx, cy: center coordinates
        size_px: beacon size in pixels
        brightness: peak intensity
        rng: random generator
    """
    h, w = image.shape
    sigma = max(size_px / 4.0, 0.5)

    # Determine region of interest
    r = int(math.ceil(sigma * 3))
    x_min = max(0, int(cx) - r)
    x_max = min(w, int(cx) + r + 1)
    y_min = max(0, int(cy) - r)
    y_max = min(h, int(cy) + r + 1)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            val = brightness * math.exp(-dist_sq / (2.0 * sigma * sigma))
            image[y, x] = min(image[y, x] + val, 255.0)


def _deposit_distractor(
    image: np.ndarray,
    cx: float,
    cy: float,
    size_px: float,
    brightness: float,
    rng: np.random.Generator,
) -> None:
    """Deposit a distractor (dimmer, possibly larger spot)."""
    _deposit_beacon_simple(image, cx, cy, size_px, brightness * 0.6, rng)


def generate_sample(
    sample_id: int,
    config: DatasetConfig,
    rng: np.random.Generator,
    include_disturbances: bool = True,
) -> DatasetSample:
    """Generate a single training sample.

    Args:
        sample_id: unique sample index
        config: dataset configuration
        rng: deterministic random generator
        include_disturbances: whether to apply random disturbances

    Returns:
        DatasetSample with image and label.
    """
    w = config.image_width
    h = config.image_height
    image = np.full((h, w), 5.0, dtype=np.float64)

    # Decide: negative sample or positive
    is_negative = rng.random() < config.negative_ratio

    label = BeaconLabel(visible=False, category="negative_empty")

    if not is_negative:
        # Generate beacon
        margin = min(config.center_margin, min(w, h) / 2.0)
        cx = rng.uniform(margin, w - margin)
        cy = rng.uniform(margin, h - margin)

        # Sub-pixel position
        cx += rng.uniform(-0.5, 0.5)
        cy += rng.uniform(-0.5, 0.5)

        size_px = float(rng.uniform(config.min_target_size, config.max_target_size))
        brightness = float(
            rng.uniform(config.min_target_brightness, config.max_target_brightness)
        )

        _deposit_beacon_simple(image, cx, cy, size_px, brightness, rng)

        # Bounding box
        half = size_px / 2.0
        bbox_x = max(0, int(cx - half))
        bbox_y = max(0, int(cy - half))
        bbox_w = min(w - bbox_x, int(size_px))
        bbox_h = min(h - bbox_y, int(size_px))

        label = BeaconLabel(
            visible=True,
            center_x=cx,
            center_y=cy,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_w,
            bbox_height=bbox_h,
            target_size_px=size_px,
            brightness=brightness,
            category="positive",
        )

        # Add distractors
        num_distractors = int(rng.integers(
            config.num_distractors_range[0],
            config.num_distractors_range[1] + 1,
        ))
        if num_distractors > 0:
            label.has_distractors = True
            label.num_distractors = int(num_distractors)
            for _ in range(num_distractors):
                dx = rng.uniform(0, w)
                dy = rng.uniform(0, h)
                ds = rng.uniform(3, config.distractor_max_size)
                db = rng.uniform(30, brightness * 0.5)
                _deposit_distractor(image, dx, dy, ds, db, rng)
    elif rng.random() < config.hard_negative_ratio:
        # Hard negatives contain beacon-like structure without a labelled
        # beacon. They prevent the model from equating any bright point with
        # the target.
        label.category = "hard_negative"
        for _ in range(int(rng.integers(1, 4))):
            dx = rng.uniform(0, w)
            dy = rng.uniform(0, h)
            _deposit_distractor(
                image,
                dx,
                dy,
                rng.uniform(3, config.distractor_max_size),
                rng.uniform(40, 220),
                rng,
            )

    # Apply disturbances
    disturbance_types = []
    if include_disturbances:
        dist_config = DisturbanceConfig(enabled=False)

        if rng.random() < config.noise_prob:
            dist_config.enabled = True
            dist_config.noise = NoiseConfig(
                enabled=True,
                gaussian_sigma=rng.uniform(0, config.noise_sigma_max),
                salt_pepper_density=rng.uniform(0, config.salt_pepper_max),
                poisson_enabled=rng.random() < config.poisson_prob,
            )
            disturbance_types.append("noise")

        if rng.random() < config.fog_prob:
            dist_config.enabled = True
            dist_config.atmosphere = AtmosphereConfig(
                enabled=True,
                fog_strength=rng.uniform(0.1, 0.5),
            )
            disturbance_types.append("fog")

        if rng.random() < config.haze_prob:
            dist_config.enabled = True
            if not dist_config.atmosphere.enabled:
                dist_config.atmosphere = AtmosphereConfig(enabled=True)
            dist_config.atmosphere.haze_strength = rng.uniform(0.1, 0.4)
            disturbance_types.append("haze")

        if rng.random() < config.low_light_prob:
            disturbance_types.append("low_light")
            # Reduce brightness of the whole image
            factor = rng.uniform(0.3, 0.8)
            image *= factor

        if dist_config.enabled:
            pipeline = DisturbancePipeline(dist_config)
            ctx = DisturbanceContext(
                timestamp_s=float(sample_id) / config.nominal_fps,
                frame_index=sample_id,
                image_width=w,
                image_height=h,
            )
            image = pipeline.apply_to_image(image, ctx)

    label.disturbance_types = disturbance_types

    # Add baseline noise
    image += rng.normal(0, 1.0, image.shape)

    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return DatasetSample(
        image=image,
        label=label,
        sample_id=sample_id,
        metadata={
            "disturbances": disturbance_types,
            "category": label.category,
            "seed_sample_id": sample_id,
        },
    )


def generate_split(
    split_name: str,
    num_samples: int,
    config: DatasetConfig,
    seed: int,
    include_disturbances: bool = True,
) -> DatasetSplit:
    """Generate a dataset split.

    Args:
        split_name: "train", "val", or "test"
        num_samples: number of samples
        config: dataset config
        seed: deterministic seed
        include_disturbances: apply disturbances

    Returns:
        DatasetSplit with all samples.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for i in range(num_samples):
        sample = generate_sample(i, config, rng, include_disturbances)
        samples.append(sample)

    return DatasetSplit(
        samples=samples,
        split_name=split_name,
        config={
            "num_samples": num_samples,
            "seed": seed,
            "include_disturbances": include_disturbances,
            "dataset_version": config.dataset_version,
        },
    )


def generate_full_dataset(
    config: DatasetConfig,
    save: bool = True,
) -> dict[str, DatasetSplit]:
    """Generate train, validation, test, and independent hard-test splits."""
    splits = {
        "train": generate_split(
            "train", config.num_samples, config, config.train_seed, True
        ),
        "validation": generate_split(
            "validation", config.val_samples, config, config.val_seed, True
        ),
        "test": generate_split(
            "test", config.test_samples, config, config.test_seed, True
        ),
        "hard_test": generate_split(
            "hard_test",
            config.hard_test_samples,
            config.model_copy(
                update={
                    "negative_ratio": max(config.negative_ratio, 0.5),
                    "hard_negative_ratio": 1.0,
                    "center_margin": 0,
                    "noise_prob": 1.0,
                    "fog_prob": 1.0,
                    "haze_prob": 1.0,
                    "low_light_prob": 1.0,
                }
            ),
            config.hard_test_seed,
            True,
        ),
    }
    if save:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset_version": config.dataset_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "splits": {},
        }
        for name, split in splits.items():
            split_dir = output_dir / name
            split_dir.mkdir(exist_ok=True)
            np.save(split_dir / "images.npy", np.stack([s.image for s in split.samples]))
            (split_dir / "labels.json").write_text(
                json.dumps([asdict(s.label) for s in split.samples], indent=2),
                encoding="utf-8",
            )
            manifest["splits"][name] = split.statistics()
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return splits


def generate_dataset(
    config: DatasetConfig,
    save: bool = True,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Generate full train/val/test dataset.

    Args:
        config: dataset configuration
        save: whether to save to disk

    Returns:
        (train, val, test) splits.
    """
    train = generate_split(
        "train", config.num_samples, config, config.train_seed,
    )
    val = generate_split(
        "val", config.val_samples, config, config.val_seed,
    )
    test = generate_split(
        "test", config.test_samples, config, config.test_seed,
    )

    if save:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in [train, val, test]:
            split_dir = output_dir / split.split_name
            split_dir.mkdir(exist_ok=True)

            # Save images
            images = np.stack([s.image for s in split.samples])
            np.save(split_dir / "images.npy", images)

            # Save labels
            labels = [asdict(s.label) for s in split.samples]
            with open(split_dir / "labels.json", "w") as f:
                json.dump(labels, f, indent=2)

            # Save metadata
            with open(split_dir / "metadata.json", "w") as f:
                json.dump(split.config, f, indent=2)

    return train, val, test
