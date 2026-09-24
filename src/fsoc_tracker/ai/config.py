"""AI subsystem configuration.

Strongly typed Pydantic models for model, training, and inference parameters.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AIBackendType(str, Enum):
    CLASSICAL = "classical"
    AI = "ai"
    HYBRID = "hybrid"


class ModelArchitecture(str, Enum):
    TINY_CNN = "tiny_cnn"
    HEATMAP_CNN = "heatmap_cnn"


class AIModelConfig(BaseModel):
    """Configuration for the AI beacon detection model."""

    architecture: ModelArchitecture = ModelArchitecture.HEATMAP_CNN

    input_width: int = Field(default=128, ge=16, le=1920,
        description="Model input width (smaller = faster)")
    input_height: int = Field(default=128, ge=16, le=1080,
        description="Model input height")

    heatmap_sigma: float = Field(default=2.0, ge=0.5, le=10.0,
        description="Sigma for Gaussian heatmap ground truth")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0,
        description="Minimum confidence to report detection")
    peak_threshold: float = Field(default=0.1, ge=0.01, le=1.0,
        description="Minimum heatmap peak value to consider")
    max_detections: int = Field(default=5, ge=1, le=50,
        description="Maximum detections per frame")

    # ROI mode
    roi_enabled: bool = False
    roi_scale: float = Field(default=3.0, ge=1.5, le=10.0,
        description="ROI size as multiple of expected target size")
    roi_min_size: int = Field(default=32, ge=16)

    # Preprocessing
    normalize: bool = True
    clip_background: bool = True
    background_level: float = Field(default=10.0, ge=0.0)

    # Device
    device: str = "cpu"


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)

    # Dataset
    train_samples: int = Field(default=5000, ge=100)
    val_samples: int = Field(default=500, ge=50)
    test_samples: int = Field(default=500, ge=50)

    train_seed: int = 1000
    val_seed: int = 2000
    test_seed: int = 3000

    # Augmentation
    aug_noise_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    aug_blur_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    aug_brightness_prob: float = Field(default=0.4, ge=0.0, le=1.0)
    aug_flip_prob: float = Field(default=0.5, ge=0.0, le=1.0)

    # Target size range
    min_target_size: int = Field(default=5, ge=3)
    max_target_size: int = Field(default=20, ge=5)

    # Image size
    image_width: int = 640
    image_height: int = 480

    # Output
    output_dir: str = "artifacts/models"
    save_every_n_epochs: int = 5

    # Reproducibility
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 10


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    num_samples: int = Field(default=1000, ge=10)
    val_samples: int = Field(default=500, ge=10)
    test_samples: int = Field(default=500, ge=10)
    image_width: int = 640
    image_height: int = 480

    # Target
    min_target_size: int = Field(default=5, ge=3)
    max_target_size: int = Field(default=20, ge=5)
    min_target_brightness: float = Field(default=100.0, ge=0.0)
    max_target_brightness: float = Field(default=255.0, ge=0.0)

    # Position distribution
    center_margin: int = Field(default=10, ge=0,
        description="Margin from edge where target can appear")

    # Negative samples
    negative_ratio: float = Field(default=0.2, ge=0.0, le=0.5,
        description="Fraction of samples with no beacon")
    hard_negative_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of negative samples containing bright distractors",
    )

    # Distractors
    num_distractors_range: tuple[int, int] = (0, 3)
    distractor_max_size: int = 15

    # Disturbance probabilities
    noise_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    fog_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    haze_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    low_light_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    # Heavy-noise training range (PS maxima). Defaults preserve the
    # legacy mild range; raise for robustness training.
    noise_sigma_max: float = Field(default=15.0, ge=0.0, le=50.0)
    salt_pepper_max: float = Field(default=0.1, ge=0.0, le=0.5)
    poisson_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    # Timing
    nominal_fps: float = Field(default=30.0, gt=0,
        description="Nominal FPS for timestamp generation in disturbance contexts")
    jitter_prob: float = Field(default=0.2, ge=0.0, le=1.0)

    # Split seeds
    train_seed: int = 1000
    val_seed: int = 2000
    test_seed: int = 3000

    # Output
    output_dir: str = "artifacts/datasets"
    dataset_version: str = "beacon-images-v2"
    hard_test_samples: int = Field(default=500, ge=10)
    hard_test_seed: int = 4000
