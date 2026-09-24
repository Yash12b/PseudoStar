from __future__ import annotations

import pytest

from fsoc_tracker.ai.config import DatasetConfig
from fsoc_tracker.ai.dataset import generate_split
from fsoc_tracker.ai.visual_training import train_visual_model


def test_visual_training_requires_optional_backend(tmp_path):
    config = DatasetConfig(
        num_samples=10,
        image_width=32,
        image_height=32,
        min_target_size=5,
        max_target_size=8,
    )
    train = generate_split("train", 2, config, seed=10, include_disturbances=False)
    validation = generate_split(
        "validation", 2, config, seed=20, include_disturbances=False
    )
    try:
        train_visual_model(train, validation, tmp_path / "visual.pt", epochs=1)
    except RuntimeError as exc:
        assert "PyTorch is required" in str(exc)
    else:
        pytest.skip("PyTorch is installed; optional backend path is available")
