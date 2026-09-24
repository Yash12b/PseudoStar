"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fsoc_tracker.config.settings import (
    CameraConfig,
    ControlConfig,
    RootConfig,
    TrackingConfig,
    load_config,
)
from fsoc_tracker.core.exceptions import ConfigurationError


class TestCameraConfig:
    """Tests for CameraConfig defaults and validation."""

    def test_defaults_match_sih_reference(self) -> None:
        cfg = CameraConfig()
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.horizontal_fov_deg == 4.0
        assert cfg.vertical_fov_deg == 3.0
        assert cfg.update_rate_hz == 30.0

    def test_rejects_zero_width(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(width=0)

    def test_rejects_negative_fov(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(horizontal_fov_deg=-1.0)


class TestTrackingConfig:
    """Tests for TrackingConfig defaults matching SIH reference values."""

    def test_sih_limits(self) -> None:
        cfg = TrackingConfig()
        assert cfg.acquisition_timeout_s == 2.0
        assert cfg.reacquisition_timeout_s == 1.0
        assert cfg.tracking_error_limit_px == 10.0
        assert cfg.target_loss_limit_percent == 5.0
        assert cfg.min_processing_fps == 20.0


class TestControlConfig:
    """Tests for ControlConfig defaults."""

    def test_sih_speed_limits(self) -> None:
        cfg = ControlConfig()
        assert cfg.max_pan_speed_deg_s == 5.0
        assert cfg.max_tilt_speed_deg_s == 5.0
        assert cfg.update_rate_hz == 20.0


class TestLoadConfig:
    """Tests for the YAML configuration loader."""

    def test_load_default(self) -> None:
        config = load_config()
        assert config.camera.width == 640
        assert config.simulation.canvas_width == 2000
        assert config.app.session_id.startswith("session-")

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        cfg = {"camera": {"width": 1280, "height": 720}}
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(cfg))

        config = load_config(config_path=config_file)
        assert config.camera.width == 1280
        assert config.camera.height == 720

    def test_overrides(self) -> None:
        config = load_config(overrides={"camera": {"width": 1920}})
        assert config.camera.width == 1920
        # other defaults preserved
        assert config.camera.height == 480

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(config_path="/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : invalid yaml {{{")
        with pytest.raises(ConfigurationError, match="Failed to parse"):
            load_config(config_path=bad)

    def test_session_id_auto_generated(self) -> None:
        config = load_config()
        assert len(config.app.session_id) > 8

    def test_custom_session_id_preserved(self) -> None:
        config = load_config(overrides={"app": {"session_id": "my-session"}})
        assert config.app.session_id == "my-session"


class TestAppConfigRoot:
    """Tests for the root configuration object."""

    def test_all_sub_configs_present(self) -> None:
        config = RootConfig()
        assert hasattr(config, "app")
        assert hasattr(config, "camera")
        assert hasattr(config, "target")
        assert hasattr(config, "simulation")
        assert hasattr(config, "tracking")
        assert hasattr(config, "control")
        assert hasattr(config, "disturbance")
        assert hasattr(config, "input")
        assert hasattr(config, "output")
        assert hasattr(config, "performance")
        assert hasattr(config, "visualization")
        assert hasattr(config, "benchmark")
