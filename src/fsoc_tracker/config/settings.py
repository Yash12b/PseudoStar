"""Centralized configuration system with YAML loading and validation.

Uses Pydantic v2 ``BaseModel`` for automatic validation, defaults,
and type coercion.  Configuration is loaded from YAML files with
optional command-line overrides.

SIH26169 reference values are encoded as defaults, not as constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from fsoc_tracker.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------

class AppSectionConfig(BaseModel):
    """Top-level application settings."""

    name: str = "FSOC Tracker"
    version: str = "0.1.0"
    session_id: str = ""
    log_level: str = "INFO"
    log_dir: str = "logs"


class CameraConfig(BaseModel):
    """Virtual / physical camera parameters.

    Reference values from SIH26169 PS:
        - 640 x 480 default resolution
        - 4° x 3° FOV
        - 30 Hz update rate
    """

    width: int = Field(default=640, ge=1)
    height: int = Field(default=480, ge=1)
    horizontal_fov_deg: float = Field(default=4.0, gt=0)
    vertical_fov_deg: float = Field(default=3.0, gt=0)
    update_rate_hz: float = Field(default=30.0, gt=0)
    color_model: str = "BGR"


class TargetConfig(BaseModel):
    """Beacon target parameters.

    Reference from SIH26169 PS:
        - ~10 x 10 px default target
        - configurable motion patterns
    """

    size_px: float = Field(default=10.0, gt=0)
    shape: str = "square"
    color_bgr: list[int] = Field(default_factory=lambda: [255, 255, 255])
    motion_type: str = "straight_line"


class SimulationConfig(BaseModel):
    """Virtual environment / simulation settings.

    Reference from SIH26169 PS:
        - minimum 2000 x 2000 virtual canvas
    """

    canvas_width: int = Field(default=2000, ge=1)
    canvas_height: int = Field(default=2000, ge=1)
    canvas_depth: float = Field(default=100.0, gt=0)
    world_unit: str = "pixel"


class TrackingConfig(BaseModel):
    """Tracking subsystem parameters.

    Reference from SIH26169 PS:
        - acquisition <= 2 sec
        - re-acquisition <= 1 sec
        - tracking error <= 10 px
        - target loss < 5%
        - processing >= 20 FPS
    """

    acquisition_timeout_s: float = Field(default=2.0, gt=0)
    reacquisition_timeout_s: float = Field(default=1.0, gt=0)
    tracking_error_limit_px: float = Field(default=10.0, gt=0)
    target_loss_limit_percent: float = Field(default=5.0, ge=0, le=100)
    min_processing_fps: float = Field(default=20.0, gt=0)
    algorithm: str = "nearest_centroid"


class PIDConfig(BaseModel):
    """PID controller gains."""

    kp_pan: float = 0.5
    ki_pan: float = 0.01
    kd_pan: float = 0.1
    kp_tilt: float = 0.5
    ki_tilt: float = 0.01
    kd_tilt: float = 0.1


class ControlConfig(BaseModel):
    """Camera control parameters.

    Reference from SIH26169 PS:
        - max pan/tilt speed: 5°/s
        - control update >= 20 Hz
    """

    max_pan_speed_deg_s: float = Field(default=5.0, gt=0)
    max_tilt_speed_deg_s: float = Field(default=5.0, gt=0)
    update_rate_hz: float = Field(default=20.0, gt=0)
    algorithm: str = "pid"
    pid: PIDConfig = Field(default_factory=PIDConfig)


class DisturbanceConfig(BaseModel):
    """Disturbance / noise model parameters."""

    enabled: bool = False
    types: list[str] = Field(default_factory=list)
    gaussian_sigma: float = Field(default=5.0, ge=0)
    salt_pepper_amount: float = Field(default=0.01, ge=0, le=1)
    poisson_gain: float = Field(default=3.0, ge=0)
    jitter_px: float = Field(default=2.0, ge=0)
    haze_density: float = Field(default=0.3, ge=0, le=1)
    fog_density: float = Field(default=0.5, ge=0, le=1)
    rain_intensity: float = Field(default=0.3, ge=0, le=1)


class InputConfig(BaseModel):
    """Input source configuration."""

    source_type: str = "synthetic"
    video_path: str = ""
    camera_id: int = 0
    stream_url: str = ""
    loop: bool = False


class OutputConfig(BaseModel):
    """Output configuration."""

    save_video: bool = False
    video_path: str = "data/outputs/"
    save_frames: bool = False
    frames_dir: str = "data/outputs/frames/"


class PerformanceConfig(BaseModel):
    """Performance tuning parameters."""

    max_fps: int = Field(default=60, gt=0)
    enable_profiling: bool = False


class VisualizationConfig(BaseModel):
    """Visualization and HUD settings."""

    enabled: bool = True
    show_hud: bool = True
    show_camera_view: bool = True
    show_global_scene: bool = False
    hud_font_scale: float = Field(default=0.6, gt=0)
    hud_color: list[int] = Field(default_factory=lambda: [0, 255, 0])


class BenchmarkConfig(BaseModel):
    """Benchmark engine settings."""

    enabled: bool = False
    video_dir: str = "data/input/"
    report_dir: str = "data/outputs/benchmarks/"


class SensorImageConfig(BaseModel):
    """Sensor/image-formation configuration.

    Reference from SIH26169 PS:
        - 640 x 480 monochrome
        - beacon default apparent size ~10 px
        - size range 5-20 px
    """

    width: int = Field(default=640, ge=1)
    height: int = Field(default=480, ge=1)
    channels: int = Field(default=1, ge=1, le=4)
    color_mode: str = "mono"
    bit_depth: int = Field(default=8, ge=1, le=32)
    max_intensity: float = Field(default=255.0, gt=0)
    background_level: float = Field(default=5.0, ge=0)
    background_noise_level: float = Field(default=0.0, ge=0)
    beacon_default_size_px: float = Field(default=10.0, gt=0)
    minimum_beacon_size_px: float = Field(default=5.0, gt=0)
    maximum_beacon_size_px: float = Field(default=20.0, gt=0)
    beacon_peak_intensity: float = Field(default=255.0, ge=0)
    beacon_shape: str = "square"
    beacon_soft_edges: bool = True
    size_mode: str = "fixed"
    physical_target_size_m: float = Field(default=0.01, gt=0)
    enable_psf: bool = False
    psf_sigma_px: float = Field(default=1.0, gt=0)
    enable_anti_aliasing: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class RootConfig(BaseModel):
    """Top-level configuration combining all sub-configs.

    This is the object returned by ``load_config()``.
    """

    app: AppSectionConfig = Field(default_factory=AppSectionConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    sensor: SensorImageConfig = Field(default_factory=SensorImageConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    disturbance: DisturbanceConfig = Field(default_factory=DisturbanceConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


# Short alias used throughout the codebase (root config).
AppConfig = RootConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env_prefix: str = "FSOC_",
) -> RootConfig:
    """Load configuration from a YAML file with optional overrides.

    Priority (highest wins):
        1. *overrides* dict (e.g. CLI arguments)
        2. Environment variables prefixed with *env_prefix*
        3. YAML file values
        4. Pydantic defaults (SIH reference values)

    Args:
        config_path: Path to a YAML configuration file.
        overrides: Dictionary of overrides applied on top of the file.
        env_prefix: Prefix for environment variable lookups.

    Returns:
        A fully validated ``RootConfig`` instance.

    Raises:
        ConfigurationError: If the file is missing or contains invalid data.
    """
    raw: dict[str, Any] = {}

    # 1. Load YAML file
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse YAML config: {exc}") from exc

    # 2. Apply overrides
    if overrides:
        _deep_merge(raw, overrides)

    # 3. Build config (pydantic handles validation and defaults)
    try:
        config = RootConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc

    # 4. Auto-generate session_id if empty
    if not config.app.session_id:
        import uuid
        config.app.session_id = f"session-{uuid.uuid4().hex[:8]}"

    return config
