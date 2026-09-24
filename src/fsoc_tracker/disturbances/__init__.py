"""Disturbance subsystem — environmental and sensor degradation models.

Modules:
    config      — DisturbanceConfig, DisturbanceMode, severity presets
    context     — DisturbanceContext typed data model
    noise       — Gaussian, Poisson, Salt-and-Pepper noise models
    atmosphere  — Clear, Haze, Fog, Rain, LowLight, Turbulence
    motion      — Camera jitter, platform motion (geometric, pre-render)
    pipeline    — DisturbancePipeline with pre/post-render separation
    telemetry   — DisturbanceTelemetry, DisturbancePerformance
"""

from fsoc_tracker.disturbances.config import (
    AtmosphereConfig,
    BlurConfig,
    BrightnessContrastConfig,
    DistractorConfig,
    DisturbanceConfig,
    DisturbanceMode,
    JitterConfig,
    MotionBlurConfig,
    NoiseConfig,
    PlatformMotionConfig,
    TargetDisappearanceConfig,
    TurbulenceConfig,
)
from fsoc_tracker.disturbances.context import DisturbanceContext
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.disturbances.telemetry import DisturbancePerformance, DisturbanceTelemetry

__all__ = [
    "DisturbanceConfig",
    "DisturbanceMode",
    "NoiseConfig",
    "AtmosphereConfig",
    "JitterConfig",
    "PlatformMotionConfig",
    "TurbulenceConfig",
    "BlurConfig",
    "MotionBlurConfig",
    "BrightnessContrastConfig",
    "TargetDisappearanceConfig",
    "DistractorConfig",
    "DisturbanceContext",
    "DisturbancePipeline",
    "DisturbanceTelemetry",
    "DisturbancePerformance",
]
