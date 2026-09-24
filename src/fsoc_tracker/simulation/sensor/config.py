"""Sensor configuration with SIH26169 defaults.

Strongly typed Pydantic model for sensor/image-formation parameters.
All values are configurable; defaults align with SIH26169 where applicable.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ColorMode(str, Enum):
    MONO = "mono"
    BGR = "bgr"
    RGB = "rgb"


class BeaconShape(str, Enum):
    SQUARE = "square"
    CIRCULAR = "circular"


class SizeMode(str, Enum):
    FIXED = "fixed"
    DISTANCE_BASED = "distance_based"


class SensorConfig(BaseModel):
    """Virtual optical sensor configuration.

    Default values align with SIH26169 PS:
        - 640 x 480 monochrome
        - beacon default apparent size ~10 px
        - size range 5-20 px
    """

    width: int = Field(default=640, ge=1)
    height: int = Field(default=480, ge=1)
    channels: int = Field(default=1, ge=1, le=4)
    color_mode: ColorMode = ColorMode.MONO

    bit_depth: int = Field(default=8, ge=1, le=32)
    max_intensity: float = Field(default=255.0, gt=0)

    background_level: float = Field(default=5.0, ge=0)
    background_noise_level: float = Field(default=0.0, ge=0)

    beacon_default_size_px: float = Field(default=10.0, gt=0)
    minimum_beacon_size_px: float = Field(default=5.0, gt=0)
    maximum_beacon_size_px: float = Field(default=20.0, gt=0)
    beacon_peak_intensity: float = Field(default=255.0, ge=0)
    beacon_shape: BeaconShape = BeaconShape.SQUARE
    beacon_soft_edges: bool = True
    # Identity-code modulation: frames per code bit. Must match the
    # tracker's identity_frames_per_bit (both default 3); the identity
    # decoder and this renderer otherwise disagree on bit timing.
    code_frames_per_bit: int = Field(default=3, ge=1, le=30)

    size_mode: SizeMode = SizeMode.FIXED
    physical_target_size_m: float = Field(default=0.01, gt=0)

    enable_psf: bool = False
    psf_sigma_px: float = Field(default=1.0, gt=0)

    enable_anti_aliasing: bool = True

    seed: int = 42

    def clamped_beacon_size(self, size_px: float) -> float:
        return max(self.minimum_beacon_size_px, min(self.maximum_beacon_size_px, size_px))
