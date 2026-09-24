"""Perception configuration.

Strongly typed Pydantic model for detection parameters.
Defaults align with SIH26169 beacon specifications.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ThresholdMode(str, Enum):
    GLOBAL = "global"
    ADAPTIVE = "adaptive"
    PERCENTILE = "percentile"


class CentroidMethod(str, Enum):
    GEOMETRIC = "geometric"
    INTENSITY_WEIGHTED = "intensity_weighted"


class PerceptionConfig(BaseModel):
    """Configuration for the perception/detection subsystem.

    Reference from SIH26169 PS:
        - beacon ~10x10 px default
        - range 5x5 to 20x20
        - one primary target
    """

    enabled: bool = True
    algorithm: str = "classical_bright_spot"

    threshold_mode: ThresholdMode = ThresholdMode.PERCENTILE
    threshold_value: float = Field(default=180.0, ge=0)
    percentile_value: float = Field(default=95.0, ge=0, le=100)
    adaptive_block_size: int = Field(default=11, ge=3)
    adaptive_constant: float = Field(default=2.0, ge=0)

    min_candidate_area: float = Field(default=5.0, gt=0)
    max_candidate_area: float = Field(default=2000.0, gt=0)
    # Anti-flood gate: salt-and-pepper noise can produce thousands of
    # above-threshold clumps; scoring all of them stalls the pipeline
    # (measured 8 s/frame). Keep the largest N — genuine beacons (tens
    # to hundreds of px) always outrank noise specks. The active bound
    # is reported per frame as diagnostics["candidate_cap"].
    max_candidates: int = Field(default=128, ge=1)
    # Full-processing cap: only the largest N masks get feature
    # extraction + centroiding + scoring (the per-candidate bottleneck,
    # ~1 ms each). Masks arrive largest-first, so genuine beacons
    # (tens-hundreds of px) always precede noise specks. Clean frames
    # (< N candidates) behave exactly as before; floods stay bounded.
    max_scored_candidates: int = Field(default=24, ge=1)
    min_width: float = Field(default=2.0, gt=0)
    max_width: float = Field(default=50.0, gt=0)
    min_height: float = Field(default=2.0, gt=0)
    max_height: float = Field(default=50.0, gt=0)

    expected_size_px: float = Field(default=10.0, gt=0)
    size_tolerance_px: float = Field(default=8.0, ge=0)

    min_confidence: float = Field(default=0.3, ge=0, le=1)

    blur_sigma: float = Field(default=0.0, ge=0)
    # Salt-and-pepper/Gaussian robustness (PS disturbances): the median
    # applies only when measured noise exceeds 1.0, so clean frames pass
    # through unchanged.
    denoise_enabled: bool = True

    use_weighted_centroid: bool = True
    centroid_method: CentroidMethod = CentroidMethod.INTENSITY_WEIGHTED

    morphology_enabled: bool = False
    morphology_kernel_size: int = Field(default=3, ge=1)

    w_intensity: float = Field(default=0.3, ge=0)
    w_size: float = Field(default=0.25, ge=0)
    w_shape: float = Field(default=0.25, ge=0)
    w_contrast: float = Field(default=0.2, ge=0)

    background_estimate_method: str = "percentile"
    background_percentile: float = Field(default=10.0, ge=0, le=100)

    normalize_contrast_enabled: bool = False

    @classmethod
    def for_video(cls) -> PerceptionConfig:
        """Create a config tuned for real-world video / live camera input.

        Simulation beacons are near-black backgrounds with a 255-intensity
        Gaussian spot — the stock defaults work there.  Real video has
        higher backgrounds, lower contrast ratios, and variable beacon
        sizes, so we relax thresholds and enable contrast normalisation.
        The 8 px area floor rejects text-glyph/OSD fragments (typically
        3-4 px) while keeping 5 px beacons (PS minimum, ~20+ px area).
        The 50th percentile (vs 95 for simulation) captures beacons down
        to roughly half the frame's dynamic range, as required for
        low-light footage where the beacon no longer dominates the
        bright tail (e.g. behind brighter overlay content).
        """
        return cls(
            percentile_value=50.0,
            min_candidate_area=8.0,
            max_candidate_area=5000.0,
            expected_size_px=8.0,
            size_tolerance_px=12.0,
            min_confidence=0.2,
            normalize_contrast_enabled=True,
            denoise_enabled=True,
            blur_sigma=0.0,
            w_intensity=0.25,
            w_size=0.2,
            w_shape=0.25,
            w_contrast=0.3,
            background_percentile=5.0,
        )
