"""Core domain models for the FSOC tracking pipeline.

The authoritative types for each subsystem live in their respective modules:
    - Frame:           core.models (shared across all subsystems)
    - TargetState:     core.models (shared across all subsystems)
    - PerceptionResult: perception.models
    - BeaconDetection:  perception.models
    - TrackingState:    tracking.state
    - ControlCommand:   control.command
    - CameraState:      simulation.camera.state

Models flow through the pipeline:

    Frame -> PerceptionResult -> TrackingState -> ControlCommand

with metadata carried via CameraState, DisturbanceState, MetricsSnapshot, and SessionMetadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColorModel(str, Enum):
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"


class SourceType(str, Enum):
    """Distinguishes where a Frame came from.

    The downstream pipeline MUST NOT branch on source_type for
    perception/tracking/control logic.  It is used only for:
      - metadata bookkeeping
      - benchmark reporting
      - deciding whether ground truth is available for offline evaluation
    """
    SIMULATION = "simulation"
    VIDEO = "video"
    LIVE = "live"
    DATASET = "dataset"
    SYNTHETIC = "synthetic"


# ---------------------------------------------------------------------------
# Core frame model
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """A single image frame from any source.

    The frame carries all timing information needed for downstream dt
    computation.  The pipeline (not the Frame itself) is responsible for
    computing inter-frame dt from timestamps.

    Attributes:
        image: Raw pixel data as a numpy array.
        width: Pixel width of the image.
        height: Pixel height of the image.
        channels: Number of colour channels.
        color_model: Colour space of the image data.
        source_id: Identifier for the originating source.
        frame_index: Sequential index within the source (0-based).
        timestamp_s: Wall-clock or source-relative timestamp in seconds.
        monotonic_time_s: Monotonic capture time if available (e.g. ``time.monotonic``).
        nominal_fps: The FPS the source claims, if known (may be ``None`` for variable-FPS).
        metadata: Arbitrary key-value metadata from the source.
            Must NEVER contain simulator ground truth. Use EvalSink.
    """

    image: np.ndarray
    width: int
    height: int
    channels: int
    color_model: ColorModel
    source_id: str
    source_type: SourceType
    frame_index: int
    timestamp_s: float
    monotonic_time_s: float | None = None
    nominal_fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(
                f"Image shape ({w}x{h}) does not match declared dimensions "
                f"({self.width}x{self.height})"
            )
        if self.metadata and "ground_truth" in self.metadata:
            raise ValueError(
                "Runtime Frame must not carry ground_truth; use EvalSink"
            )


# ---------------------------------------------------------------------------
# Target / beacon
# ---------------------------------------------------------------------------

@dataclass
class TargetState:
    """State of the beacon target in world or image coordinates.

    Positions are in pixels relative to the image frame unless noted.
    """

    x: float
    y: float
    visible: bool = True
    confidence: float = 1.0
    size_px: float = 10.0
    timestamp_s: float = 0.0


# ---------------------------------------------------------------------------
# Information boundary types
# ---------------------------------------------------------------------------
# Three distinct levels of truth, NEVER mixed in the same pipeline stage:
#
#   WORLD TRUTH  — simulator internals, true 3D positions, future trajectories
#                  Available ONLY via pipeline.EvalSink (never on Frame).
#
#   OBSERVATION  — what the detector/perceiver actually sees in the image
#                  (pixel coordinates, bounding boxes, confidence scores).
#                  This is the input to the tracking pipeline.
#
#   ESTIMATION   — what the Kalman filter / AI predicts after filtering
#                  (smoothed position, velocity, uncertainty).
#                  This is the output of the tracking pipeline.


@dataclass
class WorldTruth:
    """Ground truth from the simulator — NEVER available at runtime.

    This dataclass exists ONLY for:
      - Dataset generation (recording ground truth with frames)
      - Offline evaluation (computing RMSE against known positions)
      - Benchmark scoring

    Runtime perception/tracking/AI must NEVER receive this.
    """

    target_id: int = 0
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    target_visible: bool = True
    target_pixel_x: float = 0.0
    target_pixel_y: float = 0.0
    target_size_px: float = 10.0


@dataclass
class Observation:
    """What the detector actually sees — the ONLY input to tracking.

    Contains pixel-space measurements from the image processor.
    No world coordinates, no future predictions, no hidden state.
    """

    detected: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    confidence: float = 0.0
    size_px: float = 0.0
    timestamp_s: float = 0.0


@dataclass
class Estimate:
    """What the tracker predicts — filtered output, NOT ground truth.

    Contains smoothed pixel positions, velocities, and uncertainties.
    The downstream control system acts on this, never on raw observations
    or ground truth.
    """

    estimated_x: float = 0.0
    estimated_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    uncertainty_x: float = 0.0
    uncertainty_y: float = 0.0
    confidence: float = 0.0
    timestamp_s: float = 0.0
