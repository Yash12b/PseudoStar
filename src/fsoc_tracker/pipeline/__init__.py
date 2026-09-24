"""Pipeline package — authoritative system integration layer.

Provides TrackingPipeline (the single runtime pipeline), mode-specific
source adapters, SessionController, and run-manager.
"""

from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.pipeline.session import RunState, SessionController
from fsoc_tracker.pipeline.sources import (
    DatasetSource,
    FrameSource,
    LiveSource,
    SimulationSource,
    VideoSource,
    VirtualSimulationSource,
)

__all__ = [
    "TrackingPipeline",
    "SessionController",
    "RunState",
    "FrameSource",
    "SimulationSource",
    "VirtualSimulationSource",
    "VideoSource",
    "LiveSource",
    "DatasetSource",
]
