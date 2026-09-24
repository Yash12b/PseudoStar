"""Architecture tests proving the camera-tracking-first design.

These tests verify:
1. Simulation source produces Frame with correct source_type
2. Video source conforms to same FrameSource interface
3. Downstream components (perception, tracker, controller) consume generic Frame
4. Runtime AI cannot directly receive target ground truth
5. WorldTruth, Observation, Estimate are distinct types
"""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import (
    Estimate,
    Frame,
    Observation,
    SourceType,
    WorldTruth,
)

# ---------------------------------------------------------------------------
# 1. Simulation source produces Frame
# ---------------------------------------------------------------------------

class TestSimulationSourceProducesFrame:
    """VirtualSimulationSource must yield Frame objects with source_type=SIMULATION."""

    def test_returns_frame_type(self) -> None:
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.world import WorldConfig

        wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
        engine = SimulationEngine(wc)
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0, "vx": 0.1, "vy": 0.0},
        )
        cam_state = CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
        )
        camera = VirtualCamera(cam_state)
        sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))

        source = VirtualSimulationSource(engine, camera, sensor)
        source.open()
        frame = source.read()
        source.release()

        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.source_type == SourceType.SIMULATION
        assert frame.source_id == "simulation"
        assert frame.width == 640
        assert frame.height == 480

    def test_is_framesource_interface(self) -> None:
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        assert issubclass(VirtualSimulationSource, FrameSource)

    def test_gt_excluded_by_default(self) -> None:
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.world import WorldConfig

        wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
        engine = SimulationEngine(wc)
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0, "vx": 0.1, "vy": 0.0},
        )
        cam_state = CameraState(horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=640, height=480)
        camera = VirtualCamera(cam_state)
        sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))

        source = VirtualSimulationSource(engine, camera, sensor, ground_truth_in_metadata=False)
        source.open()
        frame = source.read()
        source.release()

        assert frame is not None
        assert "ground_truth" not in frame.metadata

    def test_eval_sink_not_frame(self) -> None:
        from fsoc_tracker.pipeline.eval import EvalSink
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.world import WorldConfig

        wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
        engine = SimulationEngine(wc)
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0, "vx": 0.1, "vy": 0.0},
        )
        cam_state = CameraState(horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=640, height=480)
        camera = VirtualCamera(cam_state)
        sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))
        sink = EvalSink()
        source = VirtualSimulationSource(engine, camera, sensor, eval_sink=sink)
        source.open()
        frame = source.read()
        source.release()

        assert frame is not None
        assert "ground_truth" not in frame.metadata
        assert len(sink) == 1
        assert sink.primary_for(0) is not None

    def test_gt_on_frame_rejected(self) -> None:

        wc_setup = self  # unused placeholder to keep class grouping
        import numpy as np

        from fsoc_tracker.core.models import ColorModel, Frame, SourceType
        img = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="ground_truth"):
            Frame(
                image=img, width=8, height=8, channels=1,
                color_model=ColorModel.GRAY, source_id="t",
                source_type=SourceType.SIMULATION, frame_index=0,
                timestamp_s=0.0, metadata={"ground_truth": object()},
            )

    def test_positional_seed_cannot_enable_gt(self) -> None:
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.world import WorldConfig

        wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
        engine = SimulationEngine(wc)
        cam = VirtualCamera(CameraState(width=64, height=48, horizontal_fov_deg=4.0, vertical_fov_deg=3.0))
        sensor = VirtualSensorRenderer(SensorConfig(width=64, height=48))
        with pytest.raises(TypeError):
            VirtualSimulationSource(engine, cam, sensor, None, 1.0 / 30.0, 42)


# ---------------------------------------------------------------------------
# 2. Video source conforms to same interface
# ---------------------------------------------------------------------------

class TestVideoSourceInterface:
    """VideoSource must implement FrameSource and produce Frame with source_type=VIDEO."""

    def test_is_framesource(self) -> None:
        from fsoc_tracker.pipeline.sources import VideoSource
        assert issubclass(VideoSource, FrameSource)

    def test_produces_frame_with_video_source_type(self) -> None:
        import os
        import tempfile

        import cv2

        # Create a tiny test video
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (64, 48))
            for i in range(5):
                img = np.zeros((48, 64, 3), dtype=np.uint8)
                cv2.circle(img, (32, 24), 5, (255, 255, 255), -1)
                out.write(img)
            out.release()

            from fsoc_tracker.pipeline.sources import VideoSource
            source = VideoSource(tmp_path)
            source.open()
            frame = source.read()
            source.release()

            assert frame is not None
            assert isinstance(frame, Frame)
            assert frame.source_type == SourceType.VIDEO
            assert frame.width == 64
            assert frame.height == 48
            assert "ground_truth" not in frame.metadata
        finally:
            os.unlink(tmp_path)

    def test_all_sources_produce_same_frame_type(self) -> None:
        """All FrameSource subclasses produce the same Frame dataclass."""
        from fsoc_tracker.pipeline.sources import (
            DatasetSource,
            LiveSource,
            VideoSource,
            VirtualSimulationSource,
        )
        for cls in [VirtualSimulationSource, VideoSource, LiveSource, DatasetSource]:
            assert issubclass(cls, FrameSource)


# ---------------------------------------------------------------------------
# 3. Downstream components consume generic Frame
# ---------------------------------------------------------------------------

class TestDownstreamConsumesGenericFrame:
    """Perception, tracking, and control must work with any Frame."""

    def test_detector_accepts_frame_image(self) -> None:
        """ClassicalBeaconDetector.detect() takes image, not Frame."""
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        detector = ClassicalBeaconDetector()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[230:250, 310:330] = (255, 255, 255)

        result = detector.detect(image, timestamp_s=0.0, frame_index=0)
        assert result is not None
        assert hasattr(result, "detected")

    def test_tracker_works_with_detection_list(self) -> None:
        """KalmanTracker.update() takes detection list + timestamp, not Frame."""
        from fsoc_tracker.tracking.tracker import KalmanTracker
        tracker = KalmanTracker()

        # No detections — should still work
        state = tracker.update([], timestamp_s=0.0)
        assert state is not None
        assert hasattr(state, "state")

    def test_controller_works_with_tracking_state(self) -> None:
        """CoarsePointingController.compute() takes TrackingState + intrinsics, not Frame."""
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.simulation.camera.state import CameraIntrinsics
        from fsoc_tracker.tracking.tracker import KalmanTracker

        tracker = KalmanTracker()
        controller = CoarsePointingController()
        intrinsics = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)

        # Update tracker with no detections
        trk_state = tracker.update([], timestamp_s=0.0)
        cmd, _ = controller.compute(trk_state, intrinsics, dt=0.033, timestamp_s=0.0)

        assert cmd is not None
        assert hasattr(cmd, "pan_rate_deg_s")
        assert hasattr(cmd, "tilt_rate_deg_s")


# ---------------------------------------------------------------------------
# 4. Runtime AI cannot receive ground truth
# ---------------------------------------------------------------------------

class TestRuntimeAINoGroundTruth:
    """BeaconAI and MissionBrain must consume only observable quantities."""

    def test_beacon_ai_update_signature(self) -> None:
        """FSocBeaconAI.update() takes TrackingState + intrinsics, not world targets."""
        import inspect

        from fsoc_tracker.ai.beacon_ai import FSocBeaconAI

        sig = inspect.signature(FSocBeaconAI.update)
        params = list(sig.parameters.keys())

        # Must NOT have world_targets parameter
        assert "world_targets" not in params
        assert "targets" not in params

        # Must have tracking/intrinsics parameters
        assert "tracking_state" in params
        assert "intrinsics" in params
        assert "detected" in params
        assert "estimated_x" in params
        assert "estimated_y" in params

    def test_beacon_ai_has_no_world_position(self) -> None:
        """BeaconAI's BeaconState must not have world_x/y/z fields."""
        from fsoc_tracker.ai.beacon_ai import BeaconState
        fields = [f.name for f in BeaconState.__dataclass_fields__.values()]
        assert "world_x" not in fields
        assert "world_y" not in fields
        assert "world_z" not in fields

    def test_mission_brain_uses_observations_only(self) -> None:
        """AIMissionBrain.decide() takes MissionObservation, not WorldTruth."""
        from fsoc_tracker.ai.mission import AIMissionBrain, MissionObservation, ObservationFeatures
        brain = AIMissionBrain()

        obs = MissionObservation(
            features=ObservationFeatures(
                timestamp_s=0.0,
                detected=True,
                confidence=0.9,
                residual_px=5.0,
                uncertainty_x_px=10.0,
                uncertainty_y_px=10.0,
                velocity_x_px_s=15.0,
                velocity_y_px_s=0.0,
                distance_from_center_px=5.0,
                time_since_detection_s=0.033,
                latency_ms=10.0,
                source_fps=30.0,
                processing_fps=25.0,
                candidate_count=1,
            ),
            camera_pan_deg=0.0,
            camera_tilt_deg=0.0,
        )
        decision = brain.decide(obs)
        assert decision is not None
        assert hasattr(decision, "situation")
        assert hasattr(decision, "action")


# ---------------------------------------------------------------------------
# 5. Information boundary types
# ---------------------------------------------------------------------------

class TestInformationBoundaryTypes:
    """WorldTruth, Observation, and Estimate must be distinct types."""

    def test_world_truth_is_separate_type(self) -> None:
        assert WorldTruth is not Observation
        assert WorldTruth is not Estimate

    def test_observation_has_pixel_fields(self) -> None:
        obs = Observation(detected=True, center_x=320.0, center_y=240.0, confidence=0.9)
        assert obs.detected is True
        assert obs.center_x == 320.0
        assert obs.center_y == 240.0

    def test_estimate_has_uncertainty(self) -> None:
        est = Estimate(estimated_x=320.0, estimated_y=240.0, uncertainty_x=5.0, uncertainty_y=5.0)
        assert est.uncertainty_x == 5.0
        assert est.uncertainty_y == 5.0

    def test_world_truth_has_world_coords(self) -> None:
        gt = WorldTruth(world_x=1000.0, world_y=1000.0, world_z=500.0)
        assert gt.world_x == 1000.0
        assert gt.world_z == 500.0

    def test_frame_source_type_is_enum(self) -> None:
        assert SourceType.SIMULATION.value == "simulation"
        assert SourceType.VIDEO.value == "video"
        assert SourceType.LIVE.value == "live"
        assert SourceType.DATASET.value == "dataset"
        assert SourceType.SYNTHETIC.value == "synthetic"
