"""Comprehensive MP4 mode tests.

Tests for:
- MP4 loading (VideoSource)
- Frame decoding and timestamps
- Seek functionality
- Frame rate handling (arbitrary FPS)
- Frame source integration
- Tracker processing on video frames
- Equirectangular detection and viewport extraction
- Worker VIDEO mode pipeline
- Distance N/A reporting for monocular
- End-to-end pipeline with sample video
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import cv2
import numpy as np
import pytest

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType
from fsoc_tracker.pipeline.sources import EquirectangularFrameSource, VideoSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_video(
    path: str,
    num_frames: int = 30,
    fps: float = 30.0,
    width: int = 64,
    height: int = 48,
    motion: str = "static",
) -> str:
    """Create a test MP4 video with a bright beacon spot."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background noise
        img[:] = (20, 20, 20)
        # Moving beacon spot
        if motion == "static":
            cx, cy = width // 2, height // 2
        elif motion == "linear":
            cx = int(width * 0.2 + (width * 0.6) * i / max(num_frames - 1, 1))
            cy = int(height * 0.3 + (height * 0.4) * i / max(num_frames - 1, 1))
        elif motion == "circular":
            angle = 2 * math.pi * i / max(num_frames, 1)
            cx = int(width / 2 + width * 0.2 * math.cos(angle))
            cy = int(height / 2 + height * 0.2 * math.sin(angle))
        else:
            cx, cy = width // 2, height // 2
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        out.write(img)
    out.release()
    return path


def _create_equirectangular_video(
    path: str,
    num_frames: int = 10,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 640,
) -> str:
    """Create a 2:1 aspect ratio equirectangular test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Place a bright spot at ~center (longitude=0, latitude=0)
        cx = width // 2
        cy = height // 2
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
        # Add another spot at ~90 deg longitude
        cv2.circle(img, (width // 4, height // 2), 6, (200, 200, 0), -1)
        out.write(img)
    out.release()
    return path


@pytest.fixture
def test_video():
    """Create a temporary test video and clean up after."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=30, fps=30.0, motion="linear")
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def test_video_60fps():
    """Create a 60 FPS test video."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=60.0, motion="circular")
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def equirect_video():
    """Create a 2:1 equirectangular test video."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_equirectangular_video(tmp.name, num_frames=10, fps=30.0)
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# 1. MP4 Loading Tests
# ---------------------------------------------------------------------------

class TestMP4Loading:
    def test_is_framesource(self):
        assert issubclass(VideoSource, FrameSource)

    def test_open_and_read(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.is_open()
        frame = src.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        src.release()
        assert not src.is_open()

    def test_missing_file_raises(self):
        src = VideoSource("/nonexistent/path/video.mp4")
        with pytest.raises((RuntimeError, FileNotFoundError)):
            src.open()

    def test_read_after_release_returns_none(self, test_video):
        src = VideoSource(test_video)
        src.open()
        src.release()
        assert src.read() is None

    def test_source_id_is_path(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.source_id == test_video
        src.release()

    def test_source_type_is_video(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert frame.source_type == SourceType.VIDEO
        src.release()


# ---------------------------------------------------------------------------
# 2. Frame Decoding Tests
# ---------------------------------------------------------------------------

class TestFrameDecoding:
    def test_frame_dimensions(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert frame.width == 64
        assert frame.height == 48
        assert frame.channels == 3
        assert frame.color_model == ColorModel.BGR
        src.release()

    def test_frame_image_is_valid(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert frame.image.shape == (48, 64, 3)
        assert frame.image.dtype == np.uint8
        src.release()

    def test_read_all_frames(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frames = []
        while True:
            f = src.read()
            if f is None:
                break
            frames.append(f)
        assert len(frames) == 30
        src.release()

    def test_end_of_stream_returns_none(self, test_video):
        src = VideoSource(test_video)
        src.open()
        for _ in range(35):  # Read past end
            src.read()
        assert src.read() is None
        src.release()

    def test_metadata_contains_video_path(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert "video_path" in frame.metadata
        assert frame.metadata["video_path"] == test_video
        src.release()

    def test_metadata_contains_timestamp_source(self, test_video):
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert "timestamp_source" in frame.metadata
        assert frame.metadata["timestamp_source"] in (
            "decoder_position", "nominal_fps", "frame_index_unknown_rate"
        )
        src.release()


# ---------------------------------------------------------------------------
# 3. Timestamp Tests
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamps_are_monotonic(self, test_video):
        src = VideoSource(test_video)
        src.open()
        timestamps = []
        for _ in range(10):
            f = src.read()
            if f is None:
                break
            timestamps.append(f.timestamp_s)
        src.release()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_timestamps_at_30fps(self, test_video):
        src = VideoSource(test_video)
        src.open()
        f1 = src.read()
        f2 = src.read()
        src.release()
        assert f1 is not None and f2 is not None
        dt = f2.timestamp_s - f1.timestamp_s
        assert 0.02 < dt < 0.05  # ~33ms ± tolerance

    def test_timestamps_at_60fps(self, test_video_60fps):
        src = VideoSource(test_video_60fps)
        src.open()
        f1 = src.read()
        f2 = src.read()
        src.release()
        assert f1 is not None and f2 is not None
        dt = f2.timestamp_s - f1.timestamp_s
        assert 0.01 < dt < 0.03  # ~16.7ms ± tolerance

    def test_nominal_fps_property(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.nominal_fps == pytest.approx(30.0, abs=1.0)
        src.release()

    def test_frame_count_property(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.frame_count == 30
        src.release()

    def test_duration_property(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.duration_s is not None
        assert src.duration_s == pytest.approx(1.0, abs=0.1)
        src.release()


# ---------------------------------------------------------------------------
# 4. Seek Tests
# ---------------------------------------------------------------------------

class TestSeek:
    def test_seek_forward(self, test_video):
        src = VideoSource(test_video)
        src.open()
        # Read first frame
        f0 = src.read()
        assert f0 is not None
        # Seek to frame 10
        result = src.seek(10)
        assert result is True
        f10 = src.read()
        assert f10 is not None
        assert f10.frame_index == 10
        src.release()

    def test_seek_backward(self, test_video):
        src = VideoSource(test_video)
        src.open()
        # Read to frame 15
        for _ in range(15):
            src.read()
        # Seek back to frame 5
        result = src.seek(5)
        assert result is True
        f5 = src.read()
        assert f5 is not None
        assert f5.frame_index == 5
        src.release()

    def test_seek_to_start(self, test_video):
        src = VideoSource(test_video)
        src.open()
        # Read some frames
        for _ in range(10):
            src.read()
        # Seek back to start
        result = src.seek(0)
        assert result is True
        f0 = src.read()
        assert f0 is not None
        assert f0.frame_index == 0
        src.release()

    def test_seek_resets_timestamp(self, test_video):
        src = VideoSource(test_video)
        src.open()
        for _ in range(10):
            src.read()
        src.seek(0)
        f = src.read()
        assert f is not None
        assert f.timestamp_s < 0.1  # Should be near start
        src.release()


# ---------------------------------------------------------------------------
# 5. Arbitrary Frame Rate Tests
# ---------------------------------------------------------------------------

class TestArbitraryFrameRate:
    def test_1fps_video(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            _create_test_video(tmp.name, num_frames=5, fps=1.0)
            src = VideoSource(tmp.name)
            src.open()
            assert src.nominal_fps == pytest.approx(1.0, abs=0.5)
            f1 = src.read()
            f2 = src.read()
            assert f1 is not None and f2 is not None
            dt = f2.timestamp_s - f1.timestamp_s
            assert dt > 0.5  # ~1 second between frames
            src.release()
        finally:
            os.unlink(tmp.name)

    def test_24fps_video(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            _create_test_video(tmp.name, num_frames=24, fps=24.0)
            src = VideoSource(tmp.name)
            src.open()
            assert src.nominal_fps == pytest.approx(24.0, abs=1.0)
            f1 = src.read()
            f2 = src.read()
            dt = f2.timestamp_s - f1.timestamp_s
            assert 0.03 < dt < 0.06  # ~41.7ms
            src.release()
        finally:
            os.unlink(tmp.name)

    def test_120fps_video(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            _create_test_video(tmp.name, num_frames=20, fps=120.0)
            src = VideoSource(tmp.name)
            src.open()
            assert src.nominal_fps == pytest.approx(120.0, abs=5.0)
            f1 = src.read()
            f2 = src.read()
            dt = f2.timestamp_s - f1.timestamp_s
            assert 0.005 < dt < 0.02  # ~8.3ms
            src.release()
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# 6. Frame Source Integration Tests
# ---------------------------------------------------------------------------

class TestFrameSourceIntegration:
    def test_context_manager(self, test_video):
        with VideoSource(test_video) as src:
            src.open()
            frame = src.read()
            assert frame is not None
        assert not src.is_open()

    def test_iterator_protocol(self, test_video):
        src = VideoSource(test_video)
        src.open()
        count = 0
        for frame in src:
            assert isinstance(frame, Frame)
            count += 1
            if count >= 5:
                break
        src.release()
        assert count == 5

    def test_get_info(self, test_video):
        src = VideoSource(test_video)
        src.open()
        info = src.get_info()
        assert info["source_type"] == "video"
        assert info["width"] == 64
        assert info["height"] == 48
        assert info["fps"] is not None
        assert info["total_frames"] == 30
        assert "duration_s" in info
        assert "is_equirectangular" in info
        src.release()

    def test_width_height_properties(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.width == 64
        assert src.height == 48
        src.release()

    def test_is_equirectangular_false_for_standard(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.is_equirectangular is False
        src.release()


# ---------------------------------------------------------------------------
# 7. Tracker Processing Tests
# ---------------------------------------------------------------------------

class TestTrackerProcessing:
    def test_detector_on_video_frame(self, test_video):
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        src = VideoSource(test_video)
        src.open()
        detector = ClassicalBeaconDetector()
        frame = src.read()
        result = detector.detect(frame.image, frame.timestamp_s, frame.frame_index)
        # Should not crash; detection result is valid
        assert result is not None
        assert hasattr(result, 'detected')
        assert hasattr(result, 'primary_detection')
        src.release()

    def test_tracker_on_video_sequence(self, test_video):
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.tracking.tracker import KalmanTracker
        src = VideoSource(test_video)
        src.open()
        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()
        states = []
        for _ in range(15):
            f = src.read()
            if f is None:
                break
            result = detector.detect(f.image, f.timestamp_s, f.frame_index)
            dets = result.detections if result.primary_detection and result.primary_detection.detected else []
            state = tracker.update(dets, f.timestamp_s)
            states.append(state)
        src.release()
        assert len(states) == 15
        # All states should have valid tracking state
        for s in states:
            assert hasattr(s, 'state')
            assert hasattr(s, 'estimated_x')
            assert hasattr(s, 'estimated_y')

    def test_full_pipeline_video(self, test_video):
        """Run perception→tracking→control on video frames."""
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.tracking.tracker import KalmanTracker

        cam_state = CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=64, height=48,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
        )
        camera = VirtualCamera(cam_state)
        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()
        controller = CoarsePointingController()

        src = VideoSource(test_video)
        src.open()
        results = []
        for _ in range(10):
            f = src.read()
            if f is None:
                break
            det = detector.detect(f.image, f.timestamp_s, f.frame_index)
            dets = det.detections if det.primary_detection and det.primary_detection.detected else []
            trk = tracker.update(dets, f.timestamp_s)
            cmd, _ = controller.compute(trk, camera.intrinsics, 1.0 / 30.0, f.timestamp_s)
            results.append({
                "detected": det.detected,
                "tracking_state": trk.state.name,
                "pan_rate": cmd.pan_rate_deg_s,
                "tilt_rate": cmd.tilt_rate_deg_s,
            })
        src.release()
        assert len(results) == 10
        # Pipeline should produce valid commands
        for r in results:
            assert isinstance(r["tracking_state"], str)
            assert isinstance(r["pan_rate"], float)


# ---------------------------------------------------------------------------
# 8. Equirectangular Detection Tests
# ---------------------------------------------------------------------------

class TestEquirectangularDetection:
    def test_2_1_ratio_detected(self, equirect_video):
        src = VideoSource(equirect_video)
        src.open()
        assert src.is_equirectangular is True
        src.release()

    def test_standard_ratio_not_detected(self, test_video):
        src = VideoSource(test_video)
        src.open()
        assert src.is_equirectangular is False
        src.release()

    def test_equirect_metadata_in_frame(self, equirect_video):
        src = VideoSource(equirect_video)
        src.open()
        frame = src.read()
        assert frame.metadata["is_equirectangular"] is True
        src.release()

    def test_equirect_info(self, equirect_video):
        src = VideoSource(equirect_video)
        src.open()
        info = src.get_info()
        assert info["is_equirectangular"] is True
        assert info["width"] == 1280
        assert info["height"] == 640
        src.release()


# ---------------------------------------------------------------------------
# 9. Equirectangular Viewport Extraction Tests
# ---------------------------------------------------------------------------

class TestEquirectangularViewport:
    def test_viewport_output_dimensions(self, equirect_video):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam_state = CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=640, height=480,
        )
        camera = VirtualCamera(cam_state)
        src = VideoSource(equirect_video)
        src.open()
        eq_src = EquirectangularFrameSource(
            src, camera, output_width=320, output_height=240,
            hfov_deg=4.0, vfov_deg=3.0,
        )
        eq_src.open()
        frame = eq_src.read()
        assert frame is not None
        assert frame.width == 320
        assert frame.height == 240
        eq_src.release()

    def test_viewport_is_perspective(self, equirect_video):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam_state = CameraState(
            horizontal_fov_deg=90.0, vertical_fov_deg=60.0,
            width=640, height=480,
        )
        camera = VirtualCamera(cam_state)
        src = VideoSource(equirect_video)
        src.open()
        eq_src = EquirectangularFrameSource(
            src, camera, output_width=320, output_height=240,
            hfov_deg=90.0, vfov_deg=60.0,
        )
        eq_src.open()
        frame = eq_src.read()
        assert frame is not None
        assert frame.metadata["equirectangular_viewport"] is True
        assert "viewport_pan_deg" in frame.metadata
        assert "viewport_tilt_deg" in frame.metadata
        eq_src.release()

    def test_viewport_changes_with_pan(self, equirect_video):
        """Viewport at pan=0 and pan=90 should produce different images."""
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam1 = VirtualCamera(CameraState(
            horizontal_fov_deg=90.0, vertical_fov_deg=60.0, width=320, height=240,
        ))
        cam2 = VirtualCamera(CameraState(
            horizontal_fov_deg=90.0, vertical_fov_deg=60.0, width=320, height=240,
        ))
        # Directly set pan angles (camera state must be set at init)
        cam1.state.pan_deg = 0.0
        cam2.state.pan_deg = 90.0

        src = VideoSource(equirect_video)
        src.open()

        eq1 = EquirectangularFrameSource(src, cam1, 320, 240, 90.0, 60.0)
        eq1.open()
        f1 = eq1.read()

        src2 = VideoSource(equirect_video)
        src2.open()
        eq2 = EquirectangularFrameSource(src2, cam2, 320, 240, 90.0, 60.0)
        eq2.open()
        f2 = eq2.read()

        assert f1 is not None and f2 is not None
        # Different pan angles should produce different viewport images
        assert not np.array_equal(f1.image, f2.image)
        eq1.release()
        eq2.release()

    def test_viewport_source_id(self, equirect_video):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=320, height=240,
        ))
        src = VideoSource(equirect_video)
        src.open()
        eq_src = EquirectangularFrameSource(src, cam, 320, 240, 4.0, 3.0)
        eq_src.open()
        assert eq_src.source_id.startswith("equirect:")
        eq_src.release()

    def test_viewport_seek(self, equirect_video):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=320, height=240,
        ))
        src = VideoSource(equirect_video)
        src.open()
        eq_src = EquirectangularFrameSource(src, cam, 320, 240, 4.0, 3.0)
        eq_src.open()
        # Read a few frames
        eq_src.read()
        eq_src.read()
        # Seek
        result = eq_src.seek(0)
        assert result is True
        f = eq_src.read()
        assert f is not None
        assert f.frame_index == 0
        eq_src.release()

    def test_viewport_frame_count(self, equirect_video):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=320, height=240,
        ))
        src = VideoSource(equirect_video)
        src.open()
        eq_src = EquirectangularFrameSource(src, cam, 320, 240, 4.0, 3.0)
        eq_src.open()
        assert eq_src.frame_count == 10
        eq_src.release()


# ---------------------------------------------------------------------------
# 10. Worker VIDEO Mode Integration Tests
# ---------------------------------------------------------------------------

class TestWorkerVideoMode:
    def test_worker_initializes_with_video(self, test_video):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": test_video,
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        assert worker._source is not None
        assert worker._mode.value == "video"
        worker._release_resources()

    def test_worker_video_state_metadata(self, test_video):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": test_video,
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        s = worker.view_state
        assert s.video_total_frames == 30
        assert s.video_fps > 0
        assert s.video_distance_status == "N/A"
        worker._release_resources()

    def test_worker_no_ground_truth_in_video(self, test_video):
        """VIDEO mode must not inject ground truth into frames."""
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": test_video,
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        frame = worker._source.read()
        assert frame is not None
        assert "ground_truth" not in frame.metadata
        worker._release_resources()

    def test_worker_video_seek(self, test_video):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": test_video,
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        # Read a few frames
        worker._source.read()
        worker._source.read()
        # Seek
        result = worker.seek_video(5)
        assert result is True
        worker._release_resources()

    def test_worker_equirectangular_detection(self, equirect_video):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": equirect_video,
            "camera_width": 320,
            "camera_height": 240,
            "hfov": 90.0,
            "vfov": 60.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        # Should auto-detect equirectangular and wrap source
        assert isinstance(worker._source, EquirectangularFrameSource)
        assert worker._state.video_is_equirectangular is True
        worker._release_resources()


# ---------------------------------------------------------------------------
# 11. End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_full_pipeline_with_video(self, test_video):
        """Run complete perception→tracking→control pipeline on video."""
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.tracking.tracker import KalmanTracker

        cam_state = CameraState(
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
            width=64, height=48,
            max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
        )
        camera = VirtualCamera(cam_state)
        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()
        controller = CoarsePointingController()

        src = VideoSource(test_video)
        src.open()

        tracking_states = []
        errors = []
        for i in range(30):
            f = src.read()
            if f is None:
                break
            det = detector.detect(f.image, f.timestamp_s, f.frame_index)
            dets = det.detections if det.primary_detection and det.primary_detection.detected else []
            trk = tracker.update(dets, f.timestamp_s)
            cmd, _ = controller.compute(trk, camera.intrinsics, 1.0 / 30.0, f.timestamp_s)

            # Apply camera motion
            new_pan = camera.state.pan_deg + cmd.pan_rate_deg_s * (1.0 / 30.0)
            new_tilt = camera.state.tilt_deg + cmd.tilt_rate_deg_s * (1.0 / 30.0)
            camera.set_target_pan_tilt(new_pan, new_tilt)
            camera.update(1.0 / 30.0)

            tracking_states.append(trk.state.name)
            if det.detected:
                # Compute detection error from image center
                err = math.sqrt(
                    (det.primary_detection.center_x - 32) ** 2 +
                    (det.primary_detection.center_y - 24) ** 2
                )
                errors.append(err)

        src.release()
        assert len(tracking_states) == 30
        # Pipeline should produce valid tracking states
        valid_states = {"TRACKING", "ACQUIRING", "LOST", "NO_TRACK"}
        for s in tracking_states:
            assert s in valid_states

    def test_pipeline_handles_end_of_video(self, test_video):
        """Pipeline should gracefully handle video reaching end."""
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.tracking.tracker import KalmanTracker

        src = VideoSource(test_video)
        src.open()
        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()

        frames_processed = 0
        while True:
            f = src.read()
            if f is None:
                break
            det = detector.detect(f.image, f.timestamp_s, f.frame_index)
            dets = det.detections if det.primary_detection and det.primary_detection.detected else []
            tracker.update(dets, f.timestamp_s)
            frames_processed += 1

        src.release()
        assert frames_processed == 30

    def test_pipeline_performance(self, test_video):
        """Pipeline should process video frames within reasonable time."""
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.tracking.tracker import KalmanTracker

        src = VideoSource(test_video)
        src.open()
        detector = ClassicalBeaconDetector()
        tracker = KalmanTracker()

        t0 = time.perf_counter()
        count = 0
        while True:
            f = src.read()
            if f is None:
                break
            det = detector.detect(f.image, f.timestamp_s, f.frame_index)
            dets = det.detections if det.primary_detection and det.primary_detection.detected else []
            tracker.update(dets, f.timestamp_s)
            count += 1
        elapsed = time.perf_counter() - t0
        src.release()

        assert count == 30
        # 30 frames should process in under 5 seconds
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s for {count} frames"


# ---------------------------------------------------------------------------
# 12. Distance N/A Tests
# ---------------------------------------------------------------------------

class TestDistanceNA:
    def test_video_distance_status_n_a(self, test_video):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "video",
            "video_path": test_video,
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        assert worker._state.video_distance_status == "N/A"
        worker._release_resources()

    def test_video_distance_not_in_frame_metadata(self, test_video):
        """Video frames must not contain distance information."""
        src = VideoSource(test_video)
        src.open()
        frame = src.read()
        assert "distance_m" not in frame.metadata
        assert "beacon_distance" not in frame.metadata
        src.release()
