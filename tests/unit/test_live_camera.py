"""Tests for the LiveSource abstraction with mock camera."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fsoc_tracker.core.models import Frame, SourceType
from fsoc_tracker.pipeline.sources import LiveSource


class MockVideoCapture:
    """Mock OpenCV VideoCapture for testing."""
    
    def __init__(self, camera_id: int = 0) -> None:
        self._camera_id = camera_id
        self._is_opened = True
        self._frame_count = 0
        self._width = 640
        self._height = 480
    
    def isOpened(self) -> bool:
        return self._is_opened
    
    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._is_opened:
            return False, None
        
        # Create a mock frame
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_count += 1
        return True, frame
    
    def get(self, prop: int) -> float:
        # CAP_PROP_FRAME_WIDTH = 3
        if prop == 3:
            return float(self._width)
        # CAP_PROP_FRAME_HEIGHT = 4
        elif prop == 4:
            return float(self._height)
        return 0.0
    
    def release(self) -> None:
        self._is_opened = False


class TestLiveSource:
    """Tests for the LiveSource implementation."""
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_open_and_read(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        assert source.is_open()
        frame = source.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        source.release()
        assert not source.is_open()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_read_returns_none_when_closed(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        source.release()
        frame = source.read()
        assert frame is None
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_frame_dimensions_match_capture(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        frame = source.read()
        assert frame is not None
        assert frame.width == 640
        assert frame.height == 480
        assert frame.image.shape == (480, 640, 3)
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_frame_index_increments(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        for i in range(5):
            frame = source.read()
            assert frame is not None
            assert frame.frame_index == i
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_timestamps_increase(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        prev_ts = -1.0
        for _ in range(5):
            frame = source.read()
            assert frame is not None
            assert frame.timestamp_s >= prev_ts
            prev_ts = frame.timestamp_s
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_source_type_is_live(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        frame = source.read()
        assert frame is not None
        assert frame.source_type == SourceType.LIVE
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_source_id(self) -> None:
        source = LiveSource(camera_id=0)
        assert source.source_id == "camera_0"
        source = LiveSource(camera_id=3)
        assert source.source_id == "camera_3"
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_get_info(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        info = source.get_info()
        assert info["source_type"] == "live"
        assert info["camera_id"] == 0
        assert info["width"] > 0
        assert info["height"] > 0
        assert info["actual_fps"] >= 0
        assert info["is_available"] is True
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_actual_fps_tracking(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        # Read a few frames to get FPS measurement
        for _ in range(3):
            source.read()
        fps = source.actual_fps
        assert fps >= 0
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_error_message_on_open(self) -> None:
        source = LiveSource(camera_id=999)
        # Mock the VideoCapture to report not opened
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_cap.return_value = mock_instance
            with pytest.raises(RuntimeError, match="Cannot open camera"):
                source.open()
            assert "Cannot open camera" in source.error_message
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_metadata_contains_camera_id(self) -> None:
        source = LiveSource(camera_id=5)
        source.open()
        frame = source.read()
        assert frame is not None
        assert "camera_id" in frame.metadata
        assert frame.metadata["camera_id"] == 5
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_metadata_contains_fps(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        frame = source.read()
        assert frame is not None
        assert "actual_fps" in frame.metadata
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_metadata_contains_dimensions(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        frame = source.read()
        assert frame is not None
        assert "width" in frame.metadata
        assert "height" in frame.metadata
        source.release()
    
    def test_enumerate_cameras(self) -> None:
        # Mock cv2.VideoCapture for enumeration test
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock first camera as available, second as not
            mock_instance1 = MagicMock()
            mock_instance1.isOpened.return_value = True
            mock_instance1.get.return_value = 640.0
            
            mock_instance2 = MagicMock()
            mock_instance2.isOpened.return_value = False
            
            mock_cap.side_effect = [mock_instance1, mock_instance2]
            
            cameras = LiveSource.enumerate_cameras(max_devices=2)
            assert len(cameras) == 2
            assert cameras[0]["camera_id"] == 0
            assert cameras[0]["is_available"] is True
            assert cameras[1]["camera_id"] == 1
            assert cameras[1]["is_available"] is False
    
    def test_is_camera_available(self) -> None:
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance
            
            assert LiveSource.is_camera_available(0) is True
            
            mock_instance.isOpened.return_value = False
            assert LiveSource.is_camera_available(0) is False
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_context_manager(self) -> None:
        with LiveSource(camera_id=0) as source:
            assert source.is_open()
            frame = source.read()
            assert frame is not None
        assert not source.is_open()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_multiple_cameras(self) -> None:
        source1 = LiveSource(camera_id=0)
        source2 = LiveSource(camera_id=1)
        source1.open()
        source2.open()
        
        frame1 = source1.read()
        frame2 = source2.read()
        
        assert frame1 is not None
        assert frame2 is not None
        assert frame1.metadata["camera_id"] == 0
        assert frame2.metadata["camera_id"] == 1
        
        source1.release()
        source2.release()
        assert not source1.is_open()
        assert not source2.is_open()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_width_height_properties(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        # Actual dimensions come from the capture device, not constructor
        assert source.width == 640
        assert source.height == 480
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_nominal_fps_is_none(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        assert source.nominal_fps is None
        source.release()
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    def test_read_preserves_color_model(self) -> None:
        source = LiveSource(camera_id=0)
        source.open()
        frame = source.read()
        assert frame is not None
        assert frame.color_model.value == "BGR"
        assert frame.channels == 3
        source.release()

class _GradientSource:
    """Hardware-free synthetic feed: horizontal gradient, 1280x720."""

    def __init__(self) -> None:
        self._i = 0
        self._opened = False

    @property
    def source_id(self) -> str:
        return "gradient"

    @property
    def nominal_fps(self):
        return 30.0

    def open(self) -> None:
        self._opened = True

    def is_open(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False

    def read(self):
        if not self._opened:
            return None
        import numpy as np

        from fsoc_tracker.core.models import ColorModel, Frame, SourceType
        w, h = 1280, 720
        row = np.linspace(0, 255, w, dtype=np.uint8)
        img = np.tile(row, (h, 1))
        f = Frame(image=img, width=w, height=h, channels=1,
                  color_model=ColorModel.GRAY, source_id="gradient",
                  source_type=SourceType.LIVE, frame_index=self._i,
                  timestamp_s=self._i / 30.0, nominal_fps=30.0, metadata={})
        self._i += 1
        return f


class TestLiveViewportSource:
    def test_output_size_constant(self):
        from fsoc_tracker.pipeline.sources import LiveViewportSource
        src = LiveViewportSource(_GradientSource(), 640, 480,
                                 hfov_deg=4.0, vfov_deg=3.0)
        src.open()
        f = src.read()
        assert f is not None
        assert (f.width, f.height) == (640, 480)
        assert f.timestamp_s == 0.0
        assert f.metadata["viewport_active"] is True
        src.release()

    def test_pan_moves_window(self):
        from fsoc_tracker.pipeline.sources import LiveViewportSource
        src = LiveViewportSource(_GradientSource(), 640, 480,
                                 hfov_deg=4.0, vfov_deg=3.0)
        src.open()
        src.set_viewport_center(0.0, 0.0)
        left = src.read().image[:, 0].astype(float).mean()
        src.set_viewport_center(1.0, 0.0)  # +1 deg right
        right = src.read().image[:, 0].astype(float).mean()
        # Gradient rises left->right: rightward pan sees brighter pixels.
        # 640px/4deg = 160 px/deg shift -> +160/1280*255 ~= +32 levels.
        assert right > left + 20.0
        assert src.read().metadata["viewport_pan_deg"] == 1.0
        src.release()

    def test_viewport_clamps_at_edges(self):
        from fsoc_tracker.pipeline.sources import LiveViewportSource
        src = LiveViewportSource(_GradientSource(), 640, 480)
        src.open()
        src.set_viewport_center(90.0, 90.0)  # far outside: must clamp
        f = src.read()
        assert f is not None
        assert (f.width, f.height) == (640, 480)
        src.release()

    def test_timestamps_pass_through(self):
        from fsoc_tracker.pipeline.sources import LiveViewportSource
        src = LiveViewportSource(_GradientSource(), 320, 240)
        src.open()
        ts = [src.read().timestamp_s for _ in range(3)]
        assert ts == [0.0, 1.0 / 30.0, 2.0 / 30.0]
        src.release()
