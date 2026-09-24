"""Tests for the FrameSource interface and SyntheticFrameSource."""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.core.exceptions import FrameSourceError
from fsoc_tracker.core.models import Frame
from fsoc_tracker.io.synthetic import SyntheticFrameSource


class TestSyntheticFrameSource:
    """Tests for the SyntheticFrameSource implementation."""

    def test_open_and_read(self) -> None:
        source = SyntheticFrameSource(total_frames=5)
        source.open()
        assert source.is_open()
        frame = source.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        source.release()
        assert not source.is_open()

    def test_read_returns_none_at_end(self) -> None:
        source = SyntheticFrameSource(total_frames=3)
        source.open()
        for _ in range(3):
            f = source.read()
            assert f is not None
        assert source.read() is None
        source.release()

    def test_read_before_open_raises(self) -> None:
        source = SyntheticFrameSource(total_frames=1)
        with pytest.raises(FrameSourceError, match="not open"):
            source.read()

    def test_frame_dimensions_match(self) -> None:
        source = SyntheticFrameSource(
            camera_width=320, camera_height=240, total_frames=2
        )
        source.open()
        frame = source.read()
        assert frame is not None
        assert frame.width == 320
        assert frame.height == 240
        assert frame.image.shape == (240, 320, 3)
        source.release()

    def test_frame_index_increments(self) -> None:
        source = SyntheticFrameSource(total_frames=5)
        source.open()
        for i in range(5):
            frame = source.read()
            assert frame is not None
            assert frame.frame_index == i
        source.release()

    def test_timestamps_increase(self) -> None:
        source = SyntheticFrameSource(fps=30.0, total_frames=10)
        source.open()
        prev_ts = -1.0
        for _ in range(10):
            frame = source.read()
            assert frame is not None
            assert frame.timestamp_s > prev_ts
            prev_ts = frame.timestamp_s
        source.release()

    def test_nominal_fps(self) -> None:
        source = SyntheticFrameSource(fps=60.0, total_frames=1)
        assert source.nominal_fps == 60.0

    def test_frame_count(self) -> None:
        source = SyntheticFrameSource(total_frames=42)
        assert source.frame_count == 42

    def test_source_id(self) -> None:
        source = SyntheticFrameSource(motion_type="circular", seed=123)
        assert "circular" in source.source_id
        assert "123" in source.source_id

    def test_context_manager(self) -> None:
        with SyntheticFrameSource(total_frames=2) as source:
            assert source.is_open()
            frame = source.read()
            assert frame is not None
        assert not source.is_open()

    def test_iterator_protocol(self) -> None:
        source = SyntheticFrameSource(total_frames=5)
        source.open()
        frames = list(source)
        assert len(frames) == 5
        source.release()

    def test_all_motion_types(self) -> None:
        for motion in ["straight_line", "circular", "figure_8", "random"]:
            source = SyntheticFrameSource(
                motion_type=motion, total_frames=10, seed=42
            )
            source.open()
            frames = list(source)
            assert len(frames) == 10
            # All frames should have valid images
            for f in frames:
                assert f.image.shape == (480, 640, 3)
                assert f.image.dtype == np.uint8
            source.release()

    def test_metadata_present(self) -> None:
        source = SyntheticFrameSource(total_frames=1)
        source.open()
        frame = source.read()
        assert frame is not None
        assert "beacon_x_canvas" in frame.metadata
        assert "beacon_y_canvas" in frame.metadata
        assert "cam_x_canvas" in frame.metadata
        source.release()


class TestSourceIsolation:
    """Video/live sources must not expose a simulation engine.

    A monocular video or webcam frame has no world behind it; the
    source objects must not carry one either.
    """

    def test_video_source_has_no_engine(self):
        import tempfile

        import cv2
        import numpy as np

        from fsoc_tracker.pipeline.sources import VideoSource
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (64, 48))
            out.write(np.zeros((48, 64, 3), dtype=np.uint8))
            out.release()
            src = VideoSource(tmp.name)
            src.open()
            assert not hasattr(src, "engine")
            assert not hasattr(src, "_sim_engine")
            assert not hasattr(src, "_engine")
            frame = src.read()
            assert frame is not None
            assert "ground_truth" not in frame.metadata
            src.release()
        finally:
            import os
            os.unlink(tmp.name)

    def test_live_source_has_no_engine(self):
        from fsoc_tracker.pipeline.sources import LiveSource
        src = LiveSource(camera_id=0)
        assert not hasattr(src, "engine")
        assert not hasattr(src, "_sim_engine")
        assert not hasattr(src, "_engine")
