"""External video benchmark source.

Supports MP4, AVI, MKV, MOV and other formats via OpenCV backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType


class VideoBenchmarkSource(FrameSource):
    """Video file source using OpenCV."""

    def __init__(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int | None = None,
        start_time_s: float | None = None,
        duration_s: float | None = None,
    ) -> None:
        self._path = video_path
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._start_time_s = start_time_s
        self._duration_limit = duration_s
        self._cap: Any = None
        self._frame_index = 0
        self._total_read = 0
        self._nominal_fps: float | None = None
        self._frame_count: int | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._duration: float | None = None
        self._timestamp_source = "derived_from_nominal_fps"
        self._start_offset_s: float = 0.0
        self._last_timestamp_s: float = -1.0
        self._base_time: float | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV is required for video benchmark source") from None

        p = Path(self._path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")

        self._nominal_fps = self._cap.get(cv2.CAP_PROP_FPS) or None
        fc = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._frame_count = int(fc) if fc > 0 else None
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._nominal_fps and self._nominal_fps > 0:
            self._duration = (self._frame_count / self._nominal_fps) if self._frame_count else None

        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._start_frame))
            self._frame_index = self._start_frame

        if self._start_time_s is not None:
            self._start_offset_s = self._start_time_s
            if self._nominal_fps and self._nominal_fps > 0:
                target_frame = int(self._start_time_s * self._nominal_fps)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
                self._frame_index = target_frame

        self._base_time = time.perf_counter()

    def read(self) -> Frame | None:
        import cv2

        if self._cap is None or not self._cap.isOpened():
            return None

        if self._end_frame is not None and self._frame_index >= self._end_frame:
            return None

        if self._duration_limit is not None and self._base_time is not None:
            elapsed = time.perf_counter() - self._base_time
            if elapsed >= self._duration_limit:
                return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            color_model = ColorModel.GRAY
        elif bgr.ndim == 2:
            gray = bgr
            color_model = ColorModel.GRAY
        else:
            gray = bgr[:, :, 0] if bgr.ndim == 3 else bgr
            color_model = ColorModel.GRAY

        ts = self._compute_timestamp()
        fi = self._frame_index
        self._frame_index += 1
        self._total_read += 1
        self._last_timestamp_s = ts

        return Frame(
            image=gray,
            width=gray.shape[1],
            height=gray.shape[0],
            channels=1,
            color_model=color_model,
            source_id=f"video:{self._path}",
            source_type=SourceType.VIDEO,
            frame_index=fi,
            timestamp_s=ts,
            nominal_fps=self._nominal_fps,
            metadata={"timestamp_source": self._timestamp_source},
        )

    def _compute_timestamp(self) -> float:
        # Prefer actual decoder timestamps (same policy as VideoSource);
        # fall back to nominal FPS, then wall clock. Never hard-code a rate.
        import cv2

        if self._cap is not None:
            position_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
            decoder_ts = position_ms / 1000.0
            if decoder_ts >= 0 and decoder_ts > self._last_timestamp_s:
                self._timestamp_source = "decoder_position"
                return decoder_ts
        if self._nominal_fps and self._nominal_fps > 0:
            self._timestamp_source = "derived_from_nominal_fps"
            return self._start_offset_s + (self._frame_index / self._nominal_fps)
        elapsed = time.perf_counter() - (self._base_time or time.perf_counter())
        self._timestamp_source = "wall_clock"
        return self._start_offset_s + elapsed

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def source_id(self) -> str:
        return f"video:{self._path}"

    @property
    def nominal_fps(self) -> float | None:
        return self._nominal_fps

    @property
    def frame_count(self) -> int | None:
        remaining = None
        if self._frame_count is not None:
            remaining = self._frame_count - self._start_frame
            if self._end_frame is not None:
                remaining = min(remaining, self._end_frame - self._start_frame)
        return remaining

    @property
    def duration_s(self) -> float | None:
        return self._duration

    @property
    def width(self) -> int | None:
        return self._width

    @property
    def height(self) -> int | None:
        return self._height

    @property
    def total_frames_read(self) -> int:
        return self._total_read

    @property
    def timestamp_source(self) -> str:
        return self._timestamp_source

    def seek(self, frame_index: int) -> bool:
        import cv2
        if self._cap is None:
            return False
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        actual = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual == int(frame_index):
            self._frame_index = int(frame_index)
            self._last_timestamp_s = -1.0
            return True
        return False

    @property
    def position_frames(self) -> int | None:
        """Decoder's current frame position, or None when unavailable."""
        if self._cap is None:
            return None
        try:
            import cv2
            return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception:
            return None

    def reopen_and_seek(self, frame_index: int) -> bool:
        """Recover a stalled decoder: re-open the file and seek.

        Same sticky-EOF recovery contract as pipeline VideoSource
        (OpenCV's ffmpeg backend returns None forever after a damaged
        packet even though later frames decode fine).
        """
        try:
            self.release()
            self.open()
        except Exception:
            return False
        return self.seek(int(frame_index))

    def __enter__(self) -> VideoBenchmarkSource:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()
