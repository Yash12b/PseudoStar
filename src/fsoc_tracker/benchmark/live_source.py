"""Live camera benchmark source.

Provides webcam/camera input for real-time performance testing.
"""

from __future__ import annotations

import time
from typing import Any

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType


class LiveCameraSource(FrameSource):
    """Live camera source using OpenCV."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        requested_fps: float = 30.0,
    ) -> None:
        self._device_index = device_index
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = requested_fps
        self._cap: Any = None
        self._frame_index = 0
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._base_time: float | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV is required for live camera source") from None

        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self._device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._requested_width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._requested_height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self._requested_fps))

        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._base_time = time.perf_counter()

    def read(self) -> Frame | None:
        import cv2

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        if bgr.ndim == 3 and bgr.shape[2] == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr if bgr.ndim == 2 else bgr[:, :, 0]

        elapsed = time.perf_counter() - (self._base_time or time.perf_counter())
        fi = self._frame_index
        self._frame_index += 1

        return Frame(
            image=gray,
            width=gray.shape[1],
            height=gray.shape[0],
            channels=1,
            color_model=ColorModel.GRAY,
            source_id=f"camera:{self._device_index}",
            source_type=SourceType.LIVE,
            frame_index=fi,
            timestamp_s=elapsed,
            nominal_fps=self._requested_fps,
            metadata={"timestamp_source": "wall_clock"},
        )

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def source_id(self) -> str:
        return f"camera:{self._device_index}"

    @property
    def nominal_fps(self) -> float:
        return self._requested_fps

    @property
    def width(self) -> int:
        return self._actual_width

    @property
    def height(self) -> int:
        return self._actual_height

    def __enter__(self) -> LiveCameraSource:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()
