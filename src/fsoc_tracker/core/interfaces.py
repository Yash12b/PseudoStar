"""Abstract interfaces for the FSOC tracking pipeline.

The central abstraction is ``FrameSource``, which produces ``Frame`` objects.
Every downstream component (perception, tracking, control) consumes and
produces typed dataclasses, never raw numpy arrays or bare dicts.

Implementations live in ``fsoc_tracker.io.*``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from fsoc_tracker.core.models import Frame


class FrameSource(ABC):
    """Abstract base class for all frame sources.

    A FrameSource produces a sequence of ``Frame`` objects.  It must never
    assume a fixed frame rate; ``Frame.timestamp_s`` is the authoritative
    time reference.

    Supported future implementations:
        - SyntheticFrameSource (virtual environment renderer)
        - VideoFileFrameSource (MP4, AVI, MKV, MOV via OpenCV/FFmpeg)
        - CameraFrameSource (webcam / live camera)
        - StreamFrameSource (network stream)

    Lifecycle:
        1. ``open()`` - acquire resources
        2. ``read()`` - iterate frames
        3. ``release()`` - free resources

    The source may also be used as a context manager.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the source and prepare for reading.

        Raises:
            FrameSourceError: If the source cannot be opened.
        """

    @abstractmethod
    def read(self) -> Frame | None:
        """Read the next frame from the source.

        Returns:
            The next Frame, or None if the source is exhausted.

        Raises:
            FrameSourceError: If reading fails.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the source is open and ready to provide frames."""

    @abstractmethod
    def release(self) -> None:
        """Release all resources held by the source."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """A unique identifier for this source instance."""

    # --- Optional metadata (subclasses override where known) ---

    @property
    def nominal_fps(self) -> float | None:
        """The FPS the source nominally produces, if known.

        Returns None for variable-FPS sources or when the rate is unknown.
        Subclasses should override when the information is available.
        """
        return None

    @property
    def frame_count(self) -> int | None:
        """Total number of frames in the source, if known.

        Returns None for infinite / live sources.
        """
        return None

    @property
    def duration_s(self) -> float | None:
        """Total duration of the source in seconds, if known.

        Returns None for live / infinite sources.
        """
        return None

    @property
    def width(self) -> int | None:
        """Frame width in pixels, if known before reading."""
        return None

    @property
    def height(self) -> int | None:
        """Frame height in pixels, if known before reading."""
        return None

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame index.

        Not all sources support seeking.  The default implementation
        returns False (seek not supported).

        Args:
            frame_index: The 0-based frame index to seek to.

        Returns:
            True if the seek succeeded, False otherwise.
        """
        return False

    # --- Context manager ---

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        self.release()

    # --- Iterator protocol ---

    def __iter__(self) -> Iterator[Frame]:
        """Allow ``for frame in source:`` iteration."""
        while self.is_open():
            frame = self.read()
            if frame is None:
                break
            yield frame
