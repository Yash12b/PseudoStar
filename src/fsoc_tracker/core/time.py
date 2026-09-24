"""Time and dt abstraction for the FSOC tracking pipeline.

Critical design principle:
    dt MUST be derived from frame timestamps, never assumed to be 1/30.

This module provides:
    - ``compute_dt``: derive inter-frame dt from timestamps with validation.
    - ``FrameTimestamp``: lightweight container for frame timing data.
    - ``SimulationClock``: manages simulation time, supporting real-time,
      accelerated, paused, and stepped modes.
    - ``ProcessingClock``: measures wall-clock processing latency.

Future simulator and controller will use ``SimulationClock`` to allow:
    - real-time playback
    - accelerated simulation
    - paused state
    - frame-by-frame stepping
    - deterministic test runs
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Frame timestamp container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameTimestamp:
    """Immutable timing data attached to a frame.

    All times are in seconds.
    """

    timestamp_s: float
    """Source-provided timestamp (wall-clock or source-relative)."""

    monotonic_s: float | None = None
    """``time.monotonic()`` reading at capture, if available."""

    nominal_fps: float | None = None
    """FPS the source claims, if known."""

    @property
    def nominal_dt(self) -> float | None:
        """Inferred dt from nominal_fps, or None."""
        if self.nominal_fps is not None and self.nominal_fps > 0:
            return 1.0 / self.nominal_fps
        return None


# ---------------------------------------------------------------------------
# dt computation
# ---------------------------------------------------------------------------

_MIN_DT = 1e-9
_MAX_DT = 10.0  # anything above 10 s between frames is suspicious


def compute_dt(
    current_ts: float,
    previous_ts: float,
    nominal_fps: float | None = None,
    fallback_dt: float = 1.0 / 30.0,
) -> float:
    """Compute inter-frame dt from timestamps with validation.

    Args:
        current_ts: Timestamp of the current frame (seconds).
        previous_ts: Timestamp of the previous frame (seconds).
        nominal_fps: Nominal FPS from the source, used as a sanity check.
        fallback_dt: Value returned when timestamps are invalid or equal.

    Returns:
        The dt in seconds, clamped to [_MIN_DT, _MAX_DT].

    Notes:
        If timestamps are identical or produce an out-of-range dt, the
        nominal_fps-derived dt or fallback_dt is used.  This prevents
        division-by-zero and runaway dt values in downstream controllers.
    """
    dt = current_ts - previous_ts

    if dt < _MIN_DT:
        # Timestamps identical or went backwards
        if nominal_fps is not None and nominal_fps > 0:
            return 1.0 / nominal_fps
        return fallback_dt

    if dt > _MAX_DT:
        # Suspiciously large gap; use fallback
        if nominal_fps is not None and nominal_fps > 0:
            return 1.0 / nominal_fps
        return fallback_dt

    return dt


# ---------------------------------------------------------------------------
# Simulation clock
# ---------------------------------------------------------------------------

class SimulationClock:
    """Manages simulation time independently of wall-clock time.

    Supports:
        - real-time mode (wall clock)
        - accelerated mode (configurable time scale)
        - paused mode
        - frame stepping

    The simulation clock is the single source of truth for time
    inside the simulation loop.  Controllers and disturbance models
    should read from this clock, not from ``time.time()``.
    """

    def __init__(self, time_scale: float = 1.0) -> None:
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self._time_scale: float = time_scale
        self._sim_time_s: float = 0.0
        self._wall_start_s: float = time.monotonic()
        self._paused: bool = False
        self._pause_offset_s: float = 0.0

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        if value <= 0:
            raise ValueError("time_scale must be positive")
        self._time_scale = value

    @property
    def sim_time_s(self) -> float:
        """Current simulation time in seconds."""
        if self._paused:
            return self._sim_time_s
        elapsed = time.monotonic() - self._wall_start_s - self._pause_offset_s
        return self._sim_time_s + elapsed * self._time_scale

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Pause the simulation clock."""
        if not self._paused:
            self._sim_time_s = self.sim_time_s
            self._paused = True

    def resume(self) -> None:
        """Resume the simulation clock."""
        if self._paused:
            self._wall_start_s = time.monotonic()
            self._pause_offset_s = 0.0
            self._paused = False

    def step(self, dt_s: float) -> float:
        """Advance the clock by a fixed dt (for frame-stepping mode).

        Returns:
            The new simulation time after stepping.
        """
        self._sim_time_s += dt_s
        return self._sim_time_s

    def reset(self) -> None:
        """Reset the clock to zero."""
        self._sim_time_s = 0.0
        self._wall_start_s = time.monotonic()
        self._pause_offset_s = 0.0


# ---------------------------------------------------------------------------
# Processing clock
# ---------------------------------------------------------------------------

class ProcessingClock:
    """Measures wall-clock processing time for benchmarking.

    Used to measure per-frame processing latency and overall throughput.
    """

    def __init__(self) -> None:
        self._frame_times: list[float] = []
        self._last_start: float = 0.0

    def start(self) -> None:
        """Mark the start of a processing step."""
        self._last_start = time.perf_counter()

    def stop(self) -> float:
        """Mark the end of a processing step and return elapsed seconds.

        Returns:
            Elapsed time in seconds since the last ``start()`` call.
        """
        elapsed = time.perf_counter() - self._last_start
        self._frame_times.append(elapsed)
        return elapsed

    @property
    def last_dt(self) -> float:
        """Most recently measured processing time in seconds."""
        if not self._frame_times:
            return 0.0
        return self._frame_times[-1]

    @property
    def average_dt(self) -> float:
        """Average processing time over all measured frames."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)

    @property
    def current_fps(self) -> float:
        """Instantaneous FPS from the last measured frame."""
        dt = self.last_dt
        if dt <= 0:
            return 0.0
        return 1.0 / dt

    @property
    def average_fps(self) -> float:
        """Average FPS over all measured frames."""
        dt = self.average_dt
        if dt <= 0:
            return 0.0
        return 1.0 / dt

    def reset(self) -> None:
        """Clear all recorded frame times."""
        self._frame_times.clear()
        self._last_start = 0.0
