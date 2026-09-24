"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from fsoc_tracker.core.exceptions import (
    BenchmarkError,
    ConfigurationError,
    ControlError,
    FrameSourceError,
    FSOCTrackerError,
    PerceptionError,
    SimulationError,
    TrackingError,
)


class TestExceptionHierarchy:
    """All custom exceptions must inherit from FSOCTrackerError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            FrameSourceError,
            PerceptionError,
            TrackingError,
            ControlError,
            SimulationError,
            BenchmarkError,
        ],
    )
    def test_inherits_from_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, FSOCTrackerError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            FrameSourceError,
            PerceptionError,
            TrackingError,
            ControlError,
            SimulationError,
            BenchmarkError,
        ],
    )
    def test_catchable_as_base(self, exc_class: type) -> None:
        with pytest.raises(FSOCTrackerError):
            raise exc_class("test error")

    def test_base_is_exception(self) -> None:
        assert issubclass(FSOCTrackerError, Exception)
