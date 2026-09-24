"""Track state machine.

Implements explicit finite state machine transitions for a single track.
Every transition is documented and logged.

State machine:
    NO_TRACK  -> SEARCHING     (tracker initialized)
    SEARCHING -> ACQUIRING     (candidate detection appears)
    ACQUIRING -> TRACKING      (sufficient consecutive hits)
    ACQUIRING -> SEARCHING     (acquisition timeout)
    TRACKING  -> LOST          (detection missing, within prediction window)
    TRACKING  -> SEARCHING     (prediction window exceeded)
    LOST      -> REACQUIRING   (valid detection returns)
    LOST      -> SEARCHING     (reacquisition timeout)
    REACQUIRING -> TRACKING    (stable valid detection)
    REACQUIRING -> SEARCHING   (reacquisition timeout)
"""

from __future__ import annotations

from fsoc_tracker.tracking.state import TrackEvent, TrackingEvent, TrackState

# Valid transitions: (from_state, to_state)
_VALID_TRANSITIONS: set[tuple[TrackState, TrackState]] = {
    (TrackState.NO_TRACK, TrackState.SEARCHING),
    (TrackState.NO_TRACK, TrackState.ACQUIRING),
    (TrackState.NO_TRACK, TrackState.TRACKING),
    (TrackState.SEARCHING, TrackState.ACQUIRING),
    (TrackState.ACQUIRING, TrackState.TRACKING),
    (TrackState.ACQUIRING, TrackState.SEARCHING),
    (TrackState.TRACKING, TrackState.LOST),
    (TrackState.TRACKING, TrackState.SEARCHING),
    (TrackState.LOST, TrackState.REACQUIRING),
    (TrackState.LOST, TrackState.SEARCHING),
    (TrackState.REACQUIRING, TrackState.TRACKING),
    (TrackState.REACQUIRING, TrackState.SEARCHING),
}

# Events emitted on each transition
_TRANSITION_EVENTS: dict[tuple[TrackState, TrackState], TrackEvent] = {
    (TrackState.NO_TRACK, TrackState.SEARCHING): TrackEvent.TRACK_INITIALIZED,
    (TrackState.NO_TRACK, TrackState.ACQUIRING): TrackEvent.TRACK_INITIALIZED,
    (TrackState.NO_TRACK, TrackState.TRACKING): TrackEvent.TRACK_ACQUIRED,
    (TrackState.SEARCHING, TrackState.ACQUIRING): TrackEvent.TRACK_INITIALIZED,
    (TrackState.ACQUIRING, TrackState.TRACKING): TrackEvent.TRACK_ACQUIRED,
    (TrackState.ACQUIRING, TrackState.SEARCHING): TrackEvent.TRACK_LOST,
    (TrackState.TRACKING, TrackState.LOST): TrackEvent.TRACK_LOST,
    (TrackState.TRACKING, TrackState.SEARCHING): TrackEvent.TRACK_LOST,
    (TrackState.LOST, TrackState.REACQUIRING): TrackEvent.TRACK_REACQUIRED,
    (TrackState.LOST, TrackState.SEARCHING): TrackEvent.TRACK_LOST,
    (TrackState.REACQUIRING, TrackState.TRACKING): TrackEvent.TRACK_ACQUIRED,
    (TrackState.REACQUIRING, TrackState.SEARCHING): TrackEvent.TRACK_LOST,
}


class TrackStateMachine:
    """Manages state transitions for a single track.

    Emits TrackingEvent on every valid transition.  Invalid transitions
    raise ValueError.
    """

    def __init__(self, track_id: int = 0) -> None:
        self._state = TrackState.NO_TRACK
        self._track_id = track_id
        self._history: list[TrackingEvent] = []

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def history(self) -> list[TrackingEvent]:
        return list(self._history)

    def transition(self, new_state: TrackState, timestamp_s: float,
                   metadata: dict | None = None) -> TrackingEvent:
        """Attempt a state transition.

        Args:
            new_state: Desired target state.
            timestamp_s: Current timestamp.
            metadata: Optional event metadata.

        Returns:
            The TrackingEvent emitted on successful transition.

        Raises:
            ValueError: If the transition is not valid.
        """
        if new_state == self._state:
            # No-op: already in target state
            return TrackingEvent(
                event=TrackEvent.TRACK_UPDATED,
                timestamp_s=timestamp_s,
                track_id=self._track_id,
            )

        pair = (self._state, new_state)
        if pair not in _VALID_TRANSITIONS:
            raise ValueError(
                f"Invalid transition: {self._state.name} -> {new_state.name}"
            )

        event_type = _TRANSITION_EVENTS[pair]
        event = TrackingEvent(
            event=event_type,
            timestamp_s=timestamp_s,
            track_id=self._track_id,
            metadata=metadata or {},
        )

        self._state = new_state
        self._history.append(event)
        return event

    def force_state(self, state: TrackState, timestamp_s: float) -> None:
        """Force a state without validation (for reset/initialization)."""
        self._state = state

    def reset(self, timestamp_s: float = 0.0) -> None:
        """Reset state machine to NO_TRACK."""
        self._state = TrackState.NO_TRACK
        self._history.clear()
