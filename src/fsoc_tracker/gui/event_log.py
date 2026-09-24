"""Event log panel — scrollable text log."""

from __future__ import annotations

from fsoc_tracker.gui.state import ApplicationViewState
from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout
except ImportError:
    from PyQt5.QtGui import QTextCursor  # type: ignore
    from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout  # type: ignore


class EventLogPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._last_count = 0
        self._last_key = None
        self._last_events: list = []

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("EVENT LOG")
        header.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(header)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(120)
        # Bound the document itself: over hours of operation the widget
        # must not grow without limit even if event flow is heavy.
        self._text.document().setMaximumBlockCount(500)
        self._text.setStyleSheet(f"QTextEdit {{ background-color: {Colors.BACKGROUND}; border: 1px solid {Colors.PANEL_BORDER}; color: {Colors.TEXT}; font-size: 9px; font-family: monospace; padding: 2px; }}")
        layout.addWidget(self._text)

    def update_state(self, state: ApplicationViewState) -> None:
        events = state.events
        self._last_events = list(events)
        # Render from the last-seen entry onward (matched by identity),
        # not merely by count: the state buffer rolls over at its cap,
        # so length alone would freeze the display on long runs.
        start = 0
        if self._last_key is not None:
            # Match from the end: repeated identical messages must not
            # cause re-rendering from their first occurrence.
            for i in range(len(events) - 1, -1, -1):
                entry = events[i]
                if (entry.timestamp_s, entry.level, entry.message) == self._last_key:
                    start = i + 1
                    break
        for entry in events[start:]:
            color = {
                "INFO": Colors.TEXT, "WARNING": Colors.WARNING,
                "ERROR": Colors.ERROR, "SUCCESS": Colors.SUCCESS,
            }.get(entry.level, Colors.TEXT)
            ts = f"{entry.timestamp_s:7.2f}s"
            self._text.append(f"<span style='color:{Colors.MUTED}'>{ts}</span> <span style='color:{color}'>[{entry.level}]</span> {entry.message}")
            self._last_key = (entry.timestamp_s, entry.level, entry.message)
        self._last_count = len(events)
        self._text.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        self._text.clear()
        self._last_count = 0
        self._last_key = None

    def export_jsonl(self, path: str) -> str:
        """Write buffered events as JSON Lines. Returns the path."""
        import json
        with open(path, "w") as f:
            for e in self._last_events:
                f.write(json.dumps({
                    "timestamp_s": float(e.timestamp_s),
                    "level": str(e.level),
                    "message": str(e.message),
                }) + "\n")
        return path

    def export_csv(self, path: str) -> str:
        """Write buffered events as CSV. Returns the path."""
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "level", "message"])
            for e in self._last_events:
                writer.writerow([float(e.timestamp_s), str(e.level),
                                 str(e.message)])
        return path
