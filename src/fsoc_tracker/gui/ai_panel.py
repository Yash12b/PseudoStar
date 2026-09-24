"""AI perception panel — model status and controls."""

from __future__ import annotations

from fsoc_tracker.gui.state import ApplicationViewState
from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QGroupBox,
        QLabel,
        QVBoxLayout,
    )
except ImportError:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (
        QFrame,  # type: ignore
        QGridLayout,
        QGroupBox,
        QLabel,
        QVBoxLayout,
    )


class AIPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("AI PERCEPTION")
        title.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(title)

        grp = QGroupBox("Model Status")
        grp.setStyleSheet(f"QGroupBox {{ color: {Colors.ACCENT}; font-size: 9px; font-weight: bold; border: 1px solid {Colors.PANEL_BORDER}; border-radius: 3px; margin-top: 6px; padding-top: 8px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 3px; }}")
        grid = QGridLayout(grp)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        fields = [
            ("Backend", "backend"),
            ("Model", "model"),
            ("Parameters", "params"),
            ("Inference MS", "inference_ms"),
            ("Confidence", "confidence"),
            ("Status", "status"),
            ("Mission Model", "mission_model"),
        ]
        for i, (label, key) in enumerate(fields):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val = QLabel("--")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._labels[key] = val
            grid.addWidget(lbl, i, 0)
            grid.addWidget(val, i, 1)

        layout.addWidget(grp)
        layout.addStretch()

    def update_state(self, state: ApplicationViewState) -> None:
        def set_val(key: str, text: str, color: str | None = None) -> None:
            if key in self._labels:
                self._labels[key].setText(text)
                if color:
                    self._labels[key].setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

        backend = state.perception.backend
        set_val("backend", backend)

        if backend == "ai":
            set_val("model", "BeaconCNN")
            set_val("params", "~15K")
            set_val("inference_ms", f"{state.perception.processing_ms:.1f}")
            set_val("confidence", f"{state.perception.confidence:.3f}")
            status_color = Colors.SUCCESS if state.perception.detected else Colors.MUTED
            set_val("status", "ACTIVE", status_color)
        elif backend == "hybrid":
            set_val("model", "Classical+AI")
            set_val("params", "Hybrid")
            set_val("inference_ms", f"{state.perception.processing_ms:.1f}")
            set_val("confidence", f"{state.perception.confidence:.3f}")
            status_color = Colors.WARNING if state.perception.detected else Colors.MUTED
            set_val("status", "HYBRID", status_color)
        else:
            set_val("model", "Classical")
            set_val("params", "Bright spot")
            set_val("inference_ms", f"{state.perception.processing_ms:.1f}")
            set_val("confidence", f"{state.perception.confidence:.3f}")
            status_color = Colors.SUCCESS if state.perception.detected else Colors.MUTED
            set_val("status", "ACTIVE", status_color)

        # Mission-brain provenance comes straight from runtime state
        # (e.g. "expert_only"): never a hard-coded model claim.
        mission_model = state.ai_state.model_status or "--"
        mission_color = (Colors.WARNING if state.ai_state.fallback_active
                         else Colors.TEXT)
        if state.ai_state.fallback_active:
            mission_model += " (FALLBACK)"
        set_val("mission_model", mission_model, mission_color)
