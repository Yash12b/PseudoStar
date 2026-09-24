"""About dialog — application info."""

from __future__ import annotations

from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )
except ImportError:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (
        QDialog,  # type: ignore
        QLabel,
        QPushButton,
        QVBoxLayout,
    )


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setFixedSize(360, 280)
        self.setStyleSheet(f"background-color: {Colors.BACKGROUND}; color: {Colors.TEXT};")
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("FSOC Coarse Pointing Tracker")
        title.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 16px; font-weight: bold; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("SIH26169 Virtual Camera Tracking System")
        subtitle.setStyleSheet(f"color: {Colors.MUTED}; font-size: 11px; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        info_lines = [
            "Virtual camera tracking for FSOC coarse alignment",
            "with AI-enhanced beacon detection and classical perception.",
            "",
            "Stages 1-11: Full pipeline + benchmark + GUI",
            "653+ unit tests across 11 stages",
            "",
            "Python 3.14 | NumPy | OpenCV | PySide6",
        ]
        info = QLabel("\n".join(info_lines))
        info.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; background: transparent;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        layout.addStretch()

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        btn.setFixedWidth(100)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
