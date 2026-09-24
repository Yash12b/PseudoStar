"""Floating panel widget for the immersive explorer.

Provides collapsible, movable panels that overlay the 3D world view.
Panels can be opened, closed, collapsed, moved, and resized.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PanelTitleBar(QWidget):
    """Draggable title bar for floating panels."""

    drag_started = Signal(QPoint)
    close_clicked = Signal()
    collapse_clicked = Signal()

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(4)

        self._title = QLabel(title)
        self._title.setStyleSheet("color: #E0E0E0; font-size: 11px; font-weight: bold; background: transparent;")
        layout.addWidget(self._title, 1)

        self._collapse_btn = QPushButton("\u2013")
        self._collapse_btn.setFixedSize(20, 20)
        self._collapse_btn.setToolTip("Collapse / Expand")
        self._collapse_btn.setStyleSheet(
            "QPushButton { background: #3A3A3A; color: #AAA; border: none; border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover { background: #505050; color: #FFF; }"
        )
        self._collapse_btn.clicked.connect(self.collapse_clicked)
        layout.addWidget(self._collapse_btn)

        self._close_btn = QPushButton("\u00D7")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setToolTip("Close panel")
        self._close_btn.setStyleSheet(
            "QPushButton { background: #3A3A3A; color: #AAA; border: none; border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover { background: #8B0000; color: #FFF; }"
        )
        self._close_btn.clicked.connect(self.close_clicked)
        layout.addWidget(self._close_btn)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapse_btn.setText("+" if collapsed else "\u2013")


class FloatingPanel(QFrame):
    """A floating, collapsible, movable panel."""

    panel_closed = Signal(str)
    panel_moved = Signal(str, QPoint)

    def __init__(
        self,
        panel_id: str,
        title: str,
        content: QWidget,
        parent=None,
        initial_pos: QPoint | None = None,
        initial_size: tuple[int, int] = (220, 200),
    ) -> None:
        super().__init__(parent)
        self._panel_id = panel_id
        self._collapsed = False
        self._drag_pos: QPoint | None = None
        self._content = content

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "FloatingPanel { background: #1A1A1A; border: 1px solid #333; border-radius: 6px; }"
        )
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(2, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._title_bar = PanelTitleBar(title)
        self._title_bar.drag_started.connect(self._on_drag_start)
        self._title_bar.close_clicked.connect(self._on_close)
        self._title_bar.collapse_clicked.connect(self._on_collapse)
        layout.addWidget(self._title_bar)

        self._content_frame = QFrame()
        self._content_frame.setStyleSheet("background: transparent; border: none;")
        content_layout = QVBoxLayout(self._content_frame)
        content_layout.setContentsMargins(6, 4, 6, 6)
        content_layout.addWidget(self._content)
        layout.addWidget(self._content_frame)

        if initial_pos:
            self.move(initial_pos)
        self.resize(*initial_size)

    @property
    def panel_id(self) -> str:
        return self._panel_id

    def _on_drag_start(self, global_pos: QPoint) -> None:
        self._drag_pos = global_pos - self.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(new_pos)
            self.panel_moved.emit(self._panel_id, new_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def _on_close(self) -> None:
        self.hide()
        self.panel_closed.emit(self._panel_id)

    def _on_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self._content_frame.setVisible(not self._collapsed)
        self._title_bar.set_collapsed(self._collapsed)
        if self._collapsed:
            self.setFixedHeight(28)
        else:
            self.setMinimumHeight(60)
            self.setMaximumHeight(16777215)

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
