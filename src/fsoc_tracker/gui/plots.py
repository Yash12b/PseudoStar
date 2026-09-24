"""Live error plots — using QPainter for lightweight rendering."""

from __future__ import annotations

from fsoc_tracker.gui.state import ApplicationViewState
from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import QFrame, QVBoxLayout
except ImportError:
    from PyQt5.QtCore import QPointF, QRectF, Qt  # type: ignore
    from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen  # type: ignore
    from PyQt5.QtWidgets import QFrame, QVBoxLayout  # type: ignore


class MiniPlot(QFrame):
    def __init__(self, title: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._color = color
        self._data: list[float] = []
        self._timestamps: list[float] = []
        self._y_min = 0.0
        self._y_max = 10.0
        self._auto_scale = True
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

    def set_data(self, timestamps: list[float], values: list[float]) -> None:
        self._timestamps = timestamps
        self._data = values
        if self._auto_scale and values:
            self._y_min = min(0, min(values))
            self._y_max = max(max(values) * 1.2, self._y_max * 0.8 + max(values) * 0.2)
            if self._y_max - self._y_min < 1e-6:
                self._y_max = self._y_min + 1.0
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            w = self.width()
            h = self.height()
            margin_l, margin_r, margin_t, margin_b = 40, 8, 16, 14
            plot_x = margin_l
            plot_y = margin_t
            plot_w = w - margin_l - margin_r
            plot_h = h - margin_t - margin_b

            painter.setPen(QPen(QColor(Colors.PANEL_BORDER), 1))
            painter.drawRect(plot_x, plot_y, plot_w, plot_h)

            painter.setPen(QPen(QColor(Colors.MUTED), 1))
            painter.setFont(QFont("Menlo", 7))
            painter.drawText(QRectF(0, 0, margin_l, h), Qt.AlignRight | Qt.AlignTop, f"{self._y_max:.1f}")
            painter.drawText(QRectF(0, h - margin_b, margin_l, margin_b), Qt.AlignRight | Qt.AlignBottom, f"{self._y_min:.1f}")
            painter.drawText(QRectF(plot_x, 0, plot_w, margin_t), Qt.AlignCenter | Qt.AlignTop, self._title)

            if len(self._data) < 2:
                return

            n = len(self._data)
            y_range = self._y_max - self._y_min
            painter.setPen(QPen(QColor(self._color), 1.5))

            path = QPainterPath()
            for i in range(n):
                x = plot_x + (i / max(n - 1, 1)) * plot_w
                y_norm = (self._data[i] - self._y_min) / y_range if y_range > 0 else 0.5
                y = plot_y + plot_h - y_norm * plot_h
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)

            last_x = plot_x + plot_w
            last_y_norm = (self._data[-1] - self._y_min) / y_range if y_range > 0 else 0.5
            last_y = plot_y + plot_h - last_y_norm * plot_h
            painter.setBrush(QColor(self._color))
            painter.drawEllipse(QPointF(last_x, last_y), 3, 3)
            painter.drawText(
                QRectF(plot_x + plot_w + 2, last_y - 6, margin_r, 12),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{self._data[-1]:.1f}",
            )
        finally:
            painter.end()


class PlotsWidget(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._err_x = MiniPlot("X Error (px)", Colors.ACCENT)
        self._err_y = MiniPlot("Y Error (px)", Colors.WARNING)
        self._err_eucl = MiniPlot("Euclidean (px)", Colors.TARGET)
        self._fps = MiniPlot("FPS", Colors.SUCCESS)

        for plot in [self._err_x, self._err_y, self._err_eucl, self._fps]:
            layout.addWidget(plot)
        layout.addStretch()

    def update_state(self, state: ApplicationViewState) -> None:
        e = state.errors
        if e.timestamps:
            self._err_x.set_data(e.timestamps, e.errors_x)
            self._err_y.set_data(e.timestamps, e.errors_y)
            self._err_eucl.set_data(e.timestamps, e.errors_euclidean)
            self._fps.set_data(e.timestamps, e.fps_history)
