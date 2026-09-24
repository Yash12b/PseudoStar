"""Camera Tracking Workspace — primary operational view.

Displays the live camera frame with overlays:
  - Target overlay (detection bounding box)
  - Centroid (detection center)
  - Camera center (crosshair)
  - FOV boundary (if applicable)
  - Prediction overlay (Kalman estimate)
  - Tracking state text

Also displays real telemetry values around/below the frame.
"""

from __future__ import annotations

import math

import numpy as np

from fsoc_tracker.gui.state import ApplicationViewState
from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QColor,
        QImage,
        QPainter,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QLabel,
        QSizePolicy,
        QVBoxLayout,
    )
except ImportError:
    from PyQt5.QtCore import QPointF, Qt  # type: ignore
    from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap  # type: ignore
    from PyQt5.QtWidgets import (
        QFrame,
        QGridLayout,
        QLabel,  # type: ignore
        QSizePolicy,  # type: ignore
        QVBoxLayout,
    )


class CameraTrackingWorkspace(QFrame):
    """Primary mission view: live camera frame + overlays + telemetry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CameraWorkspace")
        self.setMinimumSize(400, 300)
        self._state: ApplicationViewState | None = None
        self._current_pixmap: QPixmap | None = None
        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(1, 1)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet("background: #050a0f;")
        self._build_telemetry()

    def _build_telemetry(self) -> None:
        """Build the telemetry labels around the camera frame."""
        self._telem: dict[str, QLabel] = {}
        # Will be populated on first update
        self._telemetry_built = False

    def _ensure_telemetry(self, layout: QVBoxLayout) -> None:
        """Lazily build telemetry grid below the camera frame."""
        if self._telemetry_built:
            return
        self._telemetry_built = True

        telem_frame = QFrame()
        telem_frame.setStyleSheet("QFrame { background: rgba(13,20,28,240); border-top: 1px solid #20303d; }")
        telem_grid = QGridLayout(telem_frame)
        telem_grid.setContentsMargins(8, 4, 8, 4)
        telem_grid.setSpacing(2)

        fields = [
            ("TARGET", "--"), ("STATE", "--"), ("CENTROID", "--"),
            ("X ERROR", "--"), ("Y ERROR", "--"), ("EUCL ERROR", "--"),
            ("CONFIDENCE", "--"), ("VELOCITY", "--"), ("PREDICTION", "--"),
            ("UNCERTAINTY", "--"), ("PAN", "--"), ("TILT", "--"),
            ("FPS", "--"), ("LATENCY", "--"),
        ]
        for i, (name, default) in enumerate(fields):
            row = i // 7
            col = (i % 7) * 2
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            lbl_val = QLabel(default)
            lbl_val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._telem[name] = lbl_val
            telem_grid.addWidget(lbl_name, row, col)
            telem_grid.addWidget(lbl_val, row, col + 1)

        layout.addWidget(telem_frame)

    def update_frame(self, image: np.ndarray | None, state: ApplicationViewState) -> None:
        """Update the camera frame and overlays."""
        self._state = state

        # Lazy build layout
        if not self._telemetry_built:
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            root.addWidget(self._image_label)
            self._ensure_telemetry(root)

        if image is None or image.size == 0:
            return

        # Convert numpy to QPixmap
        if image.ndim == 2:
            h, w = image.shape
            bytes_per_line = w
            fmt = QImage.Format_Grayscale8
            qimg = QImage(image.data, w, h, bytes_per_line, fmt)
        else:
            h, w, ch = image.shape
            if ch == 3:
                rgb = image[:, :, ::-1].copy()
                fmt = QImage.Format_RGB888
            else:
                rgb = image.copy()
                fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, w, h, w * 3, fmt)

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Compose video + overlays onto a single pixmap. Overlays MUST be
        # baked into the displayed pixmap: anything painted in paintEvent
        # on the parent frame is covered by the image QLabel child.
        composed = self._compose_overlays(scaled)
        self._current_pixmap = composed
        self._image_label.setPixmap(composed)
        self._update_telemetry()
        self.update()

    def _compose_overlays(self, scaled: QPixmap) -> QPixmap:
        """Paint the video frame plus all HUD overlays onto one pixmap.

        The scaled frame is letterboxed onto a label-sized canvas; overlay
        coordinates map image pixels through the same scale + offset.
        """
        s = self._state
        lw = max(self._image_label.width(), 1)
        lh = max(self._image_label.height(), 1)
        canvas = QPixmap(lw, lh)
        canvas.fill(QColor("#050a0f"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        img_w = max(s.camera.image_width, 1) if s is not None else scaled.width()
        img_h = max(s.camera.image_height, 1) if s is not None else scaled.height()
        scale = min(lw / img_w, lh / img_h)
        dw, dh = img_w * scale, img_h * scale
        ox, oy = (lw - dw) / 2.0, (lh - dh) / 2.0
        painter.drawPixmap(int(ox), int(oy), scaled)

        def ix(x: float) -> int:
            return int(ox + x * scale)

        def iy(y: float) -> int:
            return int(oy + y * scale)

        if s is not None:
            cx, cy = lw // 2, lh // 2

            # --- Camera center crosshair ---
            pen_ch = QPen(QColor(Colors.CROSSHAIR), 1, Qt.DashLine)
            painter.setPen(pen_ch)
            painter.drawLine(cx - 20, cy, cx + 20, cy)
            painter.drawLine(cx, cy - 20, cx, cy + 20)

            # --- FOV boundary ---
            pen_fov = QPen(QColor(Colors.CENTERLINE), 1, Qt.DotLine)
            painter.setPen(pen_fov)
            painter.drawRect(4, 4, lw - 8, lh - 8)

            # --- Tracking reticle (square brackets + center +) ---
            reticle_active = (s.tracking.locked
                              or s.tracking.state in ("TRACKING", "ACQUIRING")
                              or s.perception.detected)
            if reticle_active:
                if s.tracking.locked or s.tracking.state in ("TRACKING", "ACQUIRING"):
                    rx, ry = ix(s.tracking.estimated_x), iy(s.tracking.estimated_y)
                else:
                    rx, ry = ix(s.perception.detection_x), iy(s.perception.detection_y)

                gate_half = max(int(s.target.size_px * scale * 0.8), 12)

                if s.tracking.locked:
                    reticle_color = QColor(Colors.SUCCESS)
                elif s.tracking.state in ("TRACKING", "ACQUIRING"):
                    reticle_color = QColor(Colors.TARGET)
                elif s.perception.detected:
                    reticle_color = QColor(Colors.WARNING)
                else:
                    reticle_color = QColor(Colors.ERROR)

                pen_ret = QPen(reticle_color, 2)
                painter.setPen(pen_ret)
                painter.setBrush(QColor(0, 0, 0, 0))

                bl = max(gate_half // 3, 6)
                painter.drawLine(rx - gate_half, ry - gate_half,
                                 rx - gate_half + bl, ry - gate_half)
                painter.drawLine(rx - gate_half, ry - gate_half,
                                 rx - gate_half, ry - gate_half + bl)
                painter.drawLine(rx + gate_half, ry - gate_half,
                                 rx + gate_half - bl, ry - gate_half)
                painter.drawLine(rx + gate_half, ry - gate_half,
                                 rx + gate_half, ry - gate_half + bl)
                painter.drawLine(rx - gate_half, ry + gate_half,
                                 rx - gate_half + bl, ry + gate_half)
                painter.drawLine(rx - gate_half, ry + gate_half,
                                 rx - gate_half, ry + gate_half - bl)
                painter.drawLine(rx + gate_half, ry + gate_half,
                                 rx + gate_half - bl, ry + gate_half)
                painter.drawLine(rx + gate_half, ry + gate_half,
                                 rx + gate_half, ry + gate_half - bl)

                ch_len = max(gate_half // 2, 5)
                painter.setPen(QPen(reticle_color, 1))
                painter.drawLine(rx - ch_len, ry, rx + ch_len, ry)
                painter.drawLine(rx, ry - ch_len, rx, ry + ch_len)

                painter.setBrush(reticle_color)
                painter.drawEllipse(QPointF(rx, ry), 2, 2)

                if s.tracking.locked or s.tracking.state in ("TRACKING", "ACQUIRING"):
                    unc_r = max(int(s.tracking.uncertainty_x * scale), 3)
                    painter.setPen(QPen(reticle_color, 1, Qt.DotLine))
                    painter.setBrush(QColor(0, 0, 0, 0))
                    painter.drawEllipse(QPointF(rx, ry), unc_r, unc_r)

            # --- Prediction marker (AI motion forecast) ---
            pred_dx = float(s.ai_state.prediction_dx)
            pred_dy = float(s.ai_state.prediction_dy)
            if s.tracking.state in ("TRACKING", "ACQUIRING") and (pred_dx != 0.0 or pred_dy != 0.0):
                est_x = float(s.tracking.estimated_x)
                est_y = float(s.tracking.estimated_y)
                pred_x, pred_y = ix(est_x + pred_dx), iy(est_y + pred_dy)
                base_x, base_y = ix(est_x), iy(est_y)

                painter.setPen(QPen(QColor(Colors.PREDICTED), 1, Qt.DashLine))
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.drawLine(base_x, base_y, pred_x, pred_y)

                d = 5
                painter.setPen(QPen(QColor(Colors.PREDICTED), 2))
                painter.drawLine(pred_x - d, pred_y, pred_x, pred_y - d)
                painter.drawLine(pred_x, pred_y - d, pred_x + d, pred_y)
                painter.drawLine(pred_x + d, pred_y, pred_x, pred_y + d)
                painter.drawLine(pred_x, pred_y + d, pred_x - d, pred_y)

                pred_unc = math.sqrt(
                    float(s.ai_state.prediction_uncertainty_x) ** 2
                    + float(s.ai_state.prediction_uncertainty_y) ** 2
                )
                if pred_unc > 0.0:
                    painter.setPen(QPen(QColor(Colors.PREDICTED), 1, Qt.DotLine))
                    painter.drawEllipse(
                        QPointF(pred_x, pred_y),
                        max(int(pred_unc * scale), 2), max(int(pred_unc * scale), 2),
                    )

                painter.setPen(QPen(QColor(Colors.PREDICTED), 1))
                tag_font = painter.font()
                tag_font.setPointSize(8)
                tag_font.setBold(False)
                painter.setFont(tag_font)
                painter.drawText(
                    pred_x + 8, pred_y - 6,
                    f"PRED +{float(s.ai_state.prediction_horizon_s):.2f}s",
                )

            # --- Tracking state text overlay ---
            state_text = f"TRK: {s.tracking.state}"
            if s.tracking.locked:
                state_text += " | LOCKED"
            painter.setPen(QPen(QColor(Colors.ACCENT), 1))
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(8, 20, state_text)

            # --- Mode indicator ---
            mode_text = f"SRC: {s.system_mode.value.upper()}"
            painter.setPen(QPen(QColor(Colors.MUTED), 1))
            font.setPointSize(9)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(8, 38, mode_text)

            # --- Distance status (for LIVE mode) ---
            if s.system_mode.value.upper() == "LIVE":
                distance_status = s.live_distance_status if hasattr(s, 'live_distance_status') else "UNAVAILABLE"
                distance_text = f"DIST: {distance_status}"
                painter.setPen(QPen(QColor(Colors.WARNING) if distance_status == "UNAVAILABLE" else Colors.SUCCESS, 1))
                painter.drawText(8, 56, distance_text)

        painter.end()
        return canvas

    def _update_telemetry(self) -> None:
        """Update the telemetry labels with real values."""
        if self._state is None or not self._telem:
            return
        s = self._state

        def set_t(key: str, text: str, color: str | None = None) -> None:
            if key in self._telem:
                self._telem[key].setText(text)
                if color:
                    self._telem[key].setStyleSheet(
                        f"color: {color}; font-size: 10px; font-weight: bold; background: transparent; border: none;"
                    )

        # Target
        if s.perception.detected:
            set_t("TARGET", f"({s.perception.detection_x:.1f}, {s.perception.detection_y:.1f})", Colors.TARGET)
        else:
            set_t("TARGET", "NONE", Colors.MUTED)

        # State
        state_colors = {
            "NO_TRACK": Colors.MUTED, "SEARCHING": Colors.WARNING,
            "ACQUIRING": Colors.WARNING, "TRACKING": Colors.SUCCESS,
            "LOST": Colors.ERROR, "REACQUIRING": Colors.WARNING,
        }
        set_t("STATE", s.tracking.state, state_colors.get(s.tracking.state, Colors.TEXT))

        # Centroid
        if s.tracking.locked or s.tracking.state in ("TRACKING", "ACQUIRING"):
            set_t("CENTROID", f"({s.tracking.estimated_x:.1f}, {s.tracking.estimated_y:.1f})", Colors.SUCCESS)
        else:
            set_t("CENTROID", "--", Colors.MUTED)

        # Errors
        err_x = s.tracking.estimated_x - s.perception.detection_x if s.perception.detected else 0.0
        err_y = s.tracking.estimated_y - s.perception.detection_y if s.perception.detected else 0.0
        err_eucl = math.sqrt(err_x**2 + err_y**2) if s.perception.detected else 0.0
        err_color = Colors.SUCCESS if err_eucl < 10 else Colors.WARNING if err_eucl < 20 else Colors.ERROR
        set_t("X ERROR", f"{err_x:+.1f}px", err_color)
        set_t("Y ERROR", f"{err_y:+.1f}px", err_color)
        set_t("EUCL ERROR", f"{err_eucl:.1f}px", err_color)

        # Confidence
        conf_color = Colors.SUCCESS if s.perception.confidence > 0.7 else Colors.WARNING if s.perception.confidence > 0.3 else Colors.ERROR
        set_t("CONFIDENCE", f"{s.perception.confidence:.2f}", conf_color)

        # Velocity
        if s.tracking.locked:
            vel = math.sqrt(s.tracking.velocity_x**2 + s.tracking.velocity_y**2)
            set_t("VELOCITY", f"{vel:.1f}px/s", Colors.TEXT)
        else:
            set_t("VELOCITY", "--", Colors.MUTED)

        # Prediction
        if s.tracking.prediction_only:
            set_t("PREDICTION", "ACTIVE", Colors.WARNING)
        else:
            set_t("PREDICTION", "NONE", Colors.MUTED)

        # Uncertainty
        unc = math.sqrt(s.tracking.uncertainty_x**2 + s.tracking.uncertainty_y**2)
        unc_color = Colors.SUCCESS if unc < 5 else Colors.WARNING if unc < 15 else Colors.ERROR
        set_t("UNCERTAINTY", f"({s.tracking.uncertainty_x:.1f}, {s.tracking.uncertainty_y:.1f})", unc_color)

        # Pan/Tilt
        set_t("PAN", f"{s.camera.pan_deg:+.2f}\u00b0", Colors.TEXT)
        set_t("TILT", f"{s.camera.tilt_deg:+.2f}\u00b0", Colors.TEXT)

        # FPS
        fps_color = Colors.SUCCESS if s.camera.fps >= 20 else Colors.WARNING if s.camera.fps >= 10 else Colors.ERROR
        set_t("FPS", f"{s.camera.fps:.1f}", fps_color)

        # Latency
        lat_color = Colors.SUCCESS if s.performance.processing_ms < 50 else Colors.WARNING if s.performance.processing_ms < 100 else Colors.ERROR
        set_t("LATENCY", f"{s.performance.processing_ms:.1f}ms", lat_color)

    def paintEvent(self, event) -> None:
        # All overlays are baked into the displayed pixmap by
        # _compose_overlays (anything painted here on the parent frame
        # would be covered by the image QLabel child).
        super().paintEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-layout is handled by Qt
