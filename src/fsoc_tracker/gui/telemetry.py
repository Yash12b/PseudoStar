"""Telemetry panel — compact system state readout."""

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


class TelemetryPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self._labels: dict[str, QLabel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        sections = [
            ("SYSTEM", ["Mode", "State", "Elapsed", "Session"]),
            ("INPUT", ["Source", "Res", "FPS", "Timestamp"]),
            ("PERCEPTION", ["Backend", "Status", "Confidence", "Proc MS"]),
            ("TRACKING", ["State", "Lock", "X/Y", "Vx/Vy", "Uncert", "Residual", "Ang Err", "FOV"]),
            ("CONTROL", ["Pan", "Tilt", "Cmd Pan", "Cmd Tilt", "Sat"]),
            ("PERFORMANCE", ["FPS", "Latency", "Frames"]),
        ]

        for section_name, fields in sections:
            grp = QGroupBox(section_name)
            grp.setStyleSheet(f"QGroupBox {{ color: {Colors.ACCENT}; font-size: 10px; font-weight: bold; border: 1px solid {Colors.PANEL_BORDER}; border-radius: 3px; margin-top: 8px; padding-top: 10px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 3px; }}")
            grid = QGridLayout(grp)
            grid.setContentsMargins(4, 4, 4, 4)
            grid.setSpacing(2)
            for i, field_name in enumerate(fields):
                lbl_name = QLabel(field_name)
                lbl_name.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
                lbl_val = QLabel("--")
                lbl_val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
                lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                key = f"{section_name}_{field_name}"
                self._labels[key] = lbl_val
                grid.addWidget(lbl_name, i, 0)
                grid.addWidget(lbl_val, i, 1)
            layout.addWidget(grp)

        layout.addStretch()

    def update_state(self, state: ApplicationViewState) -> None:
        def set_text(key: str, text: str, color: str | None = None) -> None:
            if key in self._labels:
                self._labels[key].setText(text)
                if color:
                    self._labels[key].setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold; background: transparent; border: none;")

        s = state
        mode_colors = {
            "simulation": Colors.ACCENT, "video": Colors.WARNING, "live": Colors.ERROR,
        }
        set_text("SYSTEM_Mode", s.system_mode.value.upper(), mode_colors.get(s.system_mode.value, Colors.TEXT))
        state_colors = {
            "IDLE": Colors.MUTED, "INITIALIZING": Colors.WARNING,
            "RUNNING": Colors.SUCCESS, "PAUSED": Colors.WARNING, "ERROR": Colors.ERROR,
        }
        set_text("SYSTEM_State", s.system_state.value, state_colors.get(s.system_state.value, Colors.TEXT))
        set_text("SYSTEM_Elapsed", f"{s.elapsed_s:.2f}s")
        set_text("SYSTEM_Session", s.session_id[:8] if s.session_id else "--")

        set_text("INPUT_Source", s.source_info[:16] if s.source_info else s.system_mode.value.upper())
        set_text("INPUT_Res", f"{s.camera.image_width}x{s.camera.image_height}")
        set_text("INPUT_FPS", f"{s.camera.fps:.1f}")
        set_text("INPUT_Timestamp", "source")

        set_text("PERCEPTION_Backend", s.perception.backend[:12])
        det_color = Colors.SUCCESS if s.perception.detected else Colors.MUTED
        set_text("PERCEPTION_Status", "DETECTED" if s.perception.detected else "NONE", det_color)
        set_text("PERCEPTION_Confidence", f"{s.perception.confidence:.2f}")
        set_text("PERCEPTION_Proc MS", f"{s.perception.processing_ms:.1f}")

        track_colors = {
            "NO_TRACK": Colors.MUTED, "SEARCHING": Colors.WARNING,
            "ACQUIRING": Colors.WARNING, "TRACKING": Colors.SUCCESS,
            "LOST": Colors.ERROR, "REACQUIRING": Colors.WARNING,
        }
        set_text("TRACKING_State", s.tracking.state, track_colors.get(s.tracking.state, Colors.TEXT))
        lock_color = Colors.SUCCESS if s.tracking.locked else Colors.MUTED
        set_text("TRACKING_Lock", "YES" if s.tracking.locked else "NO", lock_color)
        set_text("TRACKING_X/Y", f"{s.tracking.estimated_x:.1f}, {s.tracking.estimated_y:.1f}")
        set_text("TRACKING_Vx/Vy", f"{s.tracking.velocity_x:.1f}, {s.tracking.velocity_y:.1f}")
        set_text("TRACKING_Uncert", f"{s.tracking.uncertainty_x:.1f}, {s.tracking.uncertainty_y:.1f}")
        set_text("TRACKING_Residual", f"{s.tracking.residual:.1f}")
        set_text("TRACKING_Ang Err", f"{s.optical_link.angular_error_deg:.2f}\u00b0")
        # FOV status derived from live observables only (detection pixels
        # or current estimate against decoded frame dimensions).
        img_w = max(s.camera.image_width, 1)
        img_h = max(s.camera.image_height, 1)
        if s.perception.detected:
            fov_txt, fov_color = "IN FOV", Colors.SUCCESS
        elif s.tracking.state in ("TRACKING", "ACQUIRING", "REACQUIRING"):
            ex, ey = s.tracking.estimated_x, s.tracking.estimated_y
            if 0.0 <= ex < img_w and 0.0 <= ey < img_h:
                fov_txt, fov_color = "IN FOV (EST)", Colors.SUCCESS
            else:
                fov_txt, fov_color = "OUT OF FOV", Colors.ERROR
        else:
            fov_txt, fov_color = "NO TARGET", Colors.MUTED
        set_text("TRACKING_FOV", fov_txt, fov_color)

        set_text("CONTROL_Pan", f"{s.control.pan_deg:+.2f}\u00b0")
        set_text("CONTROL_Tilt", f"{s.control.tilt_deg:+.2f}\u00b0")
        set_text("CONTROL_Cmd Pan", f"{s.control.pan_command:+.2f}")
        set_text("CONTROL_Cmd Tilt", f"{s.control.tilt_command:+.2f}")
        sat_color = Colors.WARNING if s.control.pan_saturated or s.control.tilt_saturated else Colors.MUTED
        set_text("CONTROL_Sat", "YES" if s.control.pan_saturated or s.control.tilt_saturated else "NO", sat_color)

        set_text("PERFORMANCE_FPS", f"{s.performance.pipeline_fps:.1f}")
        set_text("PERFORMANCE_Latency", f"{s.performance.processing_ms:.1f}ms")
        set_text("PERFORMANCE_Frames", str(s.performance.frames_processed))
