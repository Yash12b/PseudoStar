from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def format_optical_range(status: str, range_m: float) -> str:
    """Format the link range readout.

    Range is genuine 3D geometry only when the link engine has two
    active terminals (SIMULATION mode). Otherwise — NO_LINK or no
    positive range — report N/A instead of a fabricated number
    (monocular VIDEO/LIVE has no depth).
    """
    if status == "NO_LINK" or range_m <= 0:
        return "RANGE: N/A"
    return f"RANGE: {range_m:.0f} m"


class HUDPanel(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel(f"<b>{title}</b>")
        self.layout.addWidget(self.title_label)
        self.setStyleSheet("background-color: rgba(20, 30, 40, 200); color: #FFF; padding: 5px;")
        self.setAttribute(Qt.WA_StyledBackground, True)


class TrackingHUD(HUDPanel):
    def __init__(self, parent=None):
        super().__init__("FSOC TRACKING", parent)
        self.target_label = QLabel("TARGET: NONE")
        self.state_label = QLabel("STATE: NO_LINK")
        self.error_label = QLabel("ERROR: 0.0 px")
        self.locked_label = QLabel("LOCKED: NO")
        self.layout.addWidget(self.target_label)
        self.layout.addWidget(self.state_label)
        self.layout.addWidget(self.error_label)
        self.layout.addWidget(self.locked_label)

    def update_state(self, state):
        if hasattr(state, "tracked_target_id") and state.tracked_target_id is not None:
            self.target_label.setText(f"TARGET: OBJ-{state.tracked_target_id}")
        else:
            self.target_label.setText("TARGET: NONE")
        self.state_label.setText(f"STATE: {state.tracking.state}")
        self.error_label.setText(f"ERROR: {state.tracking.residual:.2f} px")
        self.locked_label.setText(f"LOCKED: {'YES' if state.tracking.locked else 'NO'}")


class AIHUD(HUDPanel):
    def __init__(self, parent=None):
        super().__init__("AI COMMAND", parent)
        self.situation_label = QLabel("SITUATION: --")
        self.action_label = QLabel("ACTION: --")
        self.confidence_label = QLabel("CONFIDENCE: --")
        self.prediction_label = QLabel("PREDICTION: --")
        self.risk_label = QLabel("FAILURE RISK: 0%")
        self.fallback_label = QLabel("FALLBACK: NO")
        self.explanation_label = QLabel("REASON: --")
        self.layout.addWidget(self.situation_label)
        self.layout.addWidget(self.action_label)
        self.layout.addWidget(self.confidence_label)
        self.layout.addWidget(self.prediction_label)
        self.layout.addWidget(self.risk_label)
        self.layout.addWidget(self.fallback_label)
        self.layout.addWidget(self.explanation_label)

    def update_state(self, state):
        ai = state.ai_state
        self.situation_label.setText(f"SITUATION: {ai.situation.upper()}")
        self.action_label.setText(f"ACTION: {ai.action.upper()}")
        self.confidence_label.setText(f"CONFIDENCE: {ai.confidence:.3f}")
        self.prediction_label.setText(
            f"PREDICTION: dx={ai.prediction_dx:.1f} dy={ai.prediction_dy:.1f} "
            f"±{ai.prediction_uncertainty_x:.1f}px @ {ai.prediction_horizon_s*1000:.0f}ms"
        )
        self.risk_label.setText(f"FAILURE RISK: {ai.failure_risk:.1%}")
        self.fallback_label.setText(f"FALLBACK: {'ACTIVE' if ai.fallback_active else 'NO'}")
        self.explanation_label.setText(f"REASON: {ai.explanation}")


class LinkHUD(HUDPanel):
    def __init__(self, parent=None):
        super().__init__("OPTICAL LINK", parent)
        self.status_label = QLabel("STATUS: NO_LINK")
        self.align_label = QLabel("ALIGNMENT: 0%")
        self.range_label = QLabel("RANGE: N/A")
        self.error_label = QLabel("ANGULAR ERROR: 0 deg")
        self.quality_label = QLabel("LINK QUALITY: 0%")
        self.scan_label = QLabel("SCAN: --")
        self.atmo_label = QLabel("ATMO: CLEAR")
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.align_label)
        self.layout.addWidget(self.range_label)
        self.layout.addWidget(self.error_label)
        self.layout.addWidget(self.quality_label)
        self.layout.addWidget(self.scan_label)
        self.layout.addWidget(self.atmo_label)

    def update_state(self, state):
        if hasattr(state, "optical_link"):
            link = state.optical_link
            self.status_label.setText(f"STATUS: {link.status}")
            self.align_label.setText(f"ALIGNMENT: {link.beam_alignment_percent:.1f}%")
            self.range_label.setText(format_optical_range(link.status, link.range_m))
            self.error_label.setText(f"ANGULAR ERROR: {link.angular_error_deg:.2f} deg")
            self.quality_label.setText(f"LINK QUALITY: {link.link_quality_percent:.1f}%")
        if hasattr(state, "scan_active"):
            scan_text = f"SCANNING: {state.scan_radius:.0f}m" if state.scan_active else "SCAN: IDLE"
            self.scan_label.setText(scan_text)
        if hasattr(state, "atmospheric_disturbance"):
            self.atmo_label.setText(f"ATMO: {state.atmospheric_disturbance.upper()}")


class CommunicationHUD(HUDPanel):
    def __init__(self, parent=None):
        super().__init__("COMMUNICATION", parent)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(120)
        self.log_display.setStyleSheet("font-size: 9px; background-color: rgba(8, 15, 21, 200);")
        self.layout.addWidget(self.log_display)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Message to send...")
        self.send_btn = QPushButton("SEND")
        self.send_btn.setStyleSheet("QPushButton { font-weight: bold; min-width: 50px; }")
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        self.layout.addLayout(input_layout)

        self._comm_engine = None
        self._current_state = None

    def set_comm_engine(self, engine):
        self._comm_engine = engine
        self.send_btn.clicked.connect(self._on_send)

    def _on_send(self):
        payload = self.input_field.text().strip()
        if not payload or self._current_state is None:
            return
        source = "TERM_A"
        dest = "TERM_B"
        if self._current_state.designated_beacon_id is not None:
            dest = f"BEACON_{self._current_state.designated_beacon_id}"
        if self._comm_engine is not None:
            msg = self._comm_engine.create_message(source, dest, payload, self._current_state.elapsed_s)
            self._comm_engine.send_message(msg)
        else:
            # Fallback: use beacon AI directly
            self._current_state.add_event(f"MSG -> {dest}: {payload}", "COMM")
        self.input_field.clear()

    def update_state(self, state):
        self._current_state = state
        if hasattr(state, "messages"):
            log_text = ""
            for msg in state.messages[-15:]:
                status_icon = {"QUEUED": "[Q]", "TRANSMITTING": "[T]", "DELIVERED": "[D]", "FAILED": "[X]", "sending": "[T]", "sent": "[D]", "received": "[R]", "failed": "[X]"}.get(msg.status, "[?]")
                log_text += f"{status_icon} {msg.source} -> {msg.destination}: {msg.payload[:40]}\n"
            self.log_display.setText(log_text)
