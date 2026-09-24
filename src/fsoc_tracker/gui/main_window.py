"""Main application window — 5-page mission workflow.

Pages:
  1. MISSION SETUP   — 3D world + config + START
  2. CAMERA TRACKING — primary operational view (DEFAULT)
  3. ANALYSIS        — AI state, search, event log
  4. WORLD           — full 3D simulation context
  5. BENCHMARK       — source/scene/method + metrics + export

Navigation: left sidebar with page buttons.
Right sidebar: context-sensitive per active page.
"""

from __future__ import annotations

import json

from fsoc_tracker.gui.controller import ApplicationController
from fsoc_tracker.gui.state import ApplicationViewState, SystemMode
from fsoc_tracker.gui.theme import Colors, apply_theme
from fsoc_tracker.gui.worker import ProcessingWorker

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import Qt, QTimer  # type: ignore
    from PyQt5.QtGui import QAction, QKeySequence  # type: ignore
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,  # type: ignore
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,  # type: ignore
        QSpinBox,
        QStatusBar,
        QTabWidget,  # type: ignore
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

from fsoc_tracker.gui.about import AboutDialog
from fsoc_tracker.gui.ai_panel import AIPanel
from fsoc_tracker.gui.benchmark_panel import BenchmarkPanel
from fsoc_tracker.gui.camera_workspace import CameraTrackingWorkspace
from fsoc_tracker.gui.controls import ControlPanel, _shape_label_to_value
from fsoc_tracker.gui.event_log import EventLogPanel
from fsoc_tracker.gui.hud_panels import AIHUD, CommunicationHUD, LinkHUD
from fsoc_tracker.gui.plots import PlotsWidget
from fsoc_tracker.gui.scorecard import ScorecardWidget
from fsoc_tracker.gui.telemetry import TelemetryPanel
from fsoc_tracker.gui.world_view import WorldViewWidget

# ---------------------------------------------------------------------------
#  Page indices
# ---------------------------------------------------------------------------
PAGE_MISSION_SETUP = 0
PAGE_CAMERA_TRACKING = 1
PAGE_ANALYSIS = 2
PAGE_WORLD = 3
PAGE_BENCHMARK = 4

PAGE_TITLES = [
    "MISSION SETUP",
    "CAMERA TRACKING",
    "ANALYSIS",
    "WORLD",
    "BENCHMARK",
]

PAGE_SUBTITLES = [
    "Configure and launch",
    "Operational view",
    "AI intelligence",
    "3D context",
    "Performance metrics",
]

# ---------------------------------------------------------------------------
#  Layout presets — (central page, sidebar page) pairs. Each layout is a
#  genuinely different information hierarchy composed live without
#  restarting the simulation: the worker thread keeps running while only
#  widget visibility changes.
# ---------------------------------------------------------------------------
LAYOUT_IMMERSIVE_EXPLORER = "IMMERSIVE EXPLORER"
LAYOUT_TRACKING_CONSOLE = "TRACKING CONSOLE"
LAYOUT_AI_COMMAND = "AI COMMAND"
LAYOUT_OPTICAL_LAB = "OPTICAL LAB"
LAYOUT_BENCHMARK_LAB = "BENCHMARK LAB"

LAYOUTS: dict[str, tuple[int, int, str]] = {
    LAYOUT_IMMERSIVE_EXPLORER: (
        PAGE_WORLD, PAGE_WORLD, "3D world dominant"),
    LAYOUT_TRACKING_CONSOLE: (
        PAGE_CAMERA_TRACKING, PAGE_CAMERA_TRACKING,
        "Camera + tracking diagnostics dominant"),
    LAYOUT_AI_COMMAND: (
        PAGE_ANALYSIS, PAGE_ANALYSIS, "AI/prediction/risk/search dominant"),
    LAYOUT_OPTICAL_LAB: (
        PAGE_WORLD, PAGE_ANALYSIS, "Beam/link/terminal alignment dominant"),
    LAYOUT_BENCHMARK_LAB: (
        PAGE_BENCHMARK, PAGE_BENCHMARK, "Metrics/comparisons dominant"),
}

# ---------------------------------------------------------------------------
#  Mission flow strip — pipeline stages, all lamps driven by runtime state.
#  Multiple lamps may be active at once (stages, not exclusive phases).
# ---------------------------------------------------------------------------
FLOW_STAGES = [
    "SETUP",
    "START",
    "CAMERA POV",
    "SEARCH",
    "ACQUIRE",
    "TRACK",
    "PREDICT",
    "CONTROL",
    "RECOVER",
]

FLOW_ACTIVE_COLORS = {
    "SETUP": Colors.TEXT,
    "START": Colors.ACCENT,
    "CAMERA POV": Colors.ACCENT,
    "SEARCH": Colors.WARNING,
    "ACQUIRE": Colors.WARNING,
    "TRACK": Colors.SUCCESS,
    "PREDICT": Colors.ACCENT_BLUE,
    "CONTROL": Colors.ACCENT,
    "RECOVER": Colors.ERROR,
}

FLOW_SEARCH_STATES = ("NO_TRACK", "SEARCHING")
FLOW_RECOVER_STATES = ("LOST", "REACQUIRING")
FLOW_START_FRAMES = 5
FLOW_CONTROL_EPS_DEG_S = 0.01

_NAV_STYLE = """
QPushButton {{
    color: {color};
    font-size: 9px;
    font-weight: bold;
    padding: 8px 4px;
    background: {bg};
    border: none;
    border-left: 3px solid {border};
    text-align: center;
    border-radius: 0px;
}}
QPushButton:hover {{
    background: {hover_bg};
}}
"""
_SIDEBAR_STYLE = """
QFrame#RightSidebar {{
    background: rgba(13,20,28,240);
    border-left: 1px solid #20303d;
}}
QGroupBox {{
    color: #22d3ee;
    font-size: 9px;
    font-weight: bold;
    border: 1px solid #20303d;
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._controller = ApplicationController()
        self._worker: ProcessingWorker | None = None
        self._state = ApplicationViewState()
        self._current_page = PAGE_CAMERA_TRACKING
        self._current_layout = LAYOUT_TRACKING_CONSOLE
        # Generic world under configuration (CREATE WORLD actions build
        # this; START in simulation mode requires it — no scene presets).
        self._world_config = None
        # Last successfully opened video file (reused by START in VIDEO mode).
        self._last_video_path: str | None = None

        self.setWindowTitle("FSOC OPTICAL TRACKING SYSTEM | SIH26169")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)

        self.setStyleSheet(apply_theme(self))

        self._build_menu()
        self._build_ui()
        self._build_status_bar()
        self._install_shortcuts()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_from_worker)
        self._refresh_timer.start(33)

    # ------------------------------------------------------------------
    #  UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Left navigation bar ---
        root.addWidget(self._build_nav_bar())

        # --- Main content area ---
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top status bar (thin)
        content_layout.addWidget(self._build_top_bar())

        # Mission flow strip (pipeline stages, runtime-driven lamps)
        self._flow_strip = self._build_flow_strip()
        content_layout.addWidget(self._flow_strip)

        # Body: pages + right sidebar
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._body_stack = QStackedWidget()
        self._build_all_pages()
        body.addWidget(self._body_stack, 1)

        self._right_sidebar = self._build_right_sidebar()
        body.addWidget(self._right_sidebar)

        content_layout.addLayout(body, 1)
        root.addWidget(content, 1)

        # Start on CAMERA TRACKING page
        self._navigate_to(PAGE_CAMERA_TRACKING)

    # ------------------------------------------------------------------
    #  Navigation bar (left)
    # ------------------------------------------------------------------
    def _build_nav_bar(self) -> QFrame:
        nav = QFrame()
        nav.setFixedWidth(88)
        nav.setStyleSheet("QFrame { background: #080c12; border-right: 1px solid #20303d; }")

        layout = QVBoxLayout(nav)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        # Logo / title
        logo = QLabel("FSOC")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 11px; font-weight: bold; "
            f"background: transparent; border: none; padding: 6px 0;"
        )
        layout.addWidget(logo)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #20303d;")
        layout.addWidget(sep)
        layout.addSpacing(4)

        # Navigation buttons (short labels fit the narrow rail; full
        # titles remain in tooltips and the window title).
        self._nav_buttons: list[QPushButton] = []
        nav_labels = ["SETUP", "CAMERA", "ANALYSIS", "WORLD", "BENCH"]
        for i, (title, short) in enumerate(zip(PAGE_TITLES, nav_labels, strict=True)):
            btn = QPushButton(f"{short}")
            btn.setToolTip(title)
            btn.setFixedHeight(48)
            btn.setProperty("page_index", i)
            btn.clicked.connect(lambda checked, idx=i: self._navigate_to(idx))
            self._nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Status indicator at bottom
        self._nav_status = QLabel("IDLE")
        self._nav_status.setAlignment(Qt.AlignCenter)
        self._nav_status.setStyleSheet(
            f"color: {Colors.MUTED}; font-size: 8px; font-weight: bold; "
            f"background: transparent; border: none; padding: 4px;"
        )
        layout.addWidget(self._nav_status)

        return nav

    def _update_nav_buttons(self) -> None:
        for i, btn in enumerate(self._nav_buttons):
            is_active = i == self._current_page
            if is_active:
                btn.setStyleSheet(_NAV_STYLE.format(
                    color=Colors.ACCENT, bg="rgba(34,211,238,0.12)",
                    border=Colors.ACCENT, hover_bg="rgba(34,211,238,0.18)",
                ))
            else:
                btn.setStyleSheet(_NAV_STYLE.format(
                    color=Colors.MUTED, bg="transparent",
                    border="transparent", hover_bg="rgba(255,255,255,0.04)",
                ))

    def _navigate_to(self, page_index: int) -> None:
        self._current_page = page_index
        self._body_stack.setCurrentIndex(page_index)
        self._update_nav_buttons()
        self._update_right_sidebar_content()
        self._current_layout = "CUSTOM"
        for lname, (central, sidebar, _desc) in LAYOUTS.items():
            if central == page_index and sidebar == page_index:
                self._current_layout = lname
                break
        self._update_layout_actions()
        title = PAGE_TITLES[page_index]
        self.setWindowTitle(f"FSOC OPTICAL TRACKING SYSTEM | SIH26169 — {title}")

    def set_layout(self, name: str) -> bool:
        """Apply a named layout preset without restarting the simulation.

        Only widget visibility changes; the worker thread, scene, and
        tracking state are untouched. Returns True if applied.
        """
        spec = LAYOUTS.get(name)
        if spec is None:
            return False
        central, sidebar, _desc = spec
        self._current_page = central
        self._current_layout = name
        self._body_stack.setCurrentIndex(central)
        self._sidebar_stack.setCurrentIndex(sidebar)
        self._update_nav_buttons()
        self._update_layout_actions()
        title = PAGE_TITLES[central]
        self.setWindowTitle(f"FSOC OPTICAL TRACKING SYSTEM | SIH26169 — {title}")
        return True

    @property
    def current_layout(self) -> str:
        """Active layout preset name (or CUSTOM after manual navigation)."""
        return getattr(self, "_current_layout", LAYOUT_TRACKING_CONSOLE)

    def _update_layout_actions(self) -> None:
        """Reflect the active layout preset in menu checkmarks."""
        if not hasattr(self, "_layout_actions"):
            return
        current = self.current_layout
        for name, act in self._layout_actions.items():
            act.setChecked(name == current)

    def toggle_floating_panel(self, name: str) -> bool:
        """Toggle a floating utility panel (dockable/closable/resizable).

        Content widgets are live: they receive the same runtime state
        updates as their sidebar counterparts. Returns visibility.
        """
        try:
            from PySide6.QtCore import Qt as _Qt
            from PySide6.QtWidgets import QDockWidget as _Dock
        except ImportError:
            from PyQt5.QtCore import Qt as _Qt  # type: ignore
            from PySide6.QtWidgets import QDockWidget as _Dock  # type: ignore
        from fsoc_tracker.gui.ai_panel import AIPanel as _AIPanel
        from fsoc_tracker.gui.event_log import EventLogPanel as _EventLog
        from fsoc_tracker.gui.hud_panels import LinkHUD as _LinkHUD
        from fsoc_tracker.gui.telemetry import TelemetryPanel as _Telemetry

        if not hasattr(self, "_floating_docks"):
            self._floating_docks: dict = {}
            self._floating_panels: dict = {}
        dock = self._floating_docks.get(name)
        if dock is None:
            builders = {
                "AI": _AIPanel, "LINK": _LinkHUD,
                "TELEMETRY": _Telemetry, "EVENTS": _EventLog,
            }
            builder = builders.get(name)
            if builder is None:
                return False
            content = builder()
            dock = _Dock(name, self)
            dock.setWidget(content)
            dock.setFloating(True)
            dock.resize(300, 400)
            self.addDockWidget(_Qt.RightDockWidgetArea, dock)
            self._floating_docks[name] = dock
            self._floating_panels[name] = content
            content.update_state(self._state)
            dock.setVisible(True)
        else:
            dock.setVisible(not dock.isVisible())
        visible = bool(dock.isVisible())
        act = getattr(self, "_floating_actions", {}).get(name)
        if act is not None:
            act.setChecked(visible)
        return visible

    def _update_floating_panels(self, s: ApplicationViewState) -> None:
        """Push runtime state into visible floating panels."""
        for name, dock in getattr(self, "_floating_docks", {}).items():
            try:
                if dock.isVisible():
                    self._floating_panels[name].update_state(s)
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  Top status bar (above pages)
    # ------------------------------------------------------------------
    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet("QFrame { background: #0a1018; border-bottom: 1px solid #20303d; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 2, 12, 2)
        layout.setSpacing(12)

        # Mission identity: program + operating picture
        identity = QVBoxLayout()
        identity.setContentsMargins(0, 0, 0, 0)
        identity.setSpacing(0)
        mission_title = QLabel("FSOC \u00b7 COARSE ALIGNMENT")
        mission_title.setStyleSheet(
            f"color: {Colors.TEXT}; font-size: 9px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        mission_sub = QLabel("VIRTUAL CAMERA TRACKING")
        mission_sub.setStyleSheet(
            f"color: {Colors.MUTED}; font-size: 8px; "
            f"background: transparent; border: none;"
        )
        identity.addWidget(mission_title)
        identity.addWidget(mission_sub)
        identity_widget = QWidget()
        identity_widget.setLayout(identity)
        identity_widget.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(identity_widget)

        layout.addSpacing(4)

        # Playback controls
        self._btn_start = QPushButton("START")
        self._btn_start.setFixedHeight(22)
        self._btn_start.setStyleSheet(
            f"QPushButton {{ color: {Colors.SUCCESS}; font-size: 8px; font-weight: bold; "
            f"padding: 2px 10px; background: transparent; border: 1px solid {Colors.SUCCESS}; "
            f"border-radius: 3px; }} "
            f"QPushButton:hover {{ background: {Colors.SUCCESS}; color: #000; }}"
        )
        self._btn_start.clicked.connect(self._on_start)
        layout.addWidget(self._btn_start)

        self._btn_pause = QPushButton("PAUSE")
        self._btn_pause.setFixedHeight(22)
        self._btn_pause.setStyleSheet(
            f"QPushButton {{ color: {Colors.WARNING}; font-size: 8px; font-weight: bold; "
            f"padding: 2px 10px; background: transparent; border: 1px solid {Colors.WARNING}; "
            f"border-radius: 3px; }} "
            f"QPushButton:hover {{ background: {Colors.WARNING}; color: #000; }}"
        )
        self._btn_pause.clicked.connect(self._on_pause)
        layout.addWidget(self._btn_pause)

        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setFixedHeight(22)
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ color: {Colors.ERROR}; font-size: 8px; font-weight: bold; "
            f"padding: 2px 10px; background: transparent; border: 1px solid {Colors.ERROR}; "
            f"border-radius: 3px; }} "
            f"QPushButton:hover {{ background: {Colors.ERROR}; color: #000; }}"
        )
        self._btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(self._btn_stop)

        layout.addSpacing(8)

        # Source mode selector
        src_label = QLabel("SOURCE:")
        src_label.setStyleSheet(f"color: {Colors.MUTED}; font-size: 8px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(src_label)
        self._src_combo = QComboBox()
        self._src_combo.addItems(["SIMULATION", "VIDEO", "LIVE"])
        self._src_combo.setFixedWidth(100)
        self._src_combo.currentTextChanged.connect(self._on_source_changed)
        layout.addWidget(self._src_combo)

        # Visible video file loader — always present so the user never
        # has to hunt through menus or combos to open a video.
        self._btn_load_video = QPushButton("LOAD VIDEO")
        self._btn_load_video.setFixedHeight(22)
        self._btn_load_video.setStyleSheet(
            f"QPushButton {{ color: {Colors.CROSSHAIR}; font-size: 8px; font-weight: bold; "
            f"padding: 2px 10px; background: transparent; border: 1px solid {Colors.CROSSHAIR}; "
            f"border-radius: 3px; }} "
            f"QPushButton:hover {{ background: {Colors.CROSSHAIR}; color: #000; }}"
        )
        self._btn_load_video.clicked.connect(self._on_open_video)
        layout.addWidget(self._btn_load_video)

        layout.addSpacing(8)

        # World actions (replaces the retired Scene 1-15 selector)
        world_label = QLabel("WORLD:")
        world_label.setStyleSheet(f"color: {Colors.MUTED}; font-size: 8px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(world_label)
        self._world_combo = QComboBox()
        self._world_combo.setFixedWidth(180)
        self._world_combo.addItems([
            "No world",
            "New empty world",
            "Random world (seed)",
        ])
        self._world_combo.currentTextChanged.connect(self._on_world_action)
        layout.addWidget(self._world_combo)

        layout.addStretch()

        # Status telemetry
        self._header_state = QLabel("● STANDBY")
        self._header_state.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(self._header_state)

        layout.addSpacing(8)

        self._header_fps = QLabel("PROC -- FPS | LAT -- ms")
        self._header_fps.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
        layout.addWidget(self._header_fps)

        return bar

    # ------------------------------------------------------------------
    #  Mission flow strip — one lamp per pipeline stage.
    #  Every lamp reflects live runtime/view state; nothing is decorative.
    # ------------------------------------------------------------------
    def _build_flow_strip(self) -> QFrame:
        strip = QFrame()
        strip.setObjectName("FlowStrip")
        strip.setFixedHeight(28)
        strip.setStyleSheet(
            "QFrame#FlowStrip { background: #080d13; "
            "border-bottom: 1px solid #20303d; }"
        )
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(12, 2, 6, 2)
        layout.setSpacing(4)

        self._flow_lamps: dict[str, QLabel] = {}
        for i, stage in enumerate(FLOW_STAGES):
            if i > 0:
                sep = QLabel("\u203a")
                sep.setStyleSheet(
                    f"color: {Colors.CENTERLINE}; font-size: 10px; "
                    f"font-weight: bold; background: transparent; border: none;"
                )
                layout.addWidget(sep)
            lamp = QLabel(stage)
            lamp.setStyleSheet(
                f"color: {Colors.MUTED}; font-size: 8px; "
                f"background: transparent; border: none; padding: 2px 4px;"
            )
            self._flow_lamps[stage] = lamp
            layout.addWidget(lamp)

        layout.addStretch()

        self._flow_frame = QLabel("FRAME 0")
        self._flow_frame.setStyleSheet(
            f"color: {Colors.MUTED}; font-size: 8px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(self._flow_frame)

        # Initial state: idle mission on the camera page.
        self._update_flow_strip(self._state)
        return strip

    def _set_flow_lamp(self, stage: str, active: bool) -> None:
        lamp = self._flow_lamps.get(stage)
        if lamp is None:
            return
        if active:
            color = FLOW_ACTIVE_COLORS.get(stage, Colors.ACCENT)
            lamp.setStyleSheet(
                f"color: {color}; font-size: 8px; font-weight: bold; "
                f"background: rgba(255,255,255,0.04); border: none; "
                f"border-bottom: 2px solid {color}; padding: 2px 4px;"
            )
        else:
            lamp.setStyleSheet(
                f"color: {Colors.MUTED}; font-size: 8px; "
                f"background: transparent; border: none; padding: 2px 4px;"
            )

    def _flow_active_stages(self, s: ApplicationViewState) -> dict[str, bool]:
        """Derive active pipeline stages from runtime + view state."""
        running = s.system_state.value == "RUNNING"
        frames = s.performance.frames_processed
        trk = s.tracking.state
        pred_only = bool(s.tracking.prediction_only)
        cmd_mag = abs(s.control.pan_command) + abs(s.control.tilt_command)
        # Tracking/search/recovery lamps only mean something while the
        # loop is running; at idle the strip shows SETUP (+ view page).
        live = running or s.system_state.value == "PAUSED"
        return {
            "SETUP": s.system_state.value == "IDLE",
            "START": bool(running and frames < FLOW_START_FRAMES),
            "CAMERA POV": self._current_page == PAGE_CAMERA_TRACKING,
            "SEARCH": bool(live and trk in FLOW_SEARCH_STATES),
            "ACQUIRE": bool(live and trk == "ACQUIRING"),
            "TRACK": bool(live and trk == "TRACKING" and not pred_only),
            "PREDICT": bool(live and pred_only),
            "CONTROL": bool(live and cmd_mag > FLOW_CONTROL_EPS_DEG_S),
            "RECOVER": bool(live and trk in FLOW_RECOVER_STATES),
        }

    def _update_flow_strip(self, s: ApplicationViewState) -> None:
        if not hasattr(self, "_flow_lamps"):
            return
        for stage, active in self._flow_active_stages(s).items():
            self._set_flow_lamp(stage, active)
        self._flow_frame.setText(f"FRAME {s.performance.frames_processed}")

    # ------------------------------------------------------------------
    #  Pages
    # ------------------------------------------------------------------
    def _build_all_pages(self) -> None:
        # Shared panels (created first — pages reference them)
        self._control_panel = ControlPanel()
        self._control_panel.start_clicked.connect(self._on_start)
        self._control_panel.pause_clicked.connect(self._on_pause)
        self._control_panel.step_clicked.connect(self._on_step)
        self._control_panel.stop_clicked.connect(self._on_stop)
        self._control_panel.reset_clicked.connect(self._on_stop)
        self._control_panel.config_changed.connect(self._on_config_changed)
        self._control_panel.seek_requested.connect(self._on_seek_video)

        self._plots = PlotsWidget()
        self._telemetry = TelemetryPanel()
        self._scorecard = ScorecardWidget()
        self._ai_panel = AIPanel()
        self._event_log = EventLogPanel()
        self._ai_hud = AIHUD()
        self._link_hud = LinkHUD()
        self._comm_hud = CommunicationHUD()

        # Page 0: Mission Setup
        self._mission_setup_page = self._build_mission_setup_page()
        self._body_stack.addWidget(self._mission_setup_page)

        # Page 1: Camera Tracking
        self._camera_workspace = CameraTrackingWorkspace()
        self._body_stack.addWidget(self._camera_workspace)

        # Page 2: Analysis
        self._analysis_page = self._build_analysis_page()
        self._body_stack.addWidget(self._analysis_page)

        # Page 3: World
        self._world_view = WorldViewWidget()
        self._world_view.target_selected.connect(self._on_target_selected)
        self._world_view.terminal_selected.connect(self._on_terminal_selected)
        self._world_view.scan_started.connect(self._on_start_scan)
        self._world_view.beacon_dragged.connect(self._on_beacon_dragged)
        self._world_view.terminal_placed.connect(self._on_terminal_placed)
        self._body_stack.addWidget(self._world_view)

        # Page 4: Benchmark
        self._benchmark_panel = BenchmarkPanel()
        self._body_stack.addWidget(self._benchmark_panel)

    # ----- Page 0: Mission Setup -----
    def _build_mission_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Full-width 3D world view. Mission configuration lives ONLY in
        # the right sidebar (_build_sidebar_mission) so every control is
        # wired exactly once (a previous duplicate panel left orphaned,
        # unwired widgets here).
        self._setup_world_view = WorldViewWidget()
        self._setup_world_view.target_selected.connect(self._on_target_selected)
        self._setup_world_view.terminal_selected.connect(self._on_terminal_selected)
        self._setup_world_view.scan_started.connect(self._on_start_scan)
        self._setup_world_view.beacon_dragged.connect(self._on_beacon_dragged)
        self._setup_world_view.terminal_placed.connect(self._on_terminal_placed)
        layout.addWidget(self._setup_world_view, 1)

        return page

    def _setup_tab_trajectory(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._traj_combo = QComboBox()
        self._traj_combo.addItems(["straight_line", "circular", "figure_8", "random", "random_walk", "spiral", "sinusoidal", "user_controlled"])
        layout.addWidget(QLabel("Trajectory:"), 0, 0)
        layout.addWidget(self._traj_combo, 0, 1)

        self._target_size = QDoubleSpinBox()
        self._target_size.setRange(5, 20)
        self._target_size.setValue(10)
        self._target_size.setSuffix(" px")
        layout.addWidget(QLabel("Target Size:"), 1, 0)
        layout.addWidget(self._target_size, 1, 1)

        # Beacon spot shape (PS: user-defined, default Square). "Spot"
        # keeps the legacy soft-Gaussian rendering; Square/Circular
        # render hard-edged spots of the selected geometry.
        self._target_shape = QComboBox()
        self._target_shape.addItems(["Spot (soft)", "Square", "Circular"])
        layout.addWidget(QLabel("Target Shape:"), 2, 0)
        layout.addWidget(self._target_shape, 2, 1)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        layout.addWidget(QLabel("Seed:"), 3, 0)
        layout.addWidget(self._seed_spin, 3, 1)

        self._sim_speed = QDoubleSpinBox()
        self._sim_speed.setRange(0.1, 10.0)
        self._sim_speed.setValue(1.0)
        self._sim_speed.setSingleStep(0.1)
        layout.addWidget(QLabel("Speed:"), 4, 0)
        layout.addWidget(self._sim_speed, 4, 1)

        layout.setRowStretch(5, 1)
        return w

    def _setup_tab_beacon(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Beacon Management"), 0, 0, 1, 2)

        btn_add = QPushButton("+ Add Beacon")
        btn_add.clicked.connect(self._on_add_beacon)
        layout.addWidget(btn_add, 1, 0, 1, 2)

        btn_remove = QPushButton("- Remove Beacon")
        btn_remove.clicked.connect(self._on_remove_beacon)
        layout.addWidget(btn_remove, 2, 0, 1, 2)

        self._beacon_list = QTextEdit()
        self._beacon_list.setReadOnly(True)
        self._beacon_list.setMaximumHeight(100)
        self._beacon_list.setPlaceholderText("No beacons added yet.\nClick world to select, then SET BEACON.")
        layout.addWidget(self._beacon_list, 3, 0, 1, 2)

        btn_select = QPushButton("Select Primary")
        btn_select.clicked.connect(self._on_set_beacon)
        layout.addWidget(btn_select, 4, 0, 1, 2)

        keys_hint = QLabel(
            "Move: A/D ±X · Q/E ±Y · W/S ±Z · arrows · Space hold · R reset · M maneuver\n"
            "Needs a user_controlled beacon (Trajectory combo + Add).")
        keys_hint.setWordWrap(True)
        keys_hint.setStyleSheet(
            f"color: {Colors.MUTED}; font-size: 8px; "
            f"background: transparent; border: none;")
        layout.addWidget(keys_hint, 5, 0, 1, 2)

        self._selected_label = QLabel("Selected: none (click a beacon or Terminal A in 3D)")
        self._selected_label.setWordWrap(True)
        self._selected_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 8px; font-weight: bold; "
            f"background: transparent; border: none;")
        layout.addWidget(self._selected_label, 6, 0, 1, 2)

        # --- Per-beacon editor ---
        self._beacon_editor = QWidget()
        elayout = QGridLayout(self._beacon_editor)
        elayout.setContentsMargins(0, 0, 0, 0)
        elayout.setSpacing(3)
        self._edit_traj = QComboBox()
        self._edit_traj.addItems(["straight_line", "circular", "figure_8", "random", "random_walk",
                                  "spiral", "sinusoidal", "user_controlled"])
        elayout.addWidget(QLabel("Motion:"), 0, 0)
        elayout.addWidget(self._edit_traj, 0, 1)
        self._edit_size = QDoubleSpinBox()
        self._edit_size.setRange(5, 20)
        self._edit_size.setValue(10)
        self._edit_size.setSuffix(" px")
        elayout.addWidget(QLabel("Size:"), 1, 0)
        elayout.addWidget(self._edit_size, 1, 1)
        self._edit_bright = QDoubleSpinBox()
        self._edit_bright.setRange(0.1, 3.0)
        self._edit_bright.setValue(1.0)
        self._edit_bright.setSingleStep(0.1)
        elayout.addWidget(QLabel("Bright:"), 2, 0)
        elayout.addWidget(self._edit_bright, 2, 1)
        self._edit_seed = QSpinBox()
        self._edit_seed.setRange(0, 99999)
        self._edit_seed.setValue(42)
        elayout.addWidget(QLabel("Seed:"), 3, 0)
        elayout.addWidget(self._edit_seed, 3, 1)
        self._edit_shape = QComboBox()
        self._edit_shape.addItems(["Spot (soft)", "Square", "Circular"])
        elayout.addWidget(QLabel("Shape:"), 4, 0)
        elayout.addWidget(self._edit_shape, 4, 1)
        self._edit_x = QDoubleSpinBox()
        self._edit_x.setRange(0, 2000)
        self._edit_y = QDoubleSpinBox()
        self._edit_y.setRange(0, 2000)
        self._edit_z = QDoubleSpinBox()
        self._edit_z.setRange(0, 2000)
        for spin in (self._edit_x, self._edit_y, self._edit_z):
            spin.setMaximum(2000.0)
        elayout.addWidget(QLabel("X:"), 5, 0)
        elayout.addWidget(self._edit_x, 5, 1)
        elayout.addWidget(QLabel("Y:"), 6, 0)
        elayout.addWidget(self._edit_y, 6, 1)
        elayout.addWidget(QLabel("Z:"), 7, 0)
        elayout.addWidget(self._edit_z, 7, 1)
        btn_apply_beacon = QPushButton("Apply to beacon")
        btn_apply_beacon.clicked.connect(self._on_apply_beacon_edit)
        elayout.addWidget(btn_apply_beacon, 8, 0, 1, 2)
        self._beacon_editor.setVisible(False)
        layout.addWidget(self._beacon_editor, 7, 0, 1, 2)

        # --- Terminal A editor ---
        self._terminal_editor = QWidget()
        tlayout = QGridLayout(self._terminal_editor)
        tlayout.setContentsMargins(0, 0, 0, 0)
        tlayout.setSpacing(3)
        self._term_x = QDoubleSpinBox()
        self._term_x.setRange(0, 2000)
        self._term_x.setValue(1000)
        self._term_y = QDoubleSpinBox()
        self._term_y.setRange(0, 2000)
        self._term_y.setValue(1000)
        self._term_z = QDoubleSpinBox()
        self._term_z.setRange(0, 2000)
        self._term_z.setValue(50)
        self._term_yaw = QDoubleSpinBox()
        self._term_yaw.setRange(-180, 180)
        self._term_pitch = QDoubleSpinBox()
        self._term_pitch.setRange(-90, 90)
        tlayout.addWidget(QLabel("X:"), 0, 0)
        tlayout.addWidget(self._term_x, 0, 1)
        tlayout.addWidget(QLabel("Y:"), 1, 0)
        tlayout.addWidget(self._term_y, 1, 1)
        tlayout.addWidget(QLabel("Z:"), 2, 0)
        tlayout.addWidget(self._term_z, 2, 1)
        tlayout.addWidget(QLabel("Yaw°:"), 3, 0)
        tlayout.addWidget(self._term_yaw, 3, 1)
        tlayout.addWidget(QLabel("Pitch°:"), 4, 0)
        tlayout.addWidget(self._term_pitch, 4, 1)
        btn_apply_term = QPushButton("Apply to Terminal A")
        btn_apply_term.clicked.connect(self._on_apply_terminal_edit)
        tlayout.addWidget(btn_apply_term, 5, 0, 1, 2)
        self._terminal_editor.setVisible(False)
        layout.addWidget(self._terminal_editor, 8, 0, 1, 2)

        layout.setRowStretch(9, 1)
        return w

    def _setup_tab_camera(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._cam_w = QSpinBox()
        self._cam_w.setRange(64, 3840)
        self._cam_w.setValue(640)
        layout.addWidget(QLabel("Width:"), 0, 0)
        layout.addWidget(self._cam_w, 0, 1)

        self._cam_h = QSpinBox()
        self._cam_h.setRange(48, 2160)
        self._cam_h.setValue(480)
        layout.addWidget(QLabel("Height:"), 1, 0)
        layout.addWidget(self._cam_h, 1, 1)

        self._hfov = QDoubleSpinBox()
        self._hfov.setRange(0.5, 30.0)
        self._hfov.setValue(4.0)
        self._hfov.setSuffix("\u00b0")
        layout.addWidget(QLabel("HFOV:"), 2, 0)
        layout.addWidget(self._hfov, 2, 1)

        self._vfov = QDoubleSpinBox()
        self._vfov.setRange(0.5, 30.0)
        self._vfov.setValue(3.0)
        self._vfov.setSuffix("\u00b0")
        layout.addWidget(QLabel("VFOV:"), 3, 0)
        layout.addWidget(self._vfov, 3, 1)

        self._max_pan = QDoubleSpinBox()
        self._max_pan.setRange(0.5, 20.0)
        self._max_pan.setValue(5.0)
        self._max_pan.setSuffix("/s")
        layout.addWidget(QLabel("Max Pan:"), 4, 0)
        layout.addWidget(self._max_pan, 4, 1)

        self._max_tilt = QDoubleSpinBox()
        self._max_tilt.setRange(0.5, 20.0)
        self._max_tilt.setValue(5.0)
        self._max_tilt.setSuffix("/s")
        layout.addWidget(QLabel("Max Tilt:"), 5, 0)
        layout.addWidget(self._max_tilt, 5, 1)

        layout.setRowStretch(6, 1)
        return w

    def _setup_tab_disturbance(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        self._dist_enabled = QCheckBox("Enable")
        self._dist_enabled.toggled.connect(
            lambda _c: self._on_disturbance_ui_changed())
        layout.addWidget(self._dist_enabled, 0, 0, 1, 2)
        self._dist_preset = QComboBox()
        self._dist_preset.addItems(["clear", "light", "moderate", "severe", "extreme"])
        layout.addWidget(QLabel("Preset:"), 0, 2)
        layout.addWidget(self._dist_preset, 0, 3)

        row = 1
        effects = [
            ("Noise", "noise"), ("Fog", "fog"), ("Haze", "haze"), ("Rain", "rain"),
            ("Low Light", "low_light"), ("Jitter", "jitter"), ("Blur", "blur"),
            ("Motion Blur", "motion_blur"), ("Bri/Contrast", "brightness_contrast"),
            ("Platform", "platform_motion"), ("Disappear", "target_disappearance"),
            ("Distractors", "distractors"), ("Turbulence", "turbulence"),
            ("Gaussian", "noise_gaussian"), ("Salt&Pepper", "noise_salt_pepper"),
            ("Poisson", "noise_poisson"),
        ]
        self._dist_checks: dict[str, QCheckBox] = {}
        for i, (label, key) in enumerate(effects):
            cb = QCheckBox(label)
            cb.toggled.connect(lambda _c: self._on_disturbance_ui_changed())
            layout.addWidget(cb, row + i // 4, i % 4)
            self._dist_checks[key] = cb
        # Noise subtypes default on (legacy behavior: all types applied);
        # they gate which noise types the master Noise switch applies.
        for sub in ("noise_gaussian", "noise_salt_pepper", "noise_poisson"):
            self._dist_checks[sub].setChecked(True)
        self._dist_checks["noise_gaussian"].setToolTip("Additive Gaussian noise")
        self._dist_checks["noise_salt_pepper"].setToolTip("Salt & pepper (~10% per PS)")
        self._dist_checks["noise_poisson"].setToolTip("Photon-shot Poisson noise")

        self._dist_preset.currentTextChanged.connect(self._on_dist_preset_changed)
        self._dist_preset.currentTextChanged.connect(
            lambda _text: self._on_disturbance_ui_changed())

        row += 4
        self._dist_intensity = QDoubleSpinBox()
        self._dist_intensity.setRange(0.0, 1.0)
        self._dist_intensity.setValue(0.5)
        self._dist_intensity.setSingleStep(0.1)
        self._dist_intensity.valueChanged.connect(
            lambda _v: self._on_disturbance_ui_changed())
        layout.addWidget(QLabel("Intensity:"), row, 0)
        layout.addWidget(self._dist_intensity, row, 1, 1, 3)

        # --- PS numeric tuning (user-defined, clamped to PS maxima) ---
        row += 1
        self._dist_noise_sigma = QDoubleSpinBox()
        self._dist_noise_sigma.setRange(0.0, 20.0)
        self._dist_noise_sigma.setValue(5.0)
        self._dist_noise_sigma.setSingleStep(0.5)
        self._dist_noise_sigma.setSuffix(" σ")
        self._dist_noise_sigma.setToolTip("Gaussian noise std dev (PS max 20)")
        self._dist_noise_sigma.valueChanged.connect(
            lambda _v: self._on_disturbance_ui_changed())
        layout.addWidget(QLabel("Noise σ:"), row, 0)
        layout.addWidget(self._dist_noise_sigma, row, 1)

        self._dist_noise_density = QDoubleSpinBox()
        self._dist_noise_density.setRange(0.0, 0.5)
        self._dist_noise_density.setValue(0.05)
        self._dist_noise_density.setSingleStep(0.01)
        self._dist_noise_density.setToolTip("Salt & pepper density (PS ~10%)")
        self._dist_noise_density.valueChanged.connect(
            lambda _v: self._on_disturbance_ui_changed())
        layout.addWidget(QLabel("S&P dens:"), row, 2)
        layout.addWidget(self._dist_noise_density, row, 3)

        row += 1
        self._dist_jitter_amp = QDoubleSpinBox()
        self._dist_jitter_amp.setRange(0.0, 20.0)
        self._dist_jitter_amp.setValue(5.0)
        self._dist_jitter_amp.setSingleStep(0.5)
        self._dist_jitter_amp.setSuffix(" px")
        self._dist_jitter_amp.setToolTip("Camera jitter amplitude (PS max ±20 px/frame)")
        self._dist_jitter_amp.valueChanged.connect(
            lambda _v: self._on_disturbance_ui_changed())
        layout.addWidget(QLabel("Jitter amp:"), row, 0)
        layout.addWidget(self._dist_jitter_amp, row, 1)

        self._dist_platform_type = QComboBox()
        self._dist_platform_type.addItems(
            ["linear", "circular", "figure_eight", "spiral", "random"])
        self._dist_platform_type.setToolTip("Platform motion path (PS default linear)")
        self._dist_platform_type.currentTextChanged.connect(
            lambda _t: self._on_disturbance_ui_changed())
        layout.addWidget(QLabel("Platform:"), row, 2)
        layout.addWidget(self._dist_platform_type, row, 3)

        return w

    def _on_disturbance_ui_changed(self) -> None:
        """Push disturbance widget state to a live worker, if running.

        Pre-start values flow through _get_unified_config at START;
        this keeps mid-run adjustments live without a restart.
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.update_disturbance({
                "disturbance_preset": self._dist_preset.currentText(),
                "disturbance_enabled": self._dist_enabled.isChecked(),
                "disturbance_effects": {
                    key: cb.isChecked()
                    for key, cb in self._dist_checks.items()
                },
                "disturbance_intensity": self._dist_intensity.value(),
                **self._disturbance_tuning(),
            })

    def _disturbance_tuning(self) -> dict:
        """PS numeric disturbance tuning from widgets (PS-clamped)."""
        tuning: dict = {}
        if hasattr(self, "_dist_noise_sigma"):
            tuning["disturbance_noise_sigma"] = float(
                self._dist_noise_sigma.value())
        if hasattr(self, "_dist_noise_density"):
            tuning["disturbance_noise_density"] = float(
                self._dist_noise_density.value())
        if hasattr(self, "_dist_jitter_amp"):
            tuning["disturbance_jitter_amp"] = float(
                self._dist_jitter_amp.value())
        if hasattr(self, "_dist_platform_type"):
            tuning["disturbance_platform_type"] = str(
                self._dist_platform_type.currentText())
        return tuning

    def _on_dist_preset_changed(self, preset: str) -> None:
        presets = {
            "clear": {},
            "light": {"noise": True, "fog": True, "haze": True},
            "moderate": {"noise": True, "fog": True, "blur": True, "rain": True,
                         "motion_blur": True, "distractors": True},
            "severe": {"noise": True, "fog": True, "blur": True, "rain": True,
                        "motion_blur": True, "distractors": True, "target_disappearance": True,
                        "brightness_contrast": True, "low_light": True},
            "extreme": {"noise": True, "fog": True, "haze": True, "rain": True,
                         "blur": True, "motion_blur": True, "distractors": True,
                         "target_disappearance": True, "brightness_contrast": True,
                         "low_light": True, "jitter": True, "platform_motion": True,
                         "turbulence": True},
        }
        checks = presets.get(preset, {})
        for cb in self._dist_checks.values():
            cb.blockSignals(True)
        try:
            for key, cb in self._dist_checks.items():
                cb.setChecked(checks.get(key, False))
            # Noise subtypes follow the master Noise switch on preset
            # sync (they gate which noise types the master applies).
            noise_on = checks.get("noise", False)
            for sub in ("noise_gaussian", "noise_salt_pepper", "noise_poisson"):
                if sub in self._dist_checks:
                    self._dist_checks[sub].setChecked(noise_on)
        finally:
            for cb in self._dist_checks.values():
                cb.blockSignals(False)
        # One live push after the batch sync (instead of per-checkbox).
        self._on_disturbance_ui_changed()

    # ----- Page 2: Analysis -----
    def _build_analysis_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left column: AI + Search
        left = QVBoxLayout()
        left.setSpacing(6)

        # AI Situation
        ai_group = self._make_group("AI SITUATION")
        ai_layout = QGridLayout(ai_group)
        ai_layout.setSpacing(3)
        self._an_situation = QLabel("--")
        self._an_action = QLabel("--")
        self._an_confidence = QLabel("--")
        self._an_risk = QLabel("--")
        self._an_fallback = QLabel("--")
        self._an_explanation = QLabel("--")
        self._an_explanation.setWordWrap(True)
        for i, (lbl, val) in enumerate([
            ("SITUATION", self._an_situation), ("ACTION", self._an_action),
            ("CONFIDENCE", self._an_confidence), ("FAILURE RISK", self._an_risk),
            ("FALLBACK", self._an_fallback),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            ai_layout.addWidget(name_lbl, i, 0)
            ai_layout.addWidget(val, i, 1)
        expl_lbl = QLabel("EXPLANATION")
        expl_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
        ai_layout.addWidget(expl_lbl, 5, 0)
        ai_layout.addWidget(self._an_explanation, 5, 1)
        left.addWidget(ai_group)

        # Prediction
        pred_group = self._make_group("PREDICTION")
        pred_layout = QGridLayout(pred_group)
        pred_layout.setSpacing(3)
        self._an_pred_dx = QLabel("--")
        self._an_pred_dy = QLabel("--")
        self._an_pred_unc = QLabel("--")
        self._an_pred_horizon = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("DX", self._an_pred_dx), ("DY", self._an_pred_dy),
            ("UNCERTAINTY", self._an_pred_unc), ("HORIZON", self._an_pred_horizon),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            pred_layout.addWidget(name_lbl, i, 0)
            pred_layout.addWidget(val, i, 1)
        left.addWidget(pred_group)

        # Search
        search_group = self._make_group("SEARCH")
        search_layout = QGridLayout(search_group)
        search_layout.setSpacing(3)
        self._an_search_active = QLabel("--")
        self._an_search_strategy = QLabel("--")
        self._an_search_phase = QLabel("--")
        self._an_search_center = QLabel("--")
        self._an_search_radius = QLabel("--")
        self._an_search_time = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("ACTIVE", self._an_search_active), ("STRATEGY", self._an_search_strategy),
            ("PHASE", self._an_search_phase), ("CENTER", self._an_search_center),
            ("RADIUS", self._an_search_radius), ("TIME", self._an_search_time),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            search_layout.addWidget(name_lbl, i, 0)
            search_layout.addWidget(val, i, 1)
        left.addWidget(search_group)

        # ROI
        roi_group = self._make_group("ADAPTIVE ROI")
        roi_layout = QGridLayout(roi_group)
        roi_layout.setSpacing(3)
        self._an_roi_mode = QLabel("--")
        self._an_roi_radius = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("MODE", self._an_roi_mode), ("RADIUS", self._an_roi_radius),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            roi_layout.addWidget(name_lbl, i, 0)
            roi_layout.addWidget(val, i, 1)
        left.addWidget(roi_group)

        left.addStretch()
        layout.addLayout(left, 1)

        # Right column: Tracking + Event Log
        right = QVBoxLayout()
        right.setSpacing(6)

        # Tracking detail
        trk_group = self._make_group("TRACKING DETAIL")
        trk_layout = QGridLayout(trk_group)
        trk_layout.setSpacing(3)
        self._an_trk_state = QLabel("--")
        self._an_trk_locked = QLabel("--")
        self._an_trk_est = QLabel("--")
        self._an_trk_vel = QLabel("--")
        self._an_trk_unc = QLabel("--")
        self._an_trk_residual = QLabel("--")
        self._an_trk_time_since = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("STATE", self._an_trk_state), ("LOCKED", self._an_trk_locked),
            ("ESTIMATED", self._an_trk_est), ("VELOCITY", self._an_trk_vel),
            ("UNCERTAINTY", self._an_trk_unc), ("RESIDUAL", self._an_trk_residual),
            ("TIME SINCE DET", self._an_trk_time_since),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            trk_layout.addWidget(name_lbl, i, 0)
            trk_layout.addWidget(val, i, 1)
        right.addWidget(trk_group)

        # Plots
        right.addWidget(self._plots)

        # Event log
        event_group = self._make_group("EVENT LOG")
        event_layout = QVBoxLayout(event_group)
        event_layout.setContentsMargins(4, 4, 4, 4)
        event_layout.addWidget(self._event_log)
        right.addWidget(event_group, 1)

        layout.addLayout(right, 1)

        return page

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            f"QGroupBox {{ color: {Colors.ACCENT}; font-size: 9px; font-weight: bold; "
            f"border: 1px solid #20303d; border-radius: 3px; margin-top: 8px; padding-top: 12px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}"
        )
        return g

    # ------------------------------------------------------------------
    #  Right sidebar (context-sensitive)
    # ------------------------------------------------------------------
    def _build_right_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("RightSidebar")
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet(_SIDEBAR_STYLE)

        self._sidebar_stack = QStackedWidget()

        # Page 0 sidebar: Mission config (ControlPanel)
        self._sidebar_mission = self._build_sidebar_mission()
        self._sidebar_stack.addWidget(self._sidebar_mission)

        # Page 1 sidebar: Camera Tracking telemetry
        self._sidebar_camera = self._build_sidebar_camera()
        self._sidebar_stack.addWidget(self._sidebar_camera)

        # Page 2 sidebar: Analysis detail
        self._sidebar_analysis = self._build_sidebar_analysis()
        self._sidebar_stack.addWidget(self._sidebar_analysis)

        # Page 3 sidebar: World info
        self._sidebar_world = self._build_sidebar_world()
        self._sidebar_stack.addWidget(self._sidebar_world)

        # Page 4 sidebar: Benchmark
        self._sidebar_benchmark = self._build_sidebar_benchmark()
        self._sidebar_stack.addWidget(self._sidebar_benchmark)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._sidebar_stack)

        return sidebar

    def _update_right_sidebar_content(self) -> None:
        self._sidebar_stack.setCurrentIndex(self._current_page)

    def _build_sidebar_mission(self) -> QWidget:
        """Sidebar for Mission Setup: compact config + source mode."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("MISSION CONFIG")
        title.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(title)

        # Source mode
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("MODE:"))
        self._mission_mode_combo = QComboBox()
        self._mission_mode_combo.addItems(["SIMULATION", "VIDEO", "LIVE"])
        # Mirror of the top-bar source selector: one handler owns mode
        # changes, so this combo forwards instead of diverging.
        self._mission_mode_combo.currentTextChanged.connect(
            self._on_sidebar_mode_changed)
        src_row.addWidget(self._mission_mode_combo)
        layout.addLayout(src_row)

        # Compact config tabs
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.addTab(self._setup_tab_trajectory(), "TRJ")
        tabs.addTab(self._setup_tab_beacon(), "BCN")
        tabs.addTab(self._setup_tab_camera(), "CAM")
        tabs.addTab(self._setup_tab_disturbance(), "DST")
        layout.addWidget(tabs, 1)

        # Start button
        btn = QPushButton("START MISSION")
        btn.setFixedHeight(32)
        btn.setStyleSheet(
            f"QPushButton {{ color: #000; font-size: 11px; font-weight: bold; "
            f"background: {Colors.SUCCESS}; border: none; border-radius: 4px; }} "
            f"QPushButton:hover {{ background: #4aff9e; }}"
        )
        btn.clicked.connect(self._on_start)
        layout.addWidget(btn)

        return w

    def _build_sidebar_camera(self) -> QWidget:
        """Sidebar for Camera Tracking: telemetry + scorecard + AI summary.

        The content is taller than the viewport, so it lives in a scroll
        area — without one the layout squeezes the telemetry groups into
        illegibility. Visuals/dimensions are unchanged; only scrolling is
        added, scoped to this sidebar alone.
        """
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Compact tracking status
        status_group = self._make_group("TRACKING STATUS")
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(2)
        self._cam_state = QLabel("--")
        self._cam_conf = QLabel("--")
        self._cam_error = QLabel("--")
        self._cam_fps = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("STATE", self._cam_state), ("CONF", self._cam_conf),
            ("ERROR", self._cam_error), ("FPS", self._cam_fps),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 8px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            status_layout.addWidget(name_lbl, i, 0)
            status_layout.addWidget(val, i, 1)
        layout.addWidget(status_group)

        # Compact control status (live commands from runtime state)
        ctrl_group = self._make_group("CONTROL")
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setSpacing(2)
        self._cam_cmd_pan = QLabel("--")
        self._cam_cmd_tilt = QLabel("--")
        self._cam_ctrl_mode = QLabel("--")
        for i, (lbl, val) in enumerate([
            ("CMD PAN", self._cam_cmd_pan), ("CMD TILT", self._cam_cmd_tilt),
            ("MODE", self._cam_ctrl_mode),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 8px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            ctrl_layout.addWidget(name_lbl, i, 0)
            ctrl_layout.addWidget(val, i, 1)
        layout.addWidget(ctrl_group)

        # Telemetry (full)
        layout.addWidget(self._telemetry)

        # Scorecard
        layout.addWidget(self._scorecard)

        # AI summary
        layout.addWidget(self._ai_hud)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setWidget(content)
        return scroll

    def _build_sidebar_analysis(self) -> QWidget:
        """Sidebar for Analysis: link + comm + event log."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        tabs = QTabWidget()
        tabs.addTab(self._link_hud, "LINK")
        tabs.addTab(self._comm_hud, "COMM")
        tabs.addTab(self._event_log, "LOG")
        layout.addWidget(tabs, 1)

        return w

    def _build_sidebar_world(self) -> QWidget:
        """Sidebar for World: world info + object inspector."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        info_group = self._make_group("WORLD INFO")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(3)
        self._world_mode = QLabel("FREE")
        self._world_targets = QLabel("0")
        self._world_beacons = QLabel("0")
        for i, (lbl, val) in enumerate([
            ("CAMERA", self._world_mode), ("TARGETS", self._world_targets),
            ("BEACONS", self._world_beacons),
        ]):
            name_lbl = QLabel(lbl)
            name_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            val.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            info_layout.addWidget(name_lbl, i, 0)
            info_layout.addWidget(val, i, 1)
        layout.addWidget(info_group)

        layout.addWidget(self._ai_panel)
        layout.addWidget(self._link_hud)

        layout.addStretch()
        return w

    def _build_sidebar_benchmark(self) -> QWidget:
        """Sidebar for Benchmark: export + scorecard."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(self._scorecard)

        btn_export = QPushButton("EXPORT RESULTS")
        btn_export.setFixedHeight(28)
        btn_export.setStyleSheet(
            f"QPushButton {{ color: {Colors.ACCENT}; font-size: 9px; font-weight: bold; "
            f"background: transparent; border: 1px solid {Colors.ACCENT}; border-radius: 3px; }} "
            f"QPushButton:hover {{ background: rgba(34,211,238,0.15); }}"
        )
        btn_export.clicked.connect(self._export_performance_log)
        layout.addWidget(btn_export)

        layout.addWidget(self._event_log)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    #  Menu bar
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = self.menuBar()
        if menubar is None:
            return

        file_menu = menubar.addMenu("&File")
        if file_menu is None:
            return

        save_act = QAction("&Save Config", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_config)
        file_menu.addAction(save_act)

        load_act = QAction("&Load Config", self)
        load_act.setShortcut(QKeySequence("Ctrl+O"))
        load_act.triggered.connect(self._load_config)
        file_menu.addAction(load_act)

        export_act = QAction("Export &Performance Log", self)
        export_act.triggered.connect(self._export_performance_log)
        file_menu.addAction(export_act)

        file_menu.addSeparator()

        open_video_act = QAction("Open &Video File...", self)
        open_video_act.setShortcut(QKeySequence("Ctrl+V"))
        open_video_act.triggered.connect(self._on_open_video)
        file_menu.addAction(open_video_act)

        live_cam_act = QAction("Live &Camera", self)
        live_cam_act.setShortcut(QKeySequence("Ctrl+L"))
        live_cam_act.triggered.connect(self._on_live_camera)
        file_menu.addAction(live_cam_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        if sim_menu is None:
            return
        start_act = QAction("&Start", self)
        start_act.setShortcut(QKeySequence("F5"))
        start_act.triggered.connect(self._on_start)
        sim_menu.addAction(start_act)

        pause_act = QAction("&Pause", self)
        pause_act.setShortcut(QKeySequence("F6"))
        pause_act.triggered.connect(self._on_pause)
        sim_menu.addAction(pause_act)

        step_act = QAction("S&tep", self)
        step_act.setShortcut(QKeySequence("F7"))
        step_act.triggered.connect(self._on_step)
        sim_menu.addAction(step_act)

        stop_act = QAction("S&top", self)
        stop_act.setShortcut(QKeySequence("F8"))
        stop_act.triggered.connect(self._on_stop)
        sim_menu.addAction(stop_act)

        # View menu
        view_menu = menubar.addMenu("&View")
        if view_menu is None:
            return
        self._act_overlay = QAction("Toggle Debug &Overlay", self)
        self._act_overlay.setCheckable(True)
        self._act_overlay.setChecked(True)
        self._act_overlay.triggered.connect(self._toggle_overlay)
        view_menu.addAction(self._act_overlay)

        view_menu.addSeparator()

        # Layout presets: genuinely different information hierarchies,
        # applied live without restarting the simulation.
        self._layout_actions: dict[str, QAction] = {}
        for i, (lname, _spec) in enumerate(LAYOUTS.items()):
            _central, _sidebar, desc = _spec
            act = QAction(lname, self)
            act.setCheckable(True)
            act.setStatusTip(f"Layout: {desc}")
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda checked, n=lname: self.set_layout(n))
            view_menu.addAction(act)
            self._layout_actions[lname] = act
        self._update_layout_actions()

        view_menu.addSeparator()

        # Floating panels: dockable, collapsible, closable, resizable.
        # Hidden by default; toggled here without disturbing the layout.
        self._floating_actions: dict[str, QAction] = {}
        for fname in ("AI", "LINK", "TELEMETRY", "EVENTS"):
            fact = QAction(f"Floating {fname}", self)
            fact.setCheckable(True)
            fact.setChecked(False)
            fact.triggered.connect(
                lambda checked, n=fname: self.toggle_floating_panel(n))
            view_menu.addAction(fact)
            self._floating_actions[fname] = fact

        # Navigate menu
        nav_menu = menubar.addMenu("&Navigate")
        if nav_menu is None:
            return
        for i, title in enumerate(PAGE_TITLES):
            act = QAction(title, self)
            act.triggered.connect(lambda checked, idx=i: self._navigate_to(idx))
            nav_menu.addAction(act)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        if help_menu is None:
            return
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    #  Status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("IDLE")
        self._status.addWidget(self._status_label)
        self._mode_label = QLabel("SIMULATION")
        self._status.addPermanentWidget(self._mode_label)
        self._fps_label = QLabel("0 FPS")
        self._status.addPermanentWidget(self._fps_label)

    # ------------------------------------------------------------------
    #  Keyboard shortcuts
    # ------------------------------------------------------------------
    def _install_shortcuts(self) -> None:
        try:
            from PySide6.QtGui import QShortcut
        except ImportError:
            from PyQt5.QtWidgets import QShortcut  # type: ignore

        shortcuts = [
            ("Space", self._on_pause),
            ("S", self._on_step),
            ("R", self._on_stop),
            ("D", self._toggle_disturbances),
            ("G", self._toggle_ground_truth),
            ("T", self._toggle_overlay),
            ("Escape", self._on_stop),
            ("1", lambda: self._navigate_to(PAGE_MISSION_SETUP)),
            ("2", lambda: self._navigate_to(PAGE_CAMERA_TRACKING)),
            ("3", lambda: self._navigate_to(PAGE_ANALYSIS)),
            ("4", lambda: self._navigate_to(PAGE_WORLD)),
            ("5", lambda: self._navigate_to(PAGE_BENCHMARK)),
        ]
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            shortcut.setContext(Qt.ApplicationShortcut)

    def _toggle_disturbances(self) -> None:
        if hasattr(self, "_dist_enabled"):
            self._dist_enabled.toggle()

    def _toggle_ground_truth(self) -> None:
        if hasattr(self, "_control_panel") and hasattr(self._control_panel, "_show_gt"):
            self._control_panel._show_gt.toggle()

    # ------------------------------------------------------------------
    #  Actions — beacon / target
    # ------------------------------------------------------------------
    def _on_target_selected(self, target_id: int) -> None:
        s = self._state
        s.selected_object_id = target_id
        s.selected_terminal_a = False
        s.designated_beacon_id = target_id
        s.tracked_target_id = target_id
        s.terminal_b.active = True
        s.terminal_b.terminal_id = f"BEACON_{target_id}"
        for t in s.targets_all:
            if t.target_id == target_id:
                s.terminal_b.world_x = t.world_x
                s.terminal_b.world_y = t.world_y
                s.terminal_b.world_z = t.world_z
                break
        self._show_beacon_editor(target_id)
        self._update_beacon_list()

    def _on_terminal_selected(self) -> None:
        s = self._state
        s.selected_object_id = None
        s.selected_terminal_a = True
        term = s.terminal_a
        if hasattr(self, "_term_x"):
            self._term_x.setValue(term.world_x)
            self._term_y.setValue(term.world_y)
            self._term_z.setValue(term.world_z)
            self._term_yaw.setValue(term.yaw_deg)
            self._term_pitch.setValue(term.pitch_deg)
            self._terminal_editor.setVisible(True)
            self._beacon_editor.setVisible(False)
            self._selected_label.setText("Selected: TERMINAL A")
        s.add_event("Terminal A selected — edit pose below", "INFO")
        self._update_beacon_list()

    def _show_beacon_editor(self, target_id: int) -> None:
        if not hasattr(self, "_edit_traj"):
            return
        self._terminal_editor.setVisible(False)
        self._beacon_editor.setVisible(True)
        self._selected_label.setText(f"Selected: BEACON {target_id}")
        self._populate_beacon_editor(target_id)

    def _populate_beacon_editor(self, target_id: int) -> None:
        """Fill editor widgets from staged config or live engine state."""
        traj, size, bright, seed, shape, x, y, z = (
            "straight_line", 10.0, 1.0, 42, "spot", 1000.0, 1000.0, 500.0)
        eng = (self._worker._sim_engine
               if self._worker is not None else None)
        if eng is not None:
            for t in eng.get_state().targets:
                if t.target_id == target_id:
                    traj = t.trajectory_type or traj
                    size = t.size_px if t.size_px else size
                    bright = t.brightness
                    seed = t.seed
                    shape = t.shape or "square"
                    x, y, z = t.x, t.y, t.z
                    break
        else:
            world = getattr(self, "_world_config", None)
            if world is not None:
                for b in world.beacons:
                    if b.beacon_id == target_id:
                        traj = b.trajectory
                        size, bright, seed = b.size_px, b.brightness, b.seed
                        shape = getattr(b, "shape", "square")
                        x, y, z = b.x0, b.y0, b.z0
                        break
        self._edit_traj.setCurrentText(traj)
        self._edit_size.setValue(float(size))
        self._edit_bright.setValue(float(bright))
        self._edit_seed.setValue(int(seed))
        shape_label = {"square": "Square", "circular": "Circular"}.get(
            str(shape).lower(), "Spot (soft)")
        self._edit_shape.setCurrentText(shape_label)
        self._edit_x.setValue(float(x))
        self._edit_y.setValue(float(y))
        self._edit_z.setValue(float(z))

    def _on_apply_beacon_edit(self) -> None:
        bid = self._state.selected_object_id
        if bid is None:
            return
        traj = self._edit_traj.currentText()
        size = float(self._edit_size.value())
        bright = float(self._edit_bright.value())
        seed = int(self._edit_seed.value())
        shape = _shape_label_to_value(self._edit_shape.currentText())
        x, y, z = (float(self._edit_x.value()), float(self._edit_y.value()),
                   float(self._edit_z.value()))
        params = self._default_traj_params(traj, x, y, z, seed)
        eng = (self._worker._sim_engine
               if self._worker is not None else None)
        if eng is not None:
            ok = eng.reconfigure_beacon(
                bid, trajectory_type=traj, trajectory_params=params,
                size_px=size, brightness=bright, shape=shape, seed=seed)
            self._state.add_event(
                f"Beacon {bid} updated ({traj})" if ok
                else f"Beacon {bid} update failed", "INFO" if ok else "ERROR")
        else:
            world = getattr(self, "_world_config", None)
            if world is None:
                return
            for b in world.beacons:
                if b.beacon_id == bid:
                    b.trajectory = traj
                    b.trajectory_params = dict(params)
                    b.size_px = size
                    b.brightness = bright
                    b.shape = shape
                    b.seed = seed
                    b.x0, b.y0, b.z0 = x, y, z
                    self._state.add_event(
                        f"Beacon {bid} staged update ({traj})", "INFO")
                    break
        self._update_beacon_list()

    def _on_apply_terminal_edit(self) -> None:
        x, y, z = (float(self._term_x.value()), float(self._term_y.value()),
                   float(self._term_z.value()))
        yaw, pitch = float(self._term_yaw.value()), float(self._term_pitch.value())
        if self._worker is not None:
            ok = self._worker.set_terminal_pose_full(x, y, z, yaw, pitch)
            self._state.add_event(
                f"Terminal A moved to ({x:.0f}, {y:.0f}, {z:.0f})" if ok
                else "Terminal move failed", "INFO" if ok else "ERROR")
        world = getattr(self, "_world_config", None)
        if world is not None:
            world.terminal.x = x
            world.terminal.y = y
            world.terminal.z = z
            world.terminal.yaw_deg = yaw
            world.terminal.pitch_deg = pitch
            self._state.add_event("Terminal A staged pose updated", "INFO")
        if self._worker is None:
            self._state.terminal_a.world_x = x
            self._state.terminal_a.world_y = y
            self._state.terminal_a.world_z = z
            self._state.terminal_a.yaw_deg = yaw
            self._state.terminal_a.pitch_deg = pitch
        self._update_beacon_list()

    @staticmethod
    def _default_traj_params(traj_type: str, x: float, y: float, z: float,
                             seed: int) -> dict:
        """Sane per-type motion defaults centered on (x, y, z)."""
        if traj_type == "straight_line":
            return {"x0": x, "y0": y, "z0": z, "vx": 0.3, "vy": 0.2}
        if traj_type == "circular":
            return {"cx": x, "cy": y, "cz": z, "radius": 5.0,
                    "angular_speed_rad_s": 0.3}
        if traj_type == "figure_8":
            return {"cx": x, "cy": y, "cz": z, "amplitude_x": 8.0,
                    "amplitude_y": 5.0}
        if traj_type in ("random", "random_walk"):
            return {"x0": x, "y0": y, "z0": z, "seed": seed}
        if traj_type == "sinusoidal":
            return {"cx": x, "cy": y, "cz": z, "amplitude_x": 8.0,
                    "amplitude_y": 5.0, "freq_x": 0.3, "freq_y": 0.2}
        if traj_type == "spiral":
            return {"cx": x, "cy": y, "cz": z, "radius_start": 2.0,
                    "radius_growth": 0.5, "angular_speed_rad_s": 0.3}
        if traj_type == "user_controlled":
            return {"x0": x, "y0": y, "z0": z}
        return {"x0": x, "y0": y, "z0": z}

    def _on_add_beacon(self) -> None:
        traj_type = (self._traj_combo.currentText()
                     if hasattr(self, "_traj_combo") else "straight_line")
        seed = self._seed_spin.value() if hasattr(self, "_seed_spin") else 42
        # Spawn at the selected object's position, else at a seeded-random
        # location near the centre (PS default: random initial placement;
        # reproducible from seed + beacon count).
        x, y, z = 1000.0, 1000.0, 500.0
        sel = self._state.selected_object_id
        if sel is not None:
            for t in self._state.targets_all:
                if t.target_id == sel:
                    x, y, z = t.world_x, t.world_y, t.world_z
                    break
        else:
            import random
            eng = (self._worker._sim_engine
                   if self._worker is not None else None)
            if eng is not None:
                n = len(eng.get_state().targets)
            else:
                wc0 = getattr(self, "_world_config", None)
                n = len(wc0.beacons) if wc0 is not None else 0
            rng = random.Random(f"{seed}:{n}")
            x = 1000.0 + rng.uniform(-15.0, 15.0)
            y = 1000.0 + rng.uniform(-15.0, 15.0)
        params = self._default_traj_params(traj_type, x, y, z, seed)
        shape = (_shape_label_to_value(self._target_shape.currentText())
                 if hasattr(self, "_target_shape") else "spot")
        if self._worker is not None and self._worker._sim_engine is not None:
            new_id = self._worker.inject_beacon(
                x, y, z, traj_type, params, seed, shape=shape)
            self._state.add_event(f"Beacon {new_id} added ({traj_type})", "INFO")
        elif getattr(self, "_world_config", None) is not None:
            from fsoc_tracker.simulation.world_builder import add_beacon as wb_add
            b = wb_add(self._world_config, x, y, z, motion=traj_type,
                       motion_params=params, seed=seed, shape=shape)
            self._state.add_event(
                f"Beacon {b.beacon_id} staged in world ({traj_type})", "INFO")
        else:
            self._state.add_event("CREATE WORLD or START first", "WARNING")
        self._update_beacon_list()

    def _on_remove_beacon(self) -> None:
        if self._state.selected_object_id is None:
            return
        bid = self._state.selected_object_id
        removed = False
        if self._worker is not None and self._worker._sim_engine is not None:
            removed = self._worker._sim_engine.remove_beacon(bid)
        if not removed and getattr(self, "_world_config", None) is not None:
            from fsoc_tracker.simulation.world_builder import remove_beacon as wb_remove
            removed = wb_remove(self._world_config, bid)
        if removed:
            if self._state.designated_beacon_id == bid:
                self._state.designated_beacon_id = None
                self._state.tracked_target_id = None
            self._state.add_event(f"Beacon {bid} removed", "INFO")
        else:
            self._state.add_event(f"Beacon {bid} not found", "WARNING")
        self._state.selected_object_id = None
        self._state.selected_terminal_a = False
        if hasattr(self, "_beacon_editor"):
            self._beacon_editor.setVisible(False)
            self._terminal_editor.setVisible(False)
            self._selected_label.setText(
                "Selected: none (click a beacon or Terminal A in 3D)")
        self._update_beacon_list()

    def _on_set_beacon(self) -> None:
        if self._state.selected_object_id is not None:
            self._state.designated_beacon_id = self._state.selected_object_id
            self._state.tracked_target_id = self._state.selected_object_id
            self._state.terminal_b.active = True
            self._state.terminal_b.terminal_id = f"BEACON_{self._state.selected_object_id}"
            for t in self._state.targets_all:
                if t.target_id == self._state.selected_object_id:
                    self._state.terminal_b.world_x = t.world_x
                    self._state.terminal_b.world_y = t.world_y
                    self._state.terminal_b.world_z = t.world_z
                    break
            self._update_beacon_list()
        else:
            # Designate without spawning: never mutate the world here.
            exists = [t.target_id for t in self._state.targets_all]
            if not exists and getattr(self, "_world_config", None) is not None:
                exists = [b.beacon_id for b in self._world_config.beacons]
            if 0 in exists:
                self._state.designated_beacon_id = 0
                self._state.tracked_target_id = 0
                self._state.terminal_b.active = True
                self._state.terminal_b.terminal_id = "BEACON_0"
                for t in self._state.targets_all:
                    if t.target_id == 0:
                        self._state.terminal_b.world_x = t.world_x
                        self._state.terminal_b.world_y = t.world_y
                        self._state.terminal_b.world_z = t.world_z
                        break
                self._update_beacon_list()
            else:
                self._state.add_event("No beacon to designate", "WARNING")

    def _update_beacon_list(self) -> None:
        # Keep the 3D views in sync with staged content first.
        self._refresh_staged_targets()
        if hasattr(self, "_beacon_list"):
            lines = []
            for t in self._state.targets_all:
                marker = " ★" if t.target_id == self._state.designated_beacon_id else ""
                lines.append(f"OBJ-{t.target_id}: ({t.world_x:.0f}, {t.world_y:.0f}, {t.world_z:.0f}){marker}")
            if not lines:
                world = getattr(self, "_world_config", None)
                if world is not None:
                    for b in world.beacons:
                        marker = " ★" if b.beacon_id == world.primary_beacon_id else ""
                        lines.append(f"OBJ-{b.beacon_id}: ({b.x0:.0f}, {b.y0:.0f}, {b.z0:.0f}) {b.trajectory}{marker}")
            self._beacon_list.setPlainText("\n".join(lines) if lines else "No beacons")

    def _on_beacon_dragged(self, beacon_id: int, x: float, y: float, z: float) -> None:
        # Dragging moves only user-controlled beacons (their trajectory
        # accepts absolute positions). Scripted trajectories own their
        # beacons' motion — dragging those would fight the trajectory.
        traj_type = ""
        cur = None
        for t in self._state.targets_all:
            if t.target_id == beacon_id:
                traj_type = t.trajectory_type
                cur = (t.world_x, t.world_y, t.world_z)
                break
        if (traj_type == "user_controlled" and cur is not None
                and self._worker is not None):
            if self._worker.nudge_beacon(x - cur[0], y - cur[1], z - cur[2],
                                         beacon_id):
                return
        self._state.terminal_b.world_x = x
        self._state.terminal_b.world_y = y
        self._state.terminal_b.world_z = z
        if traj_type and traj_type != "user_controlled":
            self._state.add_event(
                f"Beacon {beacon_id} follows its {traj_type} trajectory; "
                "drag applies to user-controlled beacons", "INFO")

    def _on_terminal_placed(self, x: float, y: float, z: float) -> None:
        """Forward 3D world Terminal A placement to the worker camera."""
        if self._worker is not None:
            self._worker.set_terminal_pose(x, y, z)

    def _on_start_scan(self) -> None:
        if self._worker is None:
            return
        self._worker.start_beacon_scan()
        self._state.scan_active = True

    def _on_connect_beacon(self) -> None:
        if self._worker is None:
            return
        beacon_id = self._state.designated_beacon_id
        if beacon_id is None:
            beacon_id = self._state.selected_object_id
        if beacon_id is None:
            self._state.add_event("Select or designate a beacon first", "WARNING")
            return
        self._state.beacon_connected = True
        self._state.terminal_b.active = True
        self._state.terminal_b.terminal_id = f"BEACON_{beacon_id}"

    def _on_send_message(self) -> None:
        if self._worker is None:
            return
        self._worker.send_message_to_beacon("FSOC_LINK_TEST: Hello from Terminal A")
        self._state.add_event("Message sent to beacon", "COMM")

    # ------------------------------------------------------------------
    #  Actions — source / mode
    # ------------------------------------------------------------------
    def _on_sidebar_mode_changed(self, text: str) -> None:
        """Sidebar mirror of the top-bar source selector (single handler)."""
        self._src_combo.blockSignals(True)
        self._src_combo.setCurrentText(text)
        self._src_combo.blockSignals(False)
        self._on_source_changed(text)

    def _on_source_changed(self, text: str) -> None:
        mode_map = {"SIMULATION": "simulation", "VIDEO": "video", "LIVE": "live"}
        mode = mode_map.get(text, "simulation")
        if hasattr(self, "_mission_mode_combo"):
            self._mission_mode_combo.blockSignals(True)
            self._mission_mode_combo.setCurrentText(text)
            self._mission_mode_combo.blockSignals(False)
        if mode == "video":
            # Opening a video IS the action: prompt immediately so the
            # mode switch visibly does something.
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Video File", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
            )
            if not path:
                # Cancelled: revert the combo instead of stranding the UI
                # in a mode with no file.
                self._src_combo.blockSignals(True)
                self._src_combo.setCurrentText("SIMULATION")
                self._src_combo.blockSignals(False)
                return
            self._open_video_file(path)
            return
        if mode == "live":
            self._on_live_camera()
            return
        self._state.system_mode = SystemMode(mode)

    def _open_video_file(self, path: str) -> bool:
        """Start video tracking on path. Returns True on success."""
        # Probe synchronously: OpenCV defers open failures to first
        # read, so without this check a bad file would start a worker
        # that silently stops itself on frame zero.
        from fsoc_tracker.pipeline.sources import VideoSource
        try:
            probe = VideoSource(path)
            probe.open()
            frame = probe.read()
            probe.release()
        except Exception as e:
            self._state.add_event(f"VIDEO failed: {e}", "ERROR")
            return False
        if frame is None:
            self._state.add_event(
                f"VIDEO failed: no decodable frames in {path}", "ERROR")
            return False
        if self._worker is not None:
            self._worker.stop_run()
        try:
            self._start_worker({"mode": "video", "video_path": path})
        except Exception as e:
            self._state.add_event(f"VIDEO failed: {e}", "ERROR")
            return False
        self._last_video_path = path
        self._src_combo.blockSignals(True)
        self._src_combo.setCurrentText("VIDEO")
        self._src_combo.blockSignals(False)
        self._state.source_info = f"VIDEO: {path.split('/')[-1]}"
        self._navigate_to(PAGE_CAMERA_TRACKING)
        return True

    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if not path:
            return
        self._open_video_file(path)

    def _on_live_camera(self) -> None:
        if self._worker is not None:
            self._worker.stop_run()
        config = self._get_unified_config()
        camera_id = config.get("camera_id", 0)
        self._start_worker({"mode": "live", "camera_id": camera_id})
        self._state.source_info = f"LIVE CAMERA {camera_id}"

    # ------------------------------------------------------------------
    #  Actions — playback
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        cfg = self._get_unified_config()

        # CREATE WORLD gate: simulation requires an explicit world —
        # there are no preset scenes to fall back to.
        if cfg.get("mode", "simulation") == "simulation":
            world = getattr(self, "_world_config", None)
            if world is None:
                self._state.add_event(
                    "START blocked: CREATE WORLD first (top bar)", "ERROR")
                return
            cfg["scenario"] = world

        # VIDEO gate: a file is required — reuse the last one or ask.
        if cfg.get("mode") == "video" and not cfg.get("video_path"):
            if getattr(self, "_last_video_path", None):
                cfg["video_path"] = self._last_video_path
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Open Video File", "",
                    "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
                )
                if not path:
                    self._state.add_event(
                        "START blocked: choose a video file first", "ERROR")
                    return
                cfg["video_path"] = path
                self._last_video_path = path

        self._start_worker(cfg)
        self._navigate_to(PAGE_CAMERA_TRACKING)

    def _get_unified_config(self) -> dict:
        """Merge config from ControlPanel and mission setup widgets."""
        cfg = self._controller.config.copy()
        cfg.update(self._control_panel.get_config())

        # Override with mission setup values if available
        if hasattr(self, '_traj_combo'):
            cfg['trajectory'] = self._traj_combo.currentText()
        if hasattr(self, '_target_size'):
            cfg['target_size'] = self._target_size.value()
        if hasattr(self, '_target_shape'):
            cfg['target_shape'] = _shape_label_to_value(
                self._target_shape.currentText())
        if hasattr(self, '_seed_spin'):
            cfg['seed'] = self._seed_spin.value()
        if hasattr(self, '_sim_speed'):
            cfg['sim_dt'] = 1.0 / (30.0 * self._sim_speed.value())
        if hasattr(self, '_cam_w'):
            cfg['camera_width'] = self._cam_w.value()
        if hasattr(self, '_cam_h'):
            cfg['camera_height'] = self._cam_h.value()
        if hasattr(self, '_hfov'):
            cfg['hfov'] = self._hfov.value()
        if hasattr(self, '_vfov'):
            cfg['vfov'] = self._vfov.value()
        if hasattr(self, '_max_pan'):
            cfg['max_pan_rate'] = self._max_pan.value()
        if hasattr(self, '_max_tilt'):
            cfg['max_tilt_rate'] = self._max_tilt.value()
        if hasattr(self, '_dist_enabled'):
            cfg['disturbance_enabled'] = self._dist_enabled.isChecked()
        if hasattr(self, '_dist_preset'):
            cfg['disturbance_preset'] = self._dist_preset.currentText()
        if hasattr(self, '_dist_checks'):
            cfg['disturbance_effects'] = {
                key: cb.isChecked() for key, cb in self._dist_checks.items()
            }
        if hasattr(self, '_dist_intensity'):
            cfg['disturbance_intensity'] = self._dist_intensity.value()
        cfg.update(self._disturbance_tuning())

        return cfg

    def _start_worker(self, cfg: dict) -> None:
        self._controller._config = cfg

        if self._worker is None:
            self._worker = ProcessingWorker(self)
            self._worker.state_updated.connect(self._on_state_update)
            self._worker.error.connect(self._on_error)
            self._worker.log.connect(self._on_log)
        self._worker.configure(cfg)
        self._worker.start_run()

        self._status_label.setText("RUNNING")
        self._status_label.setStyleSheet(f"color: {Colors.SUCCESS};")
        self._header_state.setText("●  RUNNING")
        self._header_state.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")
        self._nav_status.setText("RUNNING")
        self._nav_status.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 8px; font-weight: bold; background: transparent; border: none; padding: 4px;")

        if hasattr(self, '_comm_hud'):
            self._comm_hud.set_comm_engine(self._worker.comm_engine)

    def _on_pause(self) -> None:
        if self._worker and self._worker.isRunning():
            if self._worker._paused:
                self._worker.resume()
            else:
                self._worker.pause()

    def _on_step(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.step()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop_run()
        self._status_label.setText("IDLE")
        self._status_label.setStyleSheet(f"color: {Colors.MUTED};")
        self._header_state.setText("●  STANDBY")
        self._header_state.setStyleSheet(f"color: {Colors.MUTED}; font-weight: bold;")
        self._nav_status.setText("IDLE")
        self._nav_status.setStyleSheet(f"color: {Colors.MUTED}; font-size: 8px; font-weight: bold; background: transparent; border: none; padding: 4px;")

    def _on_config_changed(self, cfg: dict) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.update_disturbance(cfg)
            self._worker.update_perception(cfg)
            self._worker.apply_runtime_toggles(cfg)

    def _on_seek_video(self, frame_index: int) -> None:
        if self._worker:
            self._worker.seek_video(frame_index)

    def _on_state_update(self, state: ApplicationViewState) -> None:
        self._state = state

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"ERROR: {msg[:50]}")
        self._status_label.setStyleSheet(f"color: {Colors.ERROR};")

    def _on_log(self, level: str, msg: str) -> None:
        self._state.add_event(msg, level)

    # ------------------------------------------------------------------
    #  Refresh from worker (33ms timer)
    # ------------------------------------------------------------------
    def _refresh_from_worker(self) -> None:
        if self._worker is None:
            return
        s = self._state
        frame = self._worker.latest_frame

        # --- Page 0: Mission Setup world view ---
        if hasattr(self, '_setup_world_view'):
            self._setup_world_view.update_state(s)

        # --- Page 1: Camera Tracking ---
        self._camera_workspace.update_frame(frame, s)

        # --- Page 3: World ---
        self._world_view.update_state(s)

        # --- Page 4: Benchmark ---
        self._benchmark_panel.update_state(s)

        # --- Shared panels ---
        self._telemetry.update_state(s)
        self._scorecard.update_state(s.scorecard)
        self._ai_panel.update_state(s)
        self._event_log.update_state(s)
        self._ai_hud.update_state(s)
        self._link_hud.update_state(s)
        self._comm_hud.update_state(s)
        self._plots.update_state(s)

        # --- Camera Tracking sidebar compact status ---
        if hasattr(self, '_cam_state'):
            state_colors = {
                "TRACKING": Colors.SUCCESS, "ACQUIRING": Colors.WARNING,
                "SEARCHING": Colors.WARNING, "LOST": Colors.ERROR,
                "NO_TRACK": Colors.MUTED, "REACQUIRING": Colors.WARNING,
            }
            c = state_colors.get(s.tracking.state, Colors.MUTED)
            self._cam_state.setText(s.tracking.state)
            self._cam_state.setStyleSheet(f"color: {c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

            conf = s.perception.confidence
            conf_c = Colors.SUCCESS if conf > 0.7 else Colors.WARNING if conf > 0.3 else Colors.ERROR
            self._cam_conf.setText(f"{conf:.2f}")
            self._cam_conf.setStyleSheet(f"color: {conf_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

            import math
            err = math.sqrt(s.tracking.estimated_x**2 + s.tracking.estimated_y**2) if s.tracking.locked else 0.0
            err_c = Colors.SUCCESS if err < 10 else Colors.WARNING if err < 20 else Colors.ERROR
            self._cam_error.setText(f"{err:.1f}px")
            self._cam_error.setStyleSheet(f"color: {err_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

            fps_c = Colors.SUCCESS if s.camera.fps >= 20 else Colors.WARNING if s.camera.fps >= 10 else Colors.ERROR
            self._cam_fps.setText(f"{s.camera.fps:.1f}")
            self._cam_fps.setStyleSheet(f"color: {fps_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

        # --- Camera sidebar compact control status (runtime commands) ---
        if hasattr(self, '_cam_cmd_pan'):
            cmd_mag = abs(s.control.pan_command) + abs(s.control.tilt_command)
            cmd_c = Colors.ACCENT if cmd_mag > FLOW_CONTROL_EPS_DEG_S else Colors.MUTED
            self._cam_cmd_pan.setText(f"{s.control.pan_command:+.2f}\u00b0/s")
            self._cam_cmd_pan.setStyleSheet(f"color: {cmd_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            self._cam_cmd_tilt.setText(f"{s.control.tilt_command:+.2f}\u00b0/s")
            self._cam_cmd_tilt.setStyleSheet(f"color: {cmd_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            mode_txt = s.control.control_mode.upper()
            if s.control.pan_saturated or s.control.tilt_saturated:
                mode_txt += " (SAT)"
                mode_c = Colors.WARNING
            else:
                mode_c = Colors.TEXT
            self._cam_ctrl_mode.setText(mode_txt)
            self._cam_ctrl_mode.setStyleSheet(f"color: {mode_c}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

        # --- Analysis page updates ---
        if self._current_page == PAGE_ANALYSIS:
            self._update_analysis_page(s)

        # --- World sidebar updates ---
        if self._current_page == PAGE_WORLD:
            self._world_mode.setText(s.camera_mode)
            self._world_targets.setText(str(len(s.targets_all)))
            self._world_beacons.setText("1" if s.terminal_b.active else "0")

        # --- Video info ---
        if s.system_mode == SystemMode.VIDEO:
            self._control_panel.update_video_info(
                current_frame=s.video_current_frame,
                total_frames=s.video_total_frames,
                timestamp_s=s.video_timestamp_s,
                fps=s.video_fps,
                processing_fps=s.video_processing_fps,
                status=s.tracking.state,
            )

        # --- Live info ---
        if s.system_mode == SystemMode.LIVE:
            self._control_panel.update_live_info(
                resolution=s.live_resolution,
                fps=s.live_fps,
                status=s.tracking.state,
                distance_status=s.live_distance_status,
            )

        # --- Mission flow strip (runtime-driven lamps) ---
        self._update_flow_strip(s)

        # --- Floating panels (live copies, menu-toggled) ---
        self._update_floating_panels(s)

        # --- Header bar ---
        state_colors = {
            "IDLE": Colors.MUTED, "INITIALIZING": Colors.WARNING,
            "RUNNING": Colors.SUCCESS, "PAUSED": Colors.WARNING, "ERROR": Colors.ERROR,
        }
        self._status_label.setText(s.system_state.value)
        self._status_label.setStyleSheet(f"color: {state_colors.get(s.system_state.value, Colors.TEXT)};")
        self._mode_label.setText(s.system_mode.value.upper())
        self._fps_label.setText(f"{s.camera.fps:.1f} FPS")
        status = s.tracking.state.replace("_", " ")
        status_color = {
            "TRACKING": Colors.SUCCESS, "ACQUIRING": Colors.WARNING,
            "SEARCHING": Colors.WARNING, "LOST": Colors.ERROR,
        }.get(s.tracking.state, Colors.MUTED)
        self._header_state.setText(f"●  {status}")
        self._header_state.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        self._header_fps.setText(
            f"PROC {s.performance.pipeline_fps:.1f} FPS  |  "
            f"LAT {s.performance.processing_ms:.1f} ms"
        )

    def _update_analysis_page(self, s: ApplicationViewState) -> None:
        """Push state into Analysis page labels."""
        ai = s.ai_state

        state_colors = {
            "normal_tracking": Colors.SUCCESS, "fast_target_motion": Colors.WARNING,
            "high_noise": Colors.WARNING, "edge_of_fov": Colors.WARNING,
            "degrading_track": Colors.WARNING, "low_confidence": Colors.WARNING,
            "prediction_uncertain": Colors.WARNING, "target_lost": Colors.ERROR,
            "reacquisition": Colors.WARNING, "recovery": Colors.SUCCESS,
        }
        c = state_colors.get(ai.situation, Colors.TEXT)
        self._an_situation.setText(ai.situation.replace("_", " ").upper())
        self._an_situation.setStyleSheet(f"color: {c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")

        self._an_action.setText(ai.action.replace("_", " ").upper())
        conf_c = Colors.SUCCESS if ai.confidence > 0.7 else Colors.WARNING if ai.confidence > 0.3 else Colors.ERROR
        self._an_confidence.setText(f"{ai.confidence:.2f}")
        self._an_confidence.setStyleSheet(f"color: {conf_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")

        risk_c = Colors.ERROR if ai.failure_risk > 0.6 else Colors.WARNING if ai.failure_risk > 0.3 else Colors.SUCCESS
        self._an_risk.setText(f"{ai.failure_risk:.2f}")
        self._an_risk.setStyleSheet(f"color: {risk_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")

        fb_c = Colors.ERROR if ai.fallback_active else Colors.SUCCESS
        self._an_fallback.setText("ACTIVE" if ai.fallback_active else "NORMAL")
        self._an_fallback.setStyleSheet(f"color: {fb_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")

        self._an_explanation.setText(ai.explanation or "--")

        # Prediction
        self._an_pred_dx.setText(f"{ai.prediction_dx:+.2f}px")
        self._an_pred_dy.setText(f"{ai.prediction_dy:+.2f}px")
        unc = (ai.prediction_uncertainty_x**2 + ai.prediction_uncertainty_y**2)**0.5
        unc_c = Colors.SUCCESS if unc < 5 else Colors.WARNING if unc < 15 else Colors.ERROR
        self._an_pred_unc.setText(f"{unc:.1f}px")
        self._an_pred_unc.setStyleSheet(f"color: {unc_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        self._an_pred_horizon.setText(f"{ai.prediction_horizon_s:.3f}s")

        # Search
        search_c = Colors.WARNING if ai.search_active else Colors.MUTED
        self._an_search_active.setText("YES" if ai.search_active else "NO")
        self._an_search_active.setStyleSheet(f"color: {search_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        self._an_search_strategy.setText(ai.search_strategy.replace("_", " ").upper() if ai.search_strategy else "--")
        self._an_search_phase.setText(ai.search_phase.replace("_", " ").upper() if ai.search_phase else "--")
        self._an_search_center.setText(f"({ai.search_center_x:.0f}, {ai.search_center_y:.0f})")
        self._an_search_radius.setText(f"{ai.search_radius_px:.0f}px")
        self._an_search_time.setText(f"{ai.search_time_s:.2f}s / {ai.search_duration_s:.1f}s")

        # ROI
        self._an_roi_mode.setText(ai.roi_mode.replace("_", " ").upper())
        self._an_roi_radius.setText(f"{ai.roi_radius_px:.0f}px")

        # Tracking detail
        trk = s.tracking
        trk_c = state_colors.get(trk.state.replace(" ", "_").lower(), Colors.TEXT)
        self._an_trk_state.setText(trk.state)
        self._an_trk_state.setStyleSheet(f"color: {trk_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        lock_c = Colors.SUCCESS if trk.locked else Colors.MUTED
        self._an_trk_locked.setText("YES" if trk.locked else "NO")
        self._an_trk_locked.setStyleSheet(f"color: {lock_c}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        self._an_trk_est.setText(f"({trk.estimated_x:.1f}, {trk.estimated_y:.1f})")
        vel = (trk.velocity_x**2 + trk.velocity_y**2)**0.5
        self._an_trk_vel.setText(f"{vel:.1f}px/s")
        self._an_trk_unc.setText(f"({trk.uncertainty_x:.1f}, {trk.uncertainty_y:.1f})")
        self._an_trk_residual.setText(f"{trk.residual:.1f}px")
        self._an_trk_time_since.setText(f"{trk.time_since_detection_s:.3f}s")

    # ------------------------------------------------------------------
    #  Config I/O
    # ------------------------------------------------------------------
    def _export_performance_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Performance Log", "fsoc_tracker_log.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        state = self._state
        payload = {
            "elapsed_s": state.elapsed_s,
            "camera": vars(state.camera),
            "perception": vars(state.perception),
            "tracking": vars(state.tracking),
            "performance": vars(state.performance),
            "error_history": list(state.errors.errors_euclidean),
            "events": [vars(event) for event in state.events],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "fsoc_config.json", "JSON (*.json)")
        if path:
            self._controller._config.update(self._get_unified_config())
            self._controller.save_config(path)

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if path:
            try:
                self._controller.load_config(path)
                self._control_panel.set_mode(self._controller.config.get("mode", "simulation"))
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _toggle_overlay(self) -> None:
        self._state.show_debug_overlay = self._act_overlay.isChecked()

    def _show_about(self) -> None:
        dlg = AboutDialog(self)
        dlg.exec()

    def _on_world_action(self, text: str) -> None:
        """CREATE WORLD entry points (replaces the retired scene presets)."""
        if text == "New empty world":
            from fsoc_tracker.simulation.world_builder import create_world
            seed = self._seed_spin.value() if hasattr(self, "_seed_spin") else 42
            self._world_config = create_world(seed=seed, name="Custom World")
            self._state.add_event(
                f"World created (empty, seed {seed})", "SUCCESS")
        elif text == "Random world (seed)":
            from fsoc_tracker.simulation.world_builder import randomize_world
            seed = self._seed_spin.value() if hasattr(self, "_seed_spin") else 42
            self._world_config = randomize_world(master_seed=seed)
            n = len(self._world_config.beacons)
            self._state.add_event(
                f"World randomized (seed {seed}, {n} beacons)", "SUCCESS")
        # Reset the combo back to a neutral display is intentionally NOT
        # done: the visible entry records the last world action.
        self._refresh_staged_targets()
        self._update_beacon_list()

    def _refresh_staged_targets(self) -> None:
        """Mirror staged (pre-start) world content into the view state.

        targets_all is otherwise populated only by the running worker,
        so staged beacons and the terminal would be invisible (and
        unclickable) in the 3D views until START. Has no effect once a
        worker exists — live engine data wins.
        """
        if self._worker is not None:
            return
        world = getattr(self, "_world_config", None)
        if world is None:
            return
        from fsoc_tracker.gui.state import TargetView
        views = []
        for b in world.beacons:
            views.append(TargetView(
                target_id=b.beacon_id,
                visible=bool(b.active),
                world_x=float(b.x0), world_y=float(b.y0), world_z=float(b.z0),
                size_px=float(b.size_px),
                trajectory_type=str(b.trajectory),
            ))
        self._state.targets_all = views
        term = world.terminal
        self._state.terminal_a.active = True
        self._state.terminal_a.terminal_id = "TERM_A"
        self._state.terminal_a.world_x = float(term.x)
        self._state.terminal_a.world_y = float(term.y)
        self._state.terminal_a.world_z = float(term.z)
        self._state.terminal_a.yaw_deg = float(term.yaw_deg)
        self._state.terminal_a.pitch_deg = float(term.pitch_deg)
        for view in (getattr(self, "_setup_world_view", None),
                     getattr(self, "_world_view", None)):
            if view is not None:
                try:
                    view.update_state(self._state)
                except Exception:
                    pass

    def _on_pov_changed(self, mode: str) -> None:
        self._state.camera_mode = mode
        if mode == "TERMINAL A POV":
            self._state.terminal_a.active = True

    def keyPressEvent(self, event) -> None:
        # AI-blind challenge: route beacon keys to the user-controlled
        # beacon through the simulation engine. The command never reaches
        # perception/tracking/AI — only rendered pixels do.
        if self._try_beacon_key(event):
            self.update()
            return
        if self._is_beacon_move_key(event.key()):
            # Movement key with no controllable beacon: say so instead of
            # silently falling through (the common "keys do nothing" trap).
            if (self._worker is not None and self._worker.isRunning()
                    and self._state is not None):
                self._state.add_event(
                    "No user-controlled beacon: set Trajectory to "
                    "user_controlled and press + Add Beacon", "WARNING")
                self.update()
                return
        if self._state and self._state.camera_mode == "FOLLOW" and hasattr(self._state, "terminal_b"):
            speed = 5.0
            if event.key() == Qt.Key_W:
                self._state.terminal_b.world_z += speed
            elif event.key() == Qt.Key_S:
                self._state.terminal_b.world_z -= speed
            elif event.key() == Qt.Key_A:
                self._state.terminal_b.world_x -= speed
            elif event.key() == Qt.Key_D:
                self._state.terminal_b.world_x += speed
            elif event.key() == Qt.Key_Q:
                self._state.terminal_b.world_y += speed
            elif event.key() == Qt.Key_E:
                self._state.terminal_b.world_y -= speed
            self.update()
        else:
            super().keyPressEvent(event)

    @staticmethod
    def _is_beacon_move_key(key: int) -> bool:
        """True for keys that drive a user-controlled beacon."""
        return key in (
            Qt.Key_A, Qt.Key_D, Qt.Key_W, Qt.Key_S, Qt.Key_Q, Qt.Key_E,
            Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
            Qt.Key_Space, Qt.Key_R, Qt.Key_M,
        )

    def _try_beacon_key(self, event) -> bool:
        """Route WASD/QE/arrows/STOP/RESET/RANDOM to a user-controlled beacon.

        Returns True if the key was consumed. No-op when no
        user-controlled beacon exists in the current scene.
        """
        if self._worker is None or self._state is None:
            return False
        beacon_id = None
        for t in self._state.targets_all:
            if t.trajectory_type == "user_controlled":
                beacon_id = t.target_id
                break
        if beacon_id is None:
            return False
        step = 10.0
        key = event.key()
        moves = {
            Qt.Key_A: (-step, 0.0, 0.0), Qt.Key_D: (step, 0.0, 0.0),
            Qt.Key_W: (0.0, 0.0, step), Qt.Key_S: (0.0, 0.0, -step),
            Qt.Key_Q: (0.0, -step, 0.0), Qt.Key_E: (0.0, step, 0.0),
            Qt.Key_Left: (-step, 0.0, 0.0), Qt.Key_Right: (step, 0.0, 0.0),
            Qt.Key_Up: (0.0, 0.0, step), Qt.Key_Down: (0.0, 0.0, -step),
        }
        if key in moves:
            dx, dy, dz = moves[key]
            return bool(self._worker.nudge_beacon(dx, dy, dz, beacon_id))
        if key == Qt.Key_Space:
            self._state.add_event("Beacon HOLD (operator stop)", "INFO")
            return True
        if key == Qt.Key_R:
            return bool(self._worker.reset_beacon(beacon_id))
        if key == Qt.Key_M:
            return bool(self._worker.random_beacon_maneuver(beacon_id))
        return False

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop_run()
            self._worker.wait(2000)
        event.accept()
