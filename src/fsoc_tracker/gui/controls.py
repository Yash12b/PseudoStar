"""Control panel — tabbed configuration controls."""

from __future__ import annotations

from typing import Any

from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import Qt, Signal  # type: ignore
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QLabel,
        QLineEdit,  # type: ignore
        QPushButton,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )


def _shape_label_to_value(label: str) -> str:
    """Map the target-shape combo label to a renderer shape value.

    "Spot (soft)" is the legacy soft-Gaussian rendering; "Square" and
    "Circular" render hard-edged spots of the selected geometry.
    """
    low = (label or "").lower()
    if low.startswith("square"):
        return "square"
    if low.startswith("circular"):
        return "circular"
    return "spot"


class ControlPanel(QFrame):
    config_changed = Signal(dict)
    start_clicked = Signal()
    pause_clicked = Signal()
    step_clicked = Signal()
    stop_clicked = Signal()
    reset_clicked = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(220)
        self._seeking = False
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        btn_row = QGridLayout()
        btn_row.setSpacing(4)
        self._btn_start = QPushButton("START")
        self._btn_start.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.PANEL_RAISED}; "
            f"color: {Colors.SUCCESS}; font-weight: bold; }}"
        )
        self._btn_pause = QPushButton("PAUSE")
        self._btn_step = QPushButton("STEP")
        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.PANEL_RAISED}; "
            f"color: {Colors.ERROR}; font-weight: bold; }}"
        )
        self._btn_reset = QPushButton("RESET")
        for i, btn in enumerate([self._btn_start, self._btn_pause, self._btn_step, self._btn_stop, self._btn_reset]):
            btn_row.addWidget(btn, 0, i)
        self._btn_start.clicked.connect(self.start_clicked)
        self._btn_pause.clicked.connect(self.pause_clicked)
        self._btn_step.clicked.connect(self.step_clicked)
        self._btn_stop.clicked.connect(self.stop_clicked)
        self._btn_reset.clicked.connect(self.reset_clicked)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["SIMULATION", "VIDEO", "LIVE"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        btn_row.addWidget(QLabel("MODE:"), 1, 0)
        btn_row.addWidget(self._mode_combo, 1, 1, 1, 2)

        main_layout.addLayout(btn_row)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.addTab(self._sim_tab(), "SIM")
        tabs.addTab(self._camera_tab(), "CAM")
        tabs.addTab(self._perception_tab(), "PERC")
        tabs.addTab(self._disturbance_tab(), "DIST")
        tabs.addTab(self._video_tab(), "VIDEO")
        tabs.addTab(self._live_tab(), "LIVE")
        main_layout.addWidget(tabs)

    def _sim_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

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
        return w

    def _camera_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

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

        self._pred_horizon = QDoubleSpinBox()
        self._pred_horizon.setRange(0.025, 0.5)
        self._pred_horizon.setValue(0.1)
        self._pred_horizon.setSingleStep(0.025)
        self._pred_horizon.setSuffix(" s")
        layout.addWidget(QLabel("Pred Horizon:"), 6, 0)
        layout.addWidget(self._pred_horizon, 6, 1)
        return w

    def _perception_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self._perc_combo = QComboBox()
        self._perc_combo.addItems(["classical", "ai", "hybrid"])
        self._perc_combo.currentTextChanged.connect(
            lambda _t: self.config_changed.emit(self.get_config()))
        layout.addWidget(QLabel("Backend:"), 0, 0)
        layout.addWidget(self._perc_combo, 0, 1)

        self._conf_thresh = QDoubleSpinBox()
        self._conf_thresh.setRange(0.05, 0.95)
        self._conf_thresh.setValue(0.3)
        self._conf_thresh.setSingleStep(0.05)
        layout.addWidget(QLabel("Conf Thresh:"), 1, 0)
        layout.addWidget(self._conf_thresh, 1, 1)

        self._show_gt = QCheckBox("Show Ground Truth")
        layout.addWidget(self._show_gt, 2, 0, 1, 2)

        self._show_trail = QCheckBox("Show Trail")
        self._show_trail.setChecked(True)
        layout.addWidget(self._show_trail, 3, 0, 1, 2)

        # Live ablation toggles (manish-review port): no restart needed.
        self._kalman_check = QCheckBox("Kalman Tracker")
        self._kalman_check.setChecked(True)
        self._kalman_check.setToolTip("Uncheck for raw-detection tracking")
        self._kalman_check.toggled.connect(
            lambda _c: self.config_changed.emit(self.get_config()))
        layout.addWidget(self._kalman_check, 4, 0, 1, 2)

        self._ai_brain_check = QCheckBox("AI Brain")
        self._ai_brain_check.setChecked(True)
        self._ai_brain_check.setToolTip("Uncheck to bypass mission AI")
        self._ai_brain_check.toggled.connect(
            lambda _c: self.config_changed.emit(self.get_config()))
        layout.addWidget(self._ai_brain_check, 5, 0, 1, 2)
        return w

    def _disturbance_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # --- Row 0: Enable + Preset ---
        self._dist_enabled = QCheckBox("Enable")
        layout.addWidget(self._dist_enabled, 0, 0, 1, 2)

        self._dist_preset = QComboBox()
        self._dist_preset.addItems(["clear", "light", "moderate", "severe", "extreme"])
        layout.addWidget(QLabel("Preset:"), 0, 2)
        layout.addWidget(self._dist_preset, 0, 3)

        # --- Row 1: Apply / Reset buttons ---
        self._dist_apply = QPushButton("Apply")
        self._dist_apply.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.PANEL_RAISED}; "
            f"color: {Colors.SUCCESS}; font-weight: bold; }}"
        )
        layout.addWidget(self._dist_apply, 1, 0, 1, 2)
        self._dist_reset = QPushButton("Reset")
        layout.addWidget(self._dist_reset, 1, 2, 1, 2)

        # --- Row 2-: Individual effect checkboxes ---
        row = 2
        self._noise_check = QCheckBox("Noise")
        layout.addWidget(self._noise_check, row, 0)
        self._fog_check = QCheckBox("Fog")
        layout.addWidget(self._fog_check, row, 1)
        self._haze_check = QCheckBox("Haze")
        layout.addWidget(self._haze_check, row, 2)
        self._rain_check = QCheckBox("Rain")
        layout.addWidget(self._rain_check, row, 3)

        row += 1
        self._lowlight_check = QCheckBox("Low Light")
        layout.addWidget(self._lowlight_check, row, 0)
        self._jitter_check = QCheckBox("Jitter")
        layout.addWidget(self._jitter_check, row, 1)
        self._blur_check = QCheckBox("Blur")
        layout.addWidget(self._blur_check, row, 2)
        self._motion_blur_check = QCheckBox("Motion Blur")
        layout.addWidget(self._motion_blur_check, row, 3)

        row += 1
        self._brightness_check = QCheckBox("Bri/Contrast")
        layout.addWidget(self._brightness_check, row, 0)
        self._platform_check = QCheckBox("Platform")
        layout.addWidget(self._platform_check, row, 1)
        self._disappearance_check = QCheckBox("Disappear")
        layout.addWidget(self._disappearance_check, row, 2)
        self._distractor_check = QCheckBox("Distractors")
        layout.addWidget(self._distractor_check, row, 3)

        row += 1
        self._turbulence_check = QCheckBox("Turbulence")
        layout.addWidget(self._turbulence_check, row, 0)
        self._gauss_check = QCheckBox("Gaussian")
        self._gauss_check.setChecked(True)
        self._gauss_check.setToolTip("Additive Gaussian noise")
        layout.addWidget(self._gauss_check, row, 1)
        self._sp_check = QCheckBox("Salt&Pepper")
        self._sp_check.setChecked(True)
        self._sp_check.setToolTip("Salt & pepper (~10% per PS)")
        layout.addWidget(self._sp_check, row, 2)
        self._poisson_check = QCheckBox("Poisson")
        self._poisson_check.setChecked(True)
        self._poisson_check.setToolTip("Photon-shot Poisson noise")
        layout.addWidget(self._poisson_check, row, 3)

        # --- Row: Intensity slider ---
        row += 1
        self._dist_intensity = QSlider(Qt.Orientation.Horizontal)
        self._dist_intensity.setRange(0, 100)
        self._dist_intensity.setValue(50)
        layout.addWidget(QLabel("Intensity:"), row, 0)
        layout.addWidget(self._dist_intensity, row, 1, 1, 3)

        # --- PS numeric tuning (user-defined, clamped to PS maxima) ---
        row += 1
        self._noise_sigma = QDoubleSpinBox()
        self._noise_sigma.setRange(0.0, 20.0)
        self._noise_sigma.setValue(5.0)
        self._noise_sigma.setSingleStep(0.5)
        self._noise_sigma.setSuffix(" σ")
        layout.addWidget(QLabel("Noise σ:"), row, 0)
        layout.addWidget(self._noise_sigma, row, 1)
        self._noise_density = QDoubleSpinBox()
        self._noise_density.setRange(0.0, 0.5)
        self._noise_density.setValue(0.05)
        self._noise_density.setSingleStep(0.01)
        layout.addWidget(QLabel("S&P dens:"), row, 2)
        layout.addWidget(self._noise_density, row, 3)

        row += 1
        self._jitter_amp = QDoubleSpinBox()
        self._jitter_amp.setRange(0.0, 20.0)
        self._jitter_amp.setValue(5.0)
        self._jitter_amp.setSingleStep(0.5)
        self._jitter_amp.setSuffix(" px")
        layout.addWidget(QLabel("Jitter amp:"), row, 0)
        layout.addWidget(self._jitter_amp, row, 1)
        self._platform_type = QComboBox()
        self._platform_type.addItems(
            ["linear", "circular", "figure_eight", "spiral", "random"])
        layout.addWidget(QLabel("Platform:"), row, 2)
        layout.addWidget(self._platform_type, row, 3)

        # Wire signals
        self._dist_apply.clicked.connect(self._on_disturbance_apply)
        self._dist_preset.currentTextChanged.connect(self._on_disturbance_preset_changed)

        return w

    def _on_disturbance_preset_changed(self, preset: str) -> None:
        """Auto-check/uncheck effect checkboxes when preset changes."""
        presets = {
            "clear":   {},
            "light":   {"noise": True, "fog": True, "haze": True},
            "moderate": {"noise": True, "fog": True, "blur": True, "rain": True,
                        "motion_blur": True, "distractors": True},
            "severe":  {"noise": True, "fog": True, "blur": True, "rain": True,
                        "motion_blur": True, "distractors": True, "target_disappearance": True,
                        "brightness_contrast": True, "low_light": True},
            "extreme": {"noise": True, "fog": True, "haze": True, "rain": True,
                        "blur": True, "motion_blur": True, "distractors": True,
                        "target_disappearance": True, "brightness_contrast": True,
                        "low_light": True, "jitter": True, "platform_motion": True,
                        "turbulence": True},
        }
        checks = presets.get(preset, {})
        self._noise_check.setChecked(checks.get("noise", False))
        self._fog_check.setChecked(checks.get("fog", False))
        self._haze_check.setChecked(checks.get("haze", False))
        self._rain_check.setChecked(checks.get("rain", False))
        self._lowlight_check.setChecked(checks.get("low_light", False))
        self._jitter_check.setChecked(checks.get("jitter", False))
        self._blur_check.setChecked(checks.get("blur", False))
        self._motion_blur_check.setChecked(checks.get("motion_blur", False))
        self._brightness_check.setChecked(checks.get("brightness_contrast", False))
        self._platform_check.setChecked(checks.get("platform_motion", False))
        self._disappearance_check.setChecked(checks.get("target_disappearance", False))
        self._distractor_check.setChecked(checks.get("distractors", False))
        self._turbulence_check.setChecked(checks.get("turbulence", False))

    def _on_disturbance_apply(self) -> None:
        """Emit config_changed with current disturbance settings."""
        self.config_changed.emit(self.get_config())

    def _video_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self._video_path = QLineEdit()
        self._video_path.setPlaceholderText("Select video file...")
        layout.addWidget(QLabel("File:"), 0, 0)
        layout.addWidget(self._video_path, 0, 1)

        self._btn_browse = QPushButton("Browse")
        self._btn_browse.clicked.connect(self._browse_video)
        layout.addWidget(self._btn_browse, 0, 2)

        # Video info label
        self._video_info = QLabel("No file selected")
        self._video_info.setStyleSheet("color: #8b949e; font-size: 9px;")
        layout.addWidget(self._video_info, 1, 0, 1, 3)

        # Frame info row
        self._video_frame_label = QLabel("Frame: --/--")
        self._video_frame_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._video_frame_label, 2, 0, 1, 2)

        self._video_time_label = QLabel("Time: 0.00s")
        self._video_time_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._video_time_label, 2, 2)

        # FPS info row
        self._video_fps_label = QLabel("FPS: --")
        self._video_fps_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._video_fps_label, 3, 0)

        self._video_proc_fps_label = QLabel("Proc: --")
        self._video_proc_fps_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._video_proc_fps_label, 3, 1)

        self._video_status_label = QLabel("IDLE")
        self._video_status_label.setStyleSheet("font-size: 9px; color: #8b949e;")
        layout.addWidget(self._video_status_label, 3, 2)

        # Seek slider
        self._video_seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._video_seek_slider.setRange(0, 0)
        self._video_seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._video_seek_slider.sliderReleased.connect(self._on_seek_released)
        layout.addWidget(QLabel("Seek:"), 4, 0)
        layout.addWidget(self._video_seek_slider, 4, 1, 1, 2)

        return w

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)",
        )
        if path:
            self._video_path.setText(path)
            self._video_info.setText(path.split("/")[-1])

    def _on_seek_pressed(self) -> None:
        """Flag that user is dragging seek slider."""
        self._seeking = True

    def _on_seek_released(self) -> None:
        """Emit seek request with target frame number."""
        self._seeking = False
        frame = self._video_seek_slider.value()
        self.seek_requested.emit(frame)

    def update_video_info(
        self, current_frame: int, total_frames: int, timestamp_s: float,
        fps: float, processing_fps: float, status: str,
    ) -> None:
        """Update video tab display with current playback info."""
        self._video_frame_label.setText(f"Frame: {current_frame}/{total_frames}")
        self._video_time_label.setText(f"Time: {timestamp_s:.2f}s")
        self._video_fps_label.setText(f"FPS: {fps:.1f}" if fps > 0 else "FPS: --")
        self._video_proc_fps_label.setText(
            f"Proc: {processing_fps:.1f}" if processing_fps > 0 else "Proc: --"
        )
        self._video_status_label.setText(status)
        color_map = {
            "IDLE": "#8b949e", "TRACKING": "#3fb950", "LOST": "#f85149",
            "SEARCHING": "#d29922", "PAUSED": "#8b949e",
        }
        c = color_map.get(status, "#8b949e")
        self._video_status_label.setStyleSheet(f"font-size: 9px; color: {c};")
        if total_frames > 0 and not self._seeking:
            self._video_seek_slider.setRange(0, total_frames - 1)
            self._video_seek_slider.setValue(current_frame)

    def _live_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Camera selector
        self._live_camera_combo = QComboBox()
        self._live_camera_combo.addItem("Select Camera...")
        layout.addWidget(QLabel("Camera:"), 0, 0)
        layout.addWidget(self._live_camera_combo, 0, 1, 1, 2)

        # Refresh cameras button
        self._btn_refresh_cameras = QPushButton("Refresh")
        self._btn_refresh_cameras.clicked.connect(self._refresh_live_cameras)
        layout.addWidget(self._btn_refresh_cameras, 0, 3)

        # Camera info
        self._live_info_label = QLabel("No camera selected")
        self._live_info_label.setStyleSheet("color: #8b949e; font-size: 9px;")
        layout.addWidget(self._live_info_label, 1, 0, 1, 4)

        # Resolution display
        self._live_resolution_label = QLabel("Resolution: --")
        self._live_resolution_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._live_resolution_label, 2, 0, 1, 2)

        # FPS display
        self._live_fps_label = QLabel("FPS: --")
        self._live_fps_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._live_fps_label, 2, 2)

        # Status
        self._live_status_label = QLabel("Status: IDLE")
        self._live_status_label.setStyleSheet("font-size: 9px; color: #8b949e;")
        layout.addWidget(self._live_status_label, 3, 0, 1, 2)

        # Distance status
        self._live_distance_label = QLabel("Distance: UNAVAILABLE")
        self._live_distance_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._live_distance_label, 3, 2)

        return w

    def _refresh_live_cameras(self) -> None:
        """Refresh the list of available cameras."""
        from fsoc_tracker.pipeline.sources import LiveSource
        cameras = LiveSource.enumerate_cameras(max_devices=10)
        self._live_camera_combo.clear()
        self._live_camera_combo.addItem("Select Camera...")
        for cam in cameras:
            status = "✓" if cam["is_available"] else "✗"
            self._live_camera_combo.addItem(f"{status} {cam['name']}", cam["camera_id"])
        self._live_info_label.setText(f"Found {len(cameras)} camera(s)")

    def update_live_info(
        self, resolution: str, fps: float, status: str, distance_status: str,
    ) -> None:
        """Update live tab display with current camera info."""
        self._live_resolution_label.setText(f"Resolution: {resolution}")
        self._live_fps_label.setText(f"FPS: {fps:.1f}" if fps > 0 else "FPS: --")
        self._live_status_label.setText(f"Status: {status}")
        self._live_distance_label.setText(f"Distance: {distance_status}")
        color_map = {
            "IDLE": "#8b949e", "TRACKING": "#3fb950", "LOST": "#f85149",
            "SEARCHING": "#d29922", "PAUSED": "#8b949e",
        }
        c = color_map.get(status, "#8b949e")
        self._live_status_label.setStyleSheet(f"font-size: 9px; color: {c};")

    def _on_mode_changed(self, mode_text: str) -> None:
        mode_text.lower()

    def get_config(self) -> dict[str, Any]:
        return {
            "mode": self._mode_combo.currentText().lower(),
            "trajectory": self._traj_combo.currentText(),
            "target_size": self._target_size.value(),
            "target_shape": _shape_label_to_value(
                self._target_shape.currentText()),
            "seed": self._seed_spin.value(),
            "sim_dt": 1.0 / (30.0 * self._sim_speed.value()),
            "camera_width": self._cam_w.value(),
            "camera_height": self._cam_h.value(),
            "hfov": self._hfov.value(),
            "vfov": self._vfov.value(),
            "max_pan_rate": self._max_pan.value(),
            "max_tilt_rate": self._max_tilt.value(),
            "prediction_horizon_s": self._pred_horizon.value(),
            "perception_backend": self._perc_combo.currentText(),
            "confidence_threshold": self._conf_thresh.value(),
            "kalman_enabled": self._kalman_check.isChecked(),
            "ai_brain_enabled": self._ai_brain_check.isChecked(),
            "disturbance_enabled": self._dist_enabled.isChecked(),
            "disturbance_preset": self._dist_preset.currentText(),
            "disturbance_effects": {
                "noise": self._noise_check.isChecked(),
                "noise_gaussian": self._gauss_check.isChecked(),
                "noise_salt_pepper": self._sp_check.isChecked(),
                "noise_poisson": self._poisson_check.isChecked(),
                "fog": self._fog_check.isChecked(),
                "haze": self._haze_check.isChecked(),
                "rain": self._rain_check.isChecked(),
                "low_light": self._lowlight_check.isChecked(),
                "jitter": self._jitter_check.isChecked(),
                "blur": self._blur_check.isChecked(),
                "motion_blur": self._motion_blur_check.isChecked(),
                "brightness_contrast": self._brightness_check.isChecked(),
                "platform_motion": self._platform_check.isChecked(),
                "target_disappearance": self._disappearance_check.isChecked(),
                "distractors": self._distractor_check.isChecked(),
                "turbulence": self._turbulence_check.isChecked(),
            },
            "disturbance_intensity": self._dist_intensity.value() / 100.0,
            "disturbance_noise_sigma": self._noise_sigma.value(),
            "disturbance_noise_density": self._noise_density.value(),
            "disturbance_jitter_amp": self._jitter_amp.value(),
            "disturbance_platform_type": self._platform_type.currentText(),
            "video_path": self._video_path.text(),
            "camera_id": self._live_camera_combo.currentData() or 0,
            "show_ground_truth": self._show_gt.isChecked(),
            "show_trail": self._show_trail.isChecked(),
        }

    def set_mode(self, mode: str) -> None:
        idx = self._mode_combo.findText(mode.upper())
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
