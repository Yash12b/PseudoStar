"""Tests for GUI module — Stage 11.

Tests state management, controller, worker pipeline, and widget state.
Does NOT test pixel rendering (no display server in CI).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fsoc_tracker.gui.controller import ApplicationController
from fsoc_tracker.gui.state import (
    ApplicationViewState,
    BenchmarkView,
    CameraViewState,
    ControlState,
    DisturbanceView,
    ErrorHistory,
    EventLogEntry,
    PerceptionState,
    PerformanceView,
    RunState,
    SIHScorecard,
    SystemMode,
    SystemState,
    TargetView,
    TrackingStateView,
)
from fsoc_tracker.gui.theme import Colors, Fonts, Spacing, apply_theme
from fsoc_tracker.gui.worker import ProcessingWorker

# --- State Tests ---

class TestCameraViewState:
    def test_default_values(self):
        s = CameraViewState()
        assert s.image_width == 640
        assert s.image_height == 480
        assert s.hfov_deg == 4.0
        assert s.vfov_deg == 3.0
        assert s.pan_deg == 0.0
        assert s.tilt_deg == 0.0
        assert s.fps == 0.0
        assert s.timestamp_s == 0.0

    def test_custom_values(self):
        s = CameraViewState(image_width=1920, image_height=1080, hfov_deg=10.0)
        assert s.image_width == 1920
        assert s.image_height == 1080
        assert s.hfov_deg == 10.0


class TestPerceptionState:
    def test_default_values(self):
        s = PerceptionState()
        assert s.backend == "classical"
        assert s.detected is False
        assert s.confidence == 0.0

    def test_detection_state(self):
        s = PerceptionState(detected=True, confidence=0.85, detection_x=320.0, detection_y=240.0)
        assert s.detected is True
        assert s.confidence == 0.85
        assert s.detection_x == 320.0
        assert s.detection_y == 240.0


class TestTrackingStateView:
    def test_default_values(self):
        s = TrackingStateView()
        assert s.state == "NO_TRACK"
        assert s.locked is False

    def test_tracking_active(self):
        s = TrackingStateView(state="TRACKING", locked=True, estimated_x=320.0, estimated_y=240.0)
        assert s.locked is True
        assert s.state == "TRACKING"


class TestTargetView:
    def test_default_values(self):
        t = TargetView()
        assert t.visible is False
        assert t.trail == []

    def test_trail(self):
        t = TargetView()
        t.trail = [(100, 100), (200, 200)]
        assert len(t.trail) == 2


class TestErrorHistory:
    def test_default_values(self):
        e = ErrorHistory()
        assert e.timestamps == []
        assert e.errors_x == []
        assert e.max_len == 300

    def test_append(self):
        e = ErrorHistory()
        e.timestamps.append(0.1)
        e.errors_x.append(1.0)
        assert len(e.timestamps) == 1

    def test_overflow_at_max(self):
        e = ErrorHistory(max_len=5)
        for i in range(5):
            e.timestamps.append(float(i))
        e.timestamps = e.timestamps[-5:]
        assert len(e.timestamps) == 5


class TestMiniPlot:
    def test_plot_can_render_with_data(self):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        from fsoc_tracker.gui.plots import MiniPlot

        app = QApplication.instance() or QApplication([])
        plot = MiniPlot("Test", "#ffffff")
        plot.resize(200, 80)
        plot.set_data([0.0, 1.0], [1.0, 2.0])
        image = QImage(200, 80, QImage.Format.Format_ARGB32)
        image.fill(0)
        plot.render(image)
        assert not image.isNull()
        plot.deleteLater()
        app.processEvents()


class TestBenchmarkView:
    def test_default_values(self):
        b = BenchmarkView()
        assert b.active is False
        assert b.frame_count == 0
        assert b.total_frames == 0
        assert b.acquisition_s is None
        assert b.rmse_px is None

    def test_active_benchmark(self):
        b = BenchmarkView(active=True, frame_count=100, total_frames=200, rmse_px=5.0)
        assert b.active is True
        assert b.frame_count == 100
        assert b.progress == 0.0


class TestApplicationViewState:
    def test_default_state(self):
        s = ApplicationViewState()
        assert s.system_mode == SystemMode.SIMULATION
        assert s.system_state == SystemState.IDLE
        assert s.run_state == RunState.STOPPED

    def test_add_event(self):
        s = ApplicationViewState()
        s.add_event("Test message")
        assert len(s.events) == 1
        assert s.events[0].message == "Test message"
        assert s.events[0].level == "INFO"

    def test_add_event_overflow(self):
        s = ApplicationViewState()
        s.events = [EventLogEntry(timestamp_s=float(i), message=f"msg{i}") for i in range(501)]
        s.add_event("overflow")
        assert len(s.events) == 500

    def test_event_levels(self):
        s = ApplicationViewState()
        s.add_event("info", "INFO")
        s.add_event("warn", "WARNING")
        s.add_event("err", "ERROR")
        s.add_event("ok", "SUCCESS")
        assert len(s.events) == 4

    def test_sub_states(self):
        s = ApplicationViewState()
        assert isinstance(s.camera, CameraViewState)
        assert isinstance(s.perception, PerceptionState)
        assert isinstance(s.tracking, TrackingStateView)
        assert isinstance(s.control, ControlState)
        assert isinstance(s.target, TargetView)
        assert isinstance(s.disturbances, DisturbanceView)
        assert isinstance(s.performance, PerformanceView)
        assert isinstance(s.errors, ErrorHistory)
        assert isinstance(s.benchmark, BenchmarkView)
        assert isinstance(s.scorecard, SIHScorecard)


class TestSIHScorecard:
    def test_default_values(self):
        sc = SIHScorecard()
        assert sc.acquisition_s is None
        assert sc.rmse_px is None
        assert sc.loss_percent is None
        assert sc.reacq_s is None
        assert sc.fps is None

    def test_thresholds(self):
        sc = SIHScorecard()
        assert sc.acquisition_threshold == 2.0
        assert sc.rmse_threshold == 10.0
        assert sc.loss_threshold == 5.0
        assert sc.reacq_threshold == 1.0
        assert sc.fps_threshold == 20.0


class TestEventLogEntry:
    def test_default_values(self):
        e = EventLogEntry()
        assert e.timestamp_s == 0.0
        assert e.level == "INFO"
        assert e.message == ""


# --- Controller Tests ---

class TestApplicationController:
    def test_default_config(self):
        ctrl = ApplicationController()
        cfg = ctrl.config
        assert cfg["mode"] == "simulation"
        assert cfg["trajectory"] == "straight_line"
        assert cfg["camera_width"] == 640
        assert cfg["camera_height"] == 480
        assert cfg["hfov"] == 4.0
        assert cfg["seed"] == 42

    def test_set_mode(self):
        ctrl = ApplicationController()
        ctrl.set_mode("video")
        assert ctrl.config["mode"] == "video"

    def test_set_trajectory(self):
        ctrl = ApplicationController()
        ctrl.set_trajectory("circular", {"radius": 500})
        assert ctrl.config["trajectory"] == "circular"
        assert ctrl.config["trajectory_params"]["radius"] == 500

    def test_set_camera(self):
        ctrl = ApplicationController()
        ctrl.set_camera(hfov=10.0, camera_width=1280)
        assert ctrl.config["hfov"] == 10.0
        assert ctrl.config["camera_width"] == 1280

    def test_set_disturbance(self):
        ctrl = ApplicationController()
        ctrl.set_disturbance(True, "moderate")
        assert ctrl.config["disturbance_enabled"] is True
        assert ctrl.config["disturbance_preset"] == "moderate"

    def test_set_perception(self):
        ctrl = ApplicationController()
        ctrl.set_perception("ai", 0.5)
        assert ctrl.config["perception_backend"] == "ai"
        assert ctrl.config["confidence_threshold"] == 0.5

    def test_save_and_load_config(self, tmp_path):
        ctrl = ApplicationController()
        ctrl.set_mode("video")
        path = str(tmp_path / "test_config.json")
        ctrl.save_config(path)
        assert Path(path).exists()

        ctrl2 = ApplicationController()
        ctrl2.load_config(path)
        assert ctrl2.config["mode"] == "video"

    def test_load_nonexistent_config(self):
        ctrl = ApplicationController()
        with pytest.raises(FileNotFoundError):
            ctrl.load_config("/nonexistent/path.json")

    def test_save_creates_dirs(self, tmp_path):
        ctrl = ApplicationController()
        path = str(tmp_path / "subdir" / "config.json")
        ctrl.save_config(path)
        assert Path(path).exists()

    def test_get_worker_config(self):
        ctrl = ApplicationController()
        ctrl.set_mode("live")
        cfg = ctrl.get_worker_config()
        assert cfg["mode"] == "live"
        assert cfg is not ctrl.config


# --- Theme Tests ---

class TestTheme:
    def test_colors_exist(self):
        assert Colors.BACKGROUND.startswith("#")
        assert Colors.PANEL.startswith("#")
        assert Colors.TEXT.startswith("#")
        assert Colors.ACCENT == "#22d3ee"
        assert Colors.SUCCESS == "#39e58c"
        assert Colors.ERROR == "#ff5c6c"
        assert Colors.TARGET == "#ffb454"

    def test_fonts_exist(self):
        assert "SF Mono" in Fonts.FAMILY or "monospace" in Fonts.FAMILY
        assert "bold" in Fonts.TITLE

    def test_spacing_exists(self):
        assert Spacing.XS == 2
        assert Spacing.SM == 4
        assert Spacing.MD == 8
        assert Spacing.LG == 12
        assert Spacing.XL == 16

    def test_apply_theme(self):
        theme = apply_theme(None)
        assert Colors.BACKGROUND in theme
        assert "QMainWindow" in theme
        assert "QPushButton" in theme
        assert "QTabBar" in theme

    def test_all_color_groups_present(self):
        theme = apply_theme(None)
        for keyword in ["QMainWindow", "QWidget", "QFrame", "QLabel",
                         "QGroupBox", "QPushButton", "QLineEdit",
                         "QTabBar", "QScrollBar", "QSlider"]:
            assert keyword in theme, f"Missing {keyword} in theme"


# --- Worker Tests ---

class TestProcessingWorker:
    def test_worker_creation(self):
        worker = ProcessingWorker()
        assert worker._running is False
        assert worker._paused is False
        assert worker._frame_index == 0
        assert worker._sim_time == 0.0

    def test_configure(self):
        worker = ProcessingWorker()
        worker.configure({
            "mode": "simulation",
            "trajectory": "circular",
            "camera_width": 1280,
            "hfov": 8.0,
        })
        assert worker._config["trajectory"] == "circular"
        assert worker._config["camera_width"] == 1280

    def test_initial_state(self):
        worker = ProcessingWorker()
        s = worker._state
        assert s.system_state == SystemState.IDLE
        assert s.run_state == RunState.STOPPED

    def test_disturbance_enabled_flag_is_authoritative(self):
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker.update_disturbance({
            "disturbance_preset": "moderate",
            "disturbance_enabled": True,
        })
        assert worker._disturbance.config.enabled is True
        worker.update_disturbance({
            "disturbance_preset": "moderate",
            "disturbance_enabled": False,
        })
        assert worker._disturbance.config.enabled is False

    def test_stop_without_start(self):
        worker = ProcessingWorker()
        worker.stop_run()
        assert worker._state.system_state == SystemState.IDLE

    def test_pause_without_start(self):
        worker = ProcessingWorker()
        worker.pause()
        assert worker._paused is True

    def test_resume_without_pause(self):
        worker = ProcessingWorker()
        worker.resume()
        assert worker._paused is False

    @patch("fsoc_tracker.gui.worker.SimulationEngine")
    def test_init_pipeline(self, mock_engine_cls):
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        assert worker._detector is not None
        assert worker._tracker is not None
        assert worker._controller is not None

    @patch("fsoc_tracker.gui.worker.SimulationEngine")
    def test_run_one_step(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_engine.get_state.return_value = MagicMock()
        mock_engine.get_state.return_value.get_active_targets.return_value = []
        mock_engine_cls.return_value = mock_engine

        worker = ProcessingWorker()
        worker.configure({"mode": "simulation", "sim_dt": 1.0 / 30.0})
        worker._init_pipeline()
        worker._state.system_state = SystemState.RUNNING
        worker._sim_dt = 1.0 / 30.0
        worker._sim_time = 0.0
        worker._frame_index = 0
        worker._run_one_step()
        assert worker._sim_time > 0
        assert worker._frame_index == 1
        assert worker._state.performance.frames_processed == 1
        assert worker._state.performance.processing_ms >= 0.0
        assert worker._state.performance.pipeline_fps > 0.0

    def test_default_simulation_target_is_in_front_of_camera(self):
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        worker._sim_engine.step(worker._sim_dt)
        target = worker._sim_engine.get_state().targets[0]
        assert target.x > 999.0
        assert target.y > 999.0
        assert target.z == 500.0

    def test_release_resources(self):
        worker = ProcessingWorker()
        worker._source = MagicMock()
        worker._release_resources()
        assert worker._source is None

    def test_release_resources_with_none(self):
        worker = ProcessingWorker()
        worker._release_resources()
        assert worker._source is None

    def test_brain_gating_enables_lead_only_for_fast_confident(self):
        from fsoc_tracker.ai.mission import (
            MissionAction,
            MissionDecision,
            SafeRecommendation,
            Situation,
        )
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()

        def decide(action, situation, confidence, approved):
            return MissionDecision(
                action=action, situation=situation, confidence=confidence,
                prediction=None, reason="test", timestamp_s=0.0,
                safety=SafeRecommendation(
                    action=action, confidence=confidence, approved=approved,
                    fallback=None, reason="test", timestamp_s=0.0),
            )

        worker._apply_brain_gating(decide(
            MissionAction.TRACK_PREDICTIVE, Situation.FAST_TARGET_MOTION,
            0.9, True))
        assert worker._controller._config.lead_compensation_enabled is True

        for action, situation, conf, approved in [
            (MissionAction.TRACK, Situation.FAST_TARGET_MOTION, 0.9, True),
            (MissionAction.TRACK_PREDICTIVE, Situation.HIGH_NOISE, 0.9, True),
            (MissionAction.TRACK_PREDICTIVE, Situation.FAST_TARGET_MOTION,
             0.4, True),
            (MissionAction.TRACK_PREDICTIVE, Situation.FAST_TARGET_MOTION,
             0.9, False),
        ]:
            worker._apply_brain_gating(decide(action, situation, conf,
                                              approved))
            assert worker._controller._config.lead_compensation_enabled is False
        worker._release_resources()

    def test_worker_configure_default(self):
        worker = ProcessingWorker()
        worker.configure({})
        assert worker._sim_dt == 1.0 / 30.0
        assert worker._mode == SystemMode.SIMULATION

    def test_worker_custom_dt(self):
        worker = ProcessingWorker()
        worker.configure({"sim_dt": 1.0 / 60.0})
        assert worker._sim_dt == 1.0 / 60.0


# --- State Persistence Tests ---

class TestStatePersistence:
    def test_config_save_load_roundtrip(self, tmp_path):
        ctrl = ApplicationController()
        ctrl.set_mode("live")
        ctrl.set_trajectory("sinusoidal", {"amplitude": 200})
        ctrl.set_camera(hfov=6.0, camera_width=1920)
        ctrl.set_disturbance(True, "extreme")
        ctrl.set_perception("hybrid", 0.6)

        path = str(tmp_path / "roundtrip.json")
        ctrl.save_config(path)

        ctrl2 = ApplicationController()
        ctrl2.load_config(path)
        assert ctrl2.config["mode"] == "live"
        assert ctrl2.config["trajectory"] == "sinusoidal"
        assert ctrl2.config["trajectory_params"]["amplitude"] == 200
        assert ctrl2.config["hfov"] == 6.0
        assert ctrl2.config["camera_width"] == 1920
        assert ctrl2.config["disturbance_enabled"] is True
        assert ctrl2.config["disturbance_preset"] == "extreme"
        assert ctrl2.config["perception_backend"] == "hybrid"
        assert ctrl2.config["confidence_threshold"] == 0.6


# --- Mode Switching Tests ---

class TestModeSwitching:
    def test_mode_switch_simulation_to_video(self):
        ctrl = ApplicationController()
        ctrl.set_mode("simulation")
        ctrl.set_mode("video")
        assert ctrl.config["mode"] == "video"

    def test_mode_switch_video_to_live(self):
        ctrl = ApplicationController()
        ctrl.set_mode("video")
        ctrl.set_mode("live")
        assert ctrl.config["mode"] == "live"

    def test_mode_switch_live_to_simulation(self):
        ctrl = ApplicationController()
        ctrl.set_mode("live")
        ctrl.set_mode("simulation")
        assert ctrl.config["mode"] == "simulation"

    def test_worker_mode_switch(self):
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        assert worker._mode == SystemMode.SIMULATION
        worker.configure({"mode": "video"})
        assert worker._mode == SystemMode.VIDEO
        worker.configure({"mode": "live"})
        assert worker._mode == SystemMode.LIVE


# --- Widget State Tests (offscreen) ---

class TestWidgetState:
    @pytest.fixture
    def qapp(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            import sys
            app = QApplication(sys.argv)
        return app

    def test_world_view_state_update(self, qapp):
        from fsoc_tracker.gui.world_view import WorldViewWidget
        widget = WorldViewWidget()
        state = ApplicationViewState()
        state.target.visible = True
        state.target.world_x = 1000.0
        state.target.world_y = 1500.0
        widget.update_state(state)
        assert widget._state is state

    def test_telemetry_panel_state_update(self, qapp):
        from fsoc_tracker.gui.telemetry import TelemetryPanel
        panel = TelemetryPanel()
        state = ApplicationViewState()
        state.system_state = SystemState.RUNNING
        state.camera.fps = 30.0
        state.perception.detected = True
        state.tracking.state = "TRACKING"
        panel.update_state(state)
        assert panel._labels["SYSTEM_State"].text() == "RUNNING"
        assert panel._labels["INPUT_FPS"].text() == "30.0"
        assert panel._labels["PERCEPTION_Status"].text() == "DETECTED"
        assert panel._labels["TRACKING_State"].text() == "TRACKING"

    def test_scorecard_widget_update(self, qapp):
        from fsoc_tracker.gui.scorecard import ScorecardWidget
        widget = ScorecardWidget()
        sc = SIHScorecard(acquisition_s=1.5, rmse_px=5.0, loss_percent=2.0, reacq_s=0.8, fps=25.0)
        widget.update_state(sc)
        assert widget._items["acquisition"]["value"].text() == "1.50 s"
        assert widget._items["rmse"]["status"].text() == "PASS"
        assert widget._items["fps"]["status"].text() == "PASS"

    def test_scorecard_widget_all_fail(self, qapp):
        from fsoc_tracker.gui.scorecard import ScorecardWidget
        widget = ScorecardWidget()
        sc = SIHScorecard(acquisition_s=3.0, rmse_px=15.0, loss_percent=10.0, reacq_s=2.0, fps=10.0)
        widget.update_state(sc)
        assert widget._items["acquisition"]["status"].text() == "FAIL"
        assert widget._items["rmse"]["status"].text() == "FAIL"
        assert widget._items["fps"]["status"].text() == "FAIL"
        assert "FAIL" in widget._overall.text()

    def test_scorecard_widget_not_evaluated(self, qapp):
        from fsoc_tracker.gui.scorecard import ScorecardWidget
        widget = ScorecardWidget()
        sc = SIHScorecard()
        widget.update_state(sc)
        assert widget._items["acquisition"]["status"].text() == "NOT EVAL"
        assert "NOT EVALUATED" in widget._overall.text()

    def test_plots_widget_update(self, qapp):
        from fsoc_tracker.gui.plots import PlotsWidget
        widget = PlotsWidget()
        state = ApplicationViewState()
        state.errors.timestamps = [0.1, 0.2, 0.3]
        state.errors.errors_x = [1.0, 2.0, 3.0]
        state.errors.errors_y = [0.5, 1.5, 2.5]
        state.errors.errors_euclidean = [1.1, 2.5, 3.9]
        state.errors.fps_history = [30.0, 29.5, 30.0]
        widget.update_state(state)
        assert len(widget._err_x._data) == 3

    def test_event_log_panel_update(self, qapp):
        from fsoc_tracker.gui.event_log import EventLogPanel
        panel = EventLogPanel()
        state = ApplicationViewState()
        state.add_event("Test 1")
        state.add_event("Test 2")
        panel.update_state(state)
        assert panel._last_count == 2

    def test_event_log_panel_clear(self, qapp):
        from fsoc_tracker.gui.event_log import EventLogPanel
        panel = EventLogPanel()
        state = ApplicationViewState()
        state.add_event("Test")
        panel.update_state(state)
        panel.clear()
        assert panel._last_count == 0

    def test_ai_panel_update(self, qapp):
        from fsoc_tracker.gui.ai_panel import AIPanel
        panel = AIPanel()
        state = ApplicationViewState()
        state.perception.backend = "ai"
        state.perception.detected = True
        state.perception.confidence = 0.9
        state.perception.processing_ms = 5.0
        panel.update_state(state)
        assert panel._labels["backend"].text() == "ai"
        assert panel._labels["model"].text() == "BeaconCNN"

    def test_ai_panel_hybrid(self, qapp):
        from fsoc_tracker.gui.ai_panel import AIPanel
        panel = AIPanel()
        state = ApplicationViewState()
        state.perception.backend = "hybrid"
        panel.update_state(state)
        assert panel._labels["model"].text() == "Classical+AI"

    def test_benchmark_panel_update(self, qapp):
        from fsoc_tracker.gui.benchmark_panel import BenchmarkPanel
        panel = BenchmarkPanel()
        # Test that the panel has the expected labels
        assert "rmse_px" in panel._labels
        assert "fps" in panel._labels
        assert "frames_processed" in panel._labels
        assert "max_error_px" in panel._labels
        assert "lock_retention_pct" in panel._labels
        assert "link_downtime_s" in panel._labels
        # Test that all five PS methods are offered
        assert panel._mode_combo.count() == 5
        assert panel._method_values == [
            "classical_pid", "kalman_expert", "learned_temporal_expert",
            "learned_temporal_learned_policy", "full_ai_mission",
        ]
        # Test that world combo lists generated benchmark worlds
        assert panel._world_combo.count() == 10
        # Test source gating: only SIMULATION is benchmarkable
        assert panel._source_combo.count() == 3
        assert panel._source_combo.model().item(0).isEnabled()
        assert not panel._source_combo.model().item(1).isEnabled()
        assert not panel._source_combo.model().item(2).isEnabled()
        # Test run-history table columns
        assert panel._history.columnCount() == 8

    def test_control_panel_config(self, qapp):
        from fsoc_tracker.gui.controls import ControlPanel
        panel = ControlPanel()
        panel.set_mode("video")
        assert panel._mode_combo.currentText() == "VIDEO"
        cfg = panel.get_config()
        assert cfg["mode"] == "video"
        assert cfg["trajectory"] == "straight_line"
        assert cfg["camera_width"] == 640

    def test_control_panel_custom_values(self, qapp):
        from fsoc_tracker.gui.controls import ControlPanel
        panel = ControlPanel()
        panel._target_size.setValue(15.0)
        panel._hfov.setValue(8.0)
        panel._cam_w.setValue(1280)
        panel._max_pan.setValue(7.0)
        panel._max_tilt.setValue(3.0)
        cfg = panel.get_config()
        assert cfg["target_size"] == 15.0
        assert cfg["hfov"] == 8.0
        assert cfg["camera_width"] == 1280
        assert cfg["max_pan_rate"] == 7.0
        assert cfg["max_tilt_rate"] == 3.0


# --- Stop/Pause Resume Tests ---

class TestStopPauseResume:
    def test_stop_sets_idle(self):
        worker = ProcessingWorker()
        worker._running = True
        worker._state.system_state = SystemState.RUNNING
        worker._state.run_state = "PLAYING"
        worker.stop_run()
        assert worker._state.system_state == SystemState.IDLE
        assert worker._state.run_state == "STOPPED"

    def test_pause_sets_paused(self):
        worker = ProcessingWorker()
        worker._state.system_state = SystemState.RUNNING
        worker._running = True
        worker.pause()
        assert worker._state.system_state == SystemState.PAUSED
        assert worker._state.run_state == "PAUSED"

    def test_resume_sets_running(self):
        worker = ProcessingWorker()
        worker._state.system_state = SystemState.PAUSED
        worker._paused = True
        worker._running = True
        worker.resume()
        assert worker._state.system_state == SystemState.RUNNING
        assert worker._state.run_state == "PLAYING"

    @patch("fsoc_tracker.gui.worker.SimulationEngine")
    def test_step_when_paused(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_engine.get_state.return_value = MagicMock()
        mock_engine.get_state.return_value.get_active_targets.return_value = []
        mock_engine_cls.return_value = mock_engine

        worker = ProcessingWorker()
        worker.configure({"mode": "simulation", "sim_dt": 1.0 / 30.0})
        worker._running = True
        worker._paused = True
        worker._state.system_state = SystemState.PAUSED
        worker._init_pipeline()
        worker._sim_dt = 1.0 / 30.0
        worker._sim_time = 0.0
        worker._frame_index = 0
        worker.step()
        assert worker._frame_index == 1


# --- Main Window Tests ---

class TestMainWindow:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_create_world_gating(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._world_config is None
        # World actions build real configs (no scene presets).
        window._on_world_action("New empty world")
        assert window._world_config is not None
        assert window._world_config.beacons == []
        window._on_world_action("Random world (seed)")
        assert len(window._world_config.beacons) >= 1
        assert window._world_config.primary_beacon_id is not None
        window.close()

    def test_start_blocked_without_world(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_config = None
        window._on_start()
        # No worker started; explanatory event logged.
        assert window._worker is None
        assert any("CREATE WORLD" in e.message
                   for e in window._state.events)
        window.close()

    def test_start_runs_with_world(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_world_action("Random world (seed)")
        window._on_start()
        assert window._worker is not None
        window._worker.stop_run()
        window.close()

    def test_main_window_creation(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert "FSOC OPTICAL TRACKING SYSTEM" in window.windowTitle()
        assert window._controller is not None
        assert window._state is not None
        window.close()

    def test_main_window_has_panels(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._world_view is not None
        assert window._telemetry is not None
        assert "STANDBY" in window._header_state.text()
        assert window._header_fps.text().startswith("PROC")
        assert window._world_view.objectName() == "WorldViewport"
        window.close()
        assert window._control_panel is not None
        assert window._scorecard is not None
        assert window._ai_panel is not None
        assert window._benchmark_panel is not None
        assert window._event_log is not None
        assert window._plots is not None
        window.close()

    def test_main_window_stop_without_worker(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_stop()
        window.close()

    def test_main_window_pause_without_worker(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_pause()
        window.close()

    def test_main_window_step_without_worker(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_step()
        window.close()

    def test_main_window_about(self, qapp):
        from fsoc_tracker.gui.about import AboutDialog
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        dlg = AboutDialog(window)
        assert dlg.windowTitle() == "About"
        assert dlg.width() == 360
        window.close()

    def test_main_window_status_bar(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._status_label.text() == "IDLE"
        assert window._mode_label.text() == "SIMULATION"
        assert window._fps_label.text() == "0 FPS"
        window.close()

    def test_refresh_with_none_worker(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._worker = None
        window._refresh_from_worker()
        window.close()


# --- Integration Tests ---

class TestGUIIntegration:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_state_lifecycle(self):
        s = ApplicationViewState()
        assert s.system_state == SystemState.IDLE
        s.system_state = SystemState.INITIALIZING
        s.add_event("Initializing pipeline")
        s.system_state = SystemState.RUNNING
        s.add_event("Pipeline started")
        s.system_state = SystemState.PAUSED
        s.add_event("Pipeline paused")
        s.system_state = SystemState.IDLE
        s.add_event("Pipeline stopped")
        assert len(s.events) == 4

    def test_perception_detection_flow(self):
        s = ApplicationViewState()
        s.perception.detected = True
        s.perception.confidence = 0.85
        s.perception.detection_x = 320.0
        s.perception.detection_y = 240.0
        s.tracking.state = "TRACKING"
        s.tracking.locked = True
        s.tracking.estimated_x = 321.0
        s.tracking.estimated_y = 239.0
        s.errors.timestamps.append(0.0)
        s.errors.errors_euclidean.append(1.414)
        s.scorecard.rmse_px = 1.414
        s.scorecard.fps = 30.0
        assert s.perception.detected
        assert s.tracking.locked
        assert s.scorecard.rmse_px == 1.414

    def test_control_flow(self):
        s = ApplicationViewState()
        s.control.pan_deg = 1.5
        s.control.tilt_deg = -0.5
        s.control.pan_command = 2.0
        s.control.tilt_command = -1.0
        s.control.pan_saturated = False
        s.control.tilt_saturated = True
        assert s.control.pan_saturated is False
        assert s.control.tilt_saturated is True

    def test_worker_config_apply(self):
        ctrl = ApplicationController()
        ctrl.set_mode("simulation")
        ctrl.set_trajectory("circular")
        ctrl.set_camera(hfov=8.0, camera_width=1280)
        ctrl.set_disturbance(True, "moderate")
        ctrl.set_perception("ai", 0.5)
        cfg = ctrl.get_worker_config()
        worker = ProcessingWorker()
        worker.configure(cfg)
        assert worker._config["mode"] == "simulation"
        assert worker._config["trajectory"] == "circular"
        assert worker._config["hfov"] == 8.0
        assert worker._config["camera_width"] == 1280
        assert worker._config["disturbance_enabled"] is True

    def test_control_panel_config_apply(self, qapp):
        from fsoc_tracker.gui.controls import ControlPanel
        panel = ControlPanel()
        panel._mode_combo.setCurrentText("VIDEO")
        panel._traj_combo.setCurrentText("circular")
        panel._cam_w.setValue(1280)
        panel._cam_h.setValue(720)
        panel._hfov.setValue(8.0)
        panel._target_size.setValue(15.0)
        ctrl = ApplicationController()
        cfg = panel.get_config()
        ctrl._config.update(cfg)
        assert ctrl.config["mode"] == "video"
        assert ctrl.config["trajectory"] == "circular"
        assert ctrl.config["camera_width"] == 1280
        assert ctrl.config["hfov"] == 8.0
        assert ctrl.config["target_size"] == 15.0

    def test_scorecard_pass_all(self, qapp):
        from fsoc_tracker.gui.scorecard import ScorecardWidget
        widget = ScorecardWidget()
        sc = SIHScorecard(acquisition_s=1.0, rmse_px=5.0, loss_percent=1.0, reacq_s=0.5, fps=30.0)
        widget.update_state(sc)
        for key in ["acquisition", "rmse", "loss", "reacq", "fps"]:
            assert widget._items[key]["status"].text() == "PASS"
        assert "PASS" in widget._overall.text()

    def test_scorecard_fail_all(self, qapp):
        from fsoc_tracker.gui.scorecard import ScorecardWidget
        widget = ScorecardWidget()
        sc = SIHScorecard(acquisition_s=5.0, rmse_px=20.0, loss_percent=15.0, reacq_s=3.0, fps=5.0)
        widget.update_state(sc)
        for key in ["acquisition", "rmse", "loss", "reacq", "fps"]:
            assert widget._items[key]["status"].text() == "FAIL"
        assert "FAIL" in widget._overall.text()

    def test_error_history_comprehensive(self):
        s = ApplicationViewState()
        for i in range(100):
            t = i * 0.033
            err = 5.0 * math.sin(t)
            s.errors.timestamps.append(t)
            s.errors.errors_x.append(err)
            s.errors.errors_y.append(err * 0.5)
            s.errors.errors_euclidean.append(abs(err))
            s.errors.fps_history.append(30.0 + math.sin(t) * 2)
        assert len(s.errors.timestamps) == 100
        assert len(s.errors.errors_x) == 100
        assert len(s.errors.errors_y) == 100
        assert len(s.errors.errors_euclidean) == 100
        assert len(s.errors.fps_history) == 100
        assert all(v > 0 for v in s.errors.fps_history)

    def test_trail_management(self):
        s = ApplicationViewState()
        for i in range(10):
            s.target.trail.append((float(i * 100), float(i * 50)))
        assert len(s.target.trail) == 10
        s.target.trail = s.target.trail[-200:]
        assert len(s.target.trail) == 10

    def test_multiple_modes(self):
        for mode in ["simulation", "video", "live"]:
            ctrl = ApplicationController()
            ctrl.set_mode(mode)
            assert ctrl.config["mode"] == mode
            worker = ProcessingWorker()
            worker.configure(ctrl.get_worker_config())
            assert worker._mode == SystemMode(mode)


class TestOpticalRangeFormatting:
    def test_no_link_reports_na(self):
        from fsoc_tracker.gui.hud_panels import format_optical_range
        assert format_optical_range("NO_LINK", 0.0) == "RANGE: N/A"
        assert format_optical_range("NO_LINK", 450.0) == "RANGE: N/A"

    def test_zero_range_reports_na(self):
        from fsoc_tracker.gui.hud_panels import format_optical_range
        assert format_optical_range("LOCKED", 0.0) == "RANGE: N/A"

    def test_genuine_range_reported(self):
        from fsoc_tracker.gui.hud_panels import format_optical_range
        assert format_optical_range("LOCKED", 450.4) == "RANGE: 450 m"
        assert format_optical_range("DEGRADED", 1950.0) == "RANGE: 1950 m"


class TestMissionFlowStrip:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def _window(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        qapp.processEvents()
        return window

    def test_flow_strip_exists_with_nine_stages(self, qapp):
        from fsoc_tracker.gui.main_window import FLOW_STAGES
        window = self._window(qapp)
        assert window._flow_strip is not None
        assert len(FLOW_STAGES) == 9
        assert FLOW_STAGES == [
            "SETUP", "START", "CAMERA POV", "SEARCH", "ACQUIRE",
            "TRACK", "PREDICT", "CONTROL", "RECOVER",
        ]
        for stage in FLOW_STAGES:
            assert stage in window._flow_lamps
        window.close()

    def test_idle_lights_setup(self, qapp):
        from fsoc_tracker.gui.main_window import PAGE_CAMERA_TRACKING
        from fsoc_tracker.gui.state import ApplicationViewState, SystemState
        window = self._window(qapp)
        s = ApplicationViewState()
        s.system_state = SystemState.IDLE
        active = window._flow_active_stages(s)
        assert active["SETUP"] is True
        assert active["START"] is False
        assert active["TRACK"] is False
        assert active["RECOVER"] is False
        # Camera page is default: CAMERA POV lamp reflects view state
        assert active["CAMERA POV"] == (window._current_page == PAGE_CAMERA_TRACKING)
        window.close()

    def test_tracking_lights_track_and_control(self, qapp):
        from fsoc_tracker.gui.state import ApplicationViewState, SystemState
        window = self._window(qapp)
        s = ApplicationViewState()
        s.system_state = SystemState.RUNNING
        s.performance.frames_processed = 50
        s.tracking.state = "TRACKING"
        s.tracking.prediction_only = False
        s.control.pan_command = 0.5
        s.control.tilt_command = 0.0
        active = window._flow_active_stages(s)
        assert active["SETUP"] is False
        assert active["START"] is False
        assert active["TRACK"] is True
        assert active["CONTROL"] is True
        assert active["PREDICT"] is False
        assert active["SEARCH"] is False
        window.close()

    def test_prediction_and_recovery_states(self, qapp):
        from fsoc_tracker.gui.state import ApplicationViewState, SystemState
        window = self._window(qapp)
        s = ApplicationViewState()
        s.system_state = SystemState.RUNNING
        s.performance.frames_processed = 50
        s.tracking.state = "TRACKING"
        s.tracking.prediction_only = True
        active = window._flow_active_stages(s)
        assert active["PREDICT"] is True
        assert active["TRACK"] is False
        s.tracking.state = "LOST"
        s.tracking.prediction_only = False
        active = window._flow_active_stages(s)
        assert active["RECOVER"] is True
        assert active["TRACK"] is False
        s.tracking.state = "SEARCHING"
        active = window._flow_active_stages(s)
        assert active["SEARCH"] is True
        s.tracking.state = "ACQUIRING"
        active = window._flow_active_stages(s)
        assert active["ACQUIRE"] is True
        window.close()

    def test_flow_frame_counter_from_runtime(self, qapp):
        from fsoc_tracker.gui.state import ApplicationViewState, SystemState
        window = self._window(qapp)
        s = ApplicationViewState()
        s.system_state = SystemState.RUNNING
        s.performance.frames_processed = 1234
        window._update_flow_strip(s)
        assert window._flow_frame.text() == "FRAME 1234"
        window.close()

    def test_control_sidebar_reflects_commands(self, qapp):
        window = self._window(qapp)
        assert hasattr(window, "_cam_cmd_pan")
        assert hasattr(window, "_cam_cmd_tilt")
        assert hasattr(window, "_cam_ctrl_mode")
        window.close()


class TestCameraPredictionOverlay:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_workspace_with_prediction_state_no_crash(self, qapp):
        import numpy as np

        from fsoc_tracker.gui.camera_workspace import CameraTrackingWorkspace
        from fsoc_tracker.gui.state import ApplicationViewState
        ws = CameraTrackingWorkspace()
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[20:28, 28:36] = (200, 200, 200)
        s = ApplicationViewState()
        s.camera.image_width = 64
        s.camera.image_height = 48
        s.perception.detected = True
        s.perception.detection_x = 32.0
        s.perception.detection_y = 24.0
        s.tracking.state = "TRACKING"
        s.tracking.locked = True
        s.tracking.estimated_x = 32.0
        s.tracking.estimated_y = 24.0
        s.ai_state.prediction_dx = 4.0
        s.ai_state.prediction_dy = -2.0
        s.ai_state.prediction_horizon_s = 0.1
        s.ai_state.prediction_uncertainty_x = 3.0
        s.ai_state.prediction_uncertainty_y = 3.0
        ws.update_frame(img, s)
        ws.resize(320, 240)
        ws.show()
        qapp.processEvents()
        ws.hide()


class TestUserControlledBeaconWiring:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def _sim_worker_user(self):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        worker.configure({
            "mode": "simulation",
            "trajectory": "user_controlled",
            "camera_width": 64,
            "camera_height": 48,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
        })
        worker._init_pipeline()
        return worker

    def test_worker_nudge_and_reset(self, qapp):
        worker = self._sim_worker_user()
        assert worker.nudge_beacon(10.0, 0.0, -5.0, 0) is True
        assert worker.nudge_beacon(1.0, 1.0, 1.0, 99) is False
        assert worker.reset_beacon(0) is True
        worker._release_resources()

    def test_worker_nudge_rejected_without_sim(self, qapp):
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        assert worker.nudge_beacon(1.0, 0.0, 0.0, 0) is False
        assert worker.set_terminal_pose(1.0, 2.0, 3.0) is False

    def test_worker_set_terminal_pose(self, qapp):
        worker = self._sim_worker_user()
        assert worker.set_terminal_pose(500.0, 1000.0, 1500.0) is True
        assert worker._camera.state.position_x == pytest.approx(500.0)
        assert worker._camera.state.position_z == pytest.approx(1500.0)
        assert worker.view_state.terminal_a.world_x == pytest.approx(500.0)
        worker._release_resources()

    def test_beacon_key_routing(self, qapp):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.state import TargetView
        window = MainWindow()
        # No worker -> key ignored
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_D, Qt.NoModifier)
        assert window._try_beacon_key(ev) is False
        # Worker with user-controlled beacon -> key consumed
        worker = self._sim_worker_user()
        window._worker = worker
        tv = TargetView(target_id=0, trajectory_type="user_controlled")
        window._state.targets_all = [tv]
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_D, Qt.NoModifier)
        assert window._try_beacon_key(ev) is True
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_R, Qt.NoModifier)
        assert window._try_beacon_key(ev) is True
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F1, Qt.NoModifier)
        assert window._try_beacon_key(ev) is False
        worker._release_resources()
        window.close()

    def test_terminal_placed_signal_wired(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        worker = self._sim_worker_user()
        window._worker = worker
        window._on_terminal_placed(400.0, 1000.0, 1200.0)
        assert worker._camera.state.position_x == pytest.approx(400.0)
        assert worker._camera.state.position_z == pytest.approx(1200.0)
        worker._release_resources()
        window.close()


class TestTelemetryAndEventExport:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_telemetry_angular_error_and_fov(self, qapp):
        from fsoc_tracker.gui.state import ApplicationViewState
        from fsoc_tracker.gui.telemetry import TelemetryPanel
        panel = TelemetryPanel()
        s = ApplicationViewState()
        s.camera.image_width = 640
        s.camera.image_height = 480
        s.optical_link.angular_error_deg = 0.42
        s.perception.detected = True
        s.perception.detection_x = 100.0
        s.perception.detection_y = 100.0
        panel.update_state(s)
        assert "TRACKING_Ang Err" in panel._labels
        assert "0.42" in panel._labels["TRACKING_Ang Err"].text()
        assert panel._labels["TRACKING_FOV"].text() == "IN FOV"
        s.perception.detected = False
        s.tracking.state = "NO_TRACK"
        panel.update_state(s)
        assert panel._labels["TRACKING_FOV"].text() == "NO TARGET"

    def test_event_log_exports(self, qapp, tmp_path):
        from fsoc_tracker.gui.event_log import EventLogPanel
        from fsoc_tracker.gui.state import ApplicationViewState
        panel = EventLogPanel()
        s = ApplicationViewState()
        s.elapsed_s = 1.5
        s.add_event("TRACK_STARTED", "SUCCESS")
        s.add_event("TARGET_LOST", "ERROR")
        panel.update_state(s)
        jp = panel.export_jsonl(str(tmp_path / "events.jsonl"))
        cp = panel.export_csv(str(tmp_path / "events.csv"))
        import json
        with open(jp, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["message"] == "TRACK_STARTED"
        assert first["timestamp_s"] == pytest.approx(1.5)
        with open(cp, encoding="utf-8") as f:
            rows = f.read().strip().split("\n")
        assert rows[0] == "timestamp_s,level,message"
        assert "TARGET_LOST" in rows[2]

    def test_ai_panel_mission_model(self, qapp):
        from fsoc_tracker.gui.ai_panel import AIPanel
        from fsoc_tracker.gui.state import ApplicationViewState
        panel = AIPanel()
        s = ApplicationViewState()
        s.ai_state.model_status = "expert_only"
        s.ai_state.fallback_active = False
        panel.update_state(s)
        assert panel._labels["mission_model"].text() == "expert_only"
        s.ai_state.fallback_active = True
        panel.update_state(s)
        assert "FALLBACK" in panel._labels["mission_model"].text()


class TestLayouts:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_five_layouts_apply_without_restart(self, qapp):
        from fsoc_tracker.gui.main_window import LAYOUTS, MainWindow
        window = MainWindow()
        for name, (central, sidebar, _desc) in LAYOUTS.items():
            assert window.set_layout(name) is True
            assert window._body_stack.currentIndex() == central
            assert window._sidebar_stack.currentIndex() == sidebar
            assert window.current_layout == name
            qapp.processEvents()
        assert len(LAYOUTS) == 5
        assert window.set_layout("NOPE") is False
        window.close()

    def test_optical_lab_differs_from_world(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window.set_layout("OPTICAL LAB")
        # OPTICAL LAB: world central + analysis sidebar (link dominant)
        from fsoc_tracker.gui.main_window import PAGE_ANALYSIS, PAGE_WORLD
        assert window._body_stack.currentIndex() == PAGE_WORLD
        assert window._sidebar_stack.currentIndex() == PAGE_ANALYSIS
        window.close()

    def test_manual_navigation_marks_custom(self, qapp):
        from fsoc_tracker.gui.main_window import (
            LAYOUT_TRACKING_CONSOLE,
            PAGE_MISSION_SETUP,
            MainWindow,
        )
        window = MainWindow()
        assert window.current_layout == LAYOUT_TRACKING_CONSOLE
        window._navigate_to(PAGE_MISSION_SETUP)
        assert window.current_layout == "CUSTOM"
        window.close()

    def test_layout_survives_running_worker(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._worker is None
        assert window.set_layout("BENCHMARK LAB") is True
        assert window.set_layout("TRACKING CONSOLE") is True
        window.close()


class TestFloatingPanels:
    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_toggle_all_panels(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        for name in ("AI", "LINK", "TELEMETRY", "EVENTS"):
            assert window.toggle_floating_panel(name) is True
            qapp.processEvents()
        assert window.toggle_floating_panel("BOGUS") is False
        # Toggle off again
        for name in ("AI", "LINK", "TELEMETRY", "EVENTS"):
            assert window.toggle_floating_panel(name) is False
        window.close()

    def test_floating_panels_receive_state(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.state import ApplicationViewState
        window = MainWindow()
        window.toggle_floating_panel("TELEMETRY")
        window.toggle_floating_panel("EVENTS")
        s = ApplicationViewState()
        s.elapsed_s = 2.0
        s.add_event("HELLO", "INFO")
        window._update_floating_panels(s)
        qapp.processEvents()
        tel = window._floating_panels["TELEMETRY"]
        assert tel._labels["SYSTEM_Elapsed"].text() == "2.00s"
        window.close()


class TestFailureRiskWiring:
    def test_trained_weights_load(self):
        import os

        from fsoc_tracker.ai.failure_predictor import FailurePredictor
        path = "artifacts/models/failure-v1/failure-v1.npz"
        if not os.path.exists(path):
            pytest.skip("failure artifact not shipped")
        p = FailurePredictor.load(path)
        assert p.trained is True

    def test_worker_loads_failure_weights(self):
        import os
        if not os.path.exists("artifacts/models/failure-v1/failure-v1.npz"):
            pytest.skip("failure artifact not shipped")
        from fsoc_tracker.gui.worker import ProcessingWorker
        worker = ProcessingWorker()
        assert worker._failure_predictor.trained is True

    def test_ai_state_defaults(self):
        from fsoc_tracker.gui.state import ApplicationViewState
        s = ApplicationViewState()
        assert s.ai_state.failure_risk == 0.0
        assert s.ai_state.roi_mode == "fixed"
        assert s.ai_state.model_status == "expert_only"


class TestPredictionHorizon:
    def test_default_and_setter(self):
        from fsoc_tracker.ai.mission import AIMissionBrain
        brain = AIMissionBrain()
        assert brain.prediction_horizon_s == pytest.approx(0.1)
        brain.prediction_horizon_s = 0.25
        assert brain.prediction_horizon_s == pytest.approx(0.25)
        with pytest.raises(ValueError):
            brain.prediction_horizon_s = -1.0
        custom = AIMissionBrain(prediction_horizon_s=0.5)
        assert custom.prediction_horizon_s == pytest.approx(0.5)

    def test_horizon_flows_to_prediction(self):
        from fsoc_tracker.ai.mission import AIMissionBrain, MissionObservation, ObservationFeatures
        brain = AIMissionBrain(prediction_horizon_s=0.25)
        obs = MissionObservation(
            features=ObservationFeatures(
                timestamp_s=0.0, detected=True, confidence=0.9,
                residual_px=1.0, uncertainty_x_px=2.0, uncertainty_y_px=2.0,
                velocity_x_px_s=100.0, velocity_y_px_s=0.0,
                distance_from_center_px=10.0, time_since_detection_s=0.0,
                latency_ms=5.0, source_fps=30.0, processing_fps=30.0,
                candidate_count=1),
            camera_pan_deg=0.0, camera_tilt_deg=0.0)
        decision = brain.decide(obs)
        assert decision.prediction is not None
        assert decision.prediction.horizon_s == pytest.approx(0.25)
        assert decision.prediction.mean_x_px == pytest.approx(25.0)

    def test_brain_model_status(self):
        from fsoc_tracker.ai.mission import AIMissionBrain
        assert AIMissionBrain().model_status == "expert_only"


class TestMissionSetupWiring:
    """Every mission-setup control must reach the backend exactly once."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_single_config_surface(self, qapp):
        """Setup page holds the 3D world only; config lives in sidebar."""
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._setup_world_view is not None
        assert not hasattr(window, "_build_setup_config_panel")
        # The setup page contains no tab widget anymore.
        from PySide6.QtWidgets import QTabWidget
        assert window._mission_setup_page.findChild(QTabWidget) is None
        window.close()

    def test_unified_config_reads_visible_widgets(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._traj_combo.setCurrentText("circular")
        window._seed_spin.setValue(7)
        window._cam_w.setValue(800)
        cfg = window._get_unified_config()
        assert cfg["trajectory"] == "circular"
        assert cfg["seed"] == 7
        assert cfg["camera_width"] == 800
        window.close()

    def test_nav_labels_fit_rail(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        for btn in window._nav_buttons:
            assert len(btn.text()) <= 8
            assert btn.toolTip() != ""
        window.close()

    def test_flow_idle_shows_setup_only(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        active = {k for k, v in
                  window._flow_active_stages(window._state).items() if v}
        assert active <= {"SETUP", "CAMERA POV"}
        assert "SEARCH" not in active
        window.close()

    def test_add_beacon_uses_widgets(self, qapp):
        """Add honors trajectory/size/seed widgets; no hard-coded spawn."""
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.worker import ProcessingWorker
        window = MainWindow()
        window._traj_combo.setCurrentText("circular")
        window._seed_spin.setValue(1234)
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        window._worker = worker
        before = len(worker._sim_engine.get_state().targets)
        window._on_add_beacon()
        after = worker._sim_engine.get_state().targets
        assert len(after) == before + 1
        assert after[-1].trajectory_type == "circular"
        worker._release_resources()
        window.close()

    def test_add_beacon_stages_into_idle_world(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_world_action("New empty world")
        window._traj_combo.setCurrentText("figure_8")
        window._on_add_beacon()
        assert len(window._world_config.beacons) == 1
        assert window._world_config.beacons[0].trajectory == "figure_8"
        window.close()

    def test_remove_beacon_reaches_engine(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.worker import ProcessingWorker
        window = MainWindow()
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        window._worker = worker
        targets = worker._sim_engine.get_state().targets
        assert len(targets) >= 1
        bid = targets[0].target_id
        window._state.selected_object_id = bid
        window._on_remove_beacon()
        remaining = [t.target_id
                     for t in worker._sim_engine.get_state().targets]
        assert bid not in remaining
        assert window._state.selected_object_id is None
        worker._release_resources()
        window.close()

    def test_remove_beacon_from_idle_world(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._on_world_action("Random world (seed)")
        bid = window._world_config.beacons[0].beacon_id
        window._state.selected_object_id = bid
        window._on_remove_beacon()
        assert all(b.beacon_id != bid
                   for b in window._world_config.beacons)
        window.close()

    def test_set_beacon_never_spawns(self, qapp):
        """Designation must not mutate the world (no stray beacons)."""
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.worker import ProcessingWorker
        window = MainWindow()
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        window._worker = worker
        n_before = len(worker._sim_engine.get_state().targets)
        window._state.selected_object_id = None
        window._on_set_beacon()
        n_after = len(worker._sim_engine.get_state().targets)
        assert n_after == n_before
        worker._release_resources()
        window.close()

    def test_drag_nudges_user_controlled_only(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.state import TargetView
        from fsoc_tracker.gui.worker import ProcessingWorker
        window = MainWindow()
        window._state.targets_all = [
            TargetView(target_id=0, world_x=100.0, world_y=100.0,
                       world_z=100.0, trajectory_type="user_controlled"),
            TargetView(target_id=1, world_x=200.0, world_y=200.0,
                       world_z=200.0, trajectory_type="straight_line"),
        ]
        nudges = []
        worker = ProcessingWorker()
        worker.nudge_beacon = lambda dx, dy, dz, bid: nudges.append(
            (dx, dy, dz, bid)) or True
        window._worker = worker
        window._on_beacon_dragged(0, 110.0, 100.0, 100.0)
        assert nudges == [(10.0, 0.0, 0.0, 0)]
        window._on_beacon_dragged(1, 210.0, 200.0, 200.0)
        assert len(nudges) == 1  # scripted beacon untouched
        assert any("trajectory" in e.message
                   for e in window._state.events)
        window.close()

    def test_disturbance_widgets_push_live(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        from fsoc_tracker.gui.worker import ProcessingWorker
        window = MainWindow()
        worker = ProcessingWorker()
        worker.configure({"mode": "simulation"})
        worker._init_pipeline()
        window._worker = worker
        worker.start_run()
        try:
            window._dist_checks["noise"].setChecked(True)
            assert worker._disturbance.config.noise.enabled is True
            window._dist_checks["noise"].setChecked(False)
            assert worker._disturbance.config.noise.enabled is False
        finally:
            worker.stop_run()
            worker._release_resources()
        window.close()


class TestVideoSourceSelection:
    """VIDEO mode must offer a working file path, not a dead end."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_source_video_prompts_and_starts(self, qapp, tmp_path, monkeypatch):
        import os

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        mp4 = os.path.join("test_videos", "test_target.mp4")
        monkeypatch.setattr(
            "fsoc_tracker.gui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (mp4, ""),
        )
        window._src_combo.setCurrentText("VIDEO")
        qapp.processEvents()
        assert window._worker is not None
        assert window._last_video_path == mp4
        assert window._src_combo.currentText() == "VIDEO"
        window._worker.stop_run()
        window.close()

    def test_source_video_cancel_reverts_combo(self, qapp, monkeypatch):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        monkeypatch.setattr(
            "fsoc_tracker.gui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        window._src_combo.setCurrentText("VIDEO")
        qapp.processEvents()
        assert window._src_combo.currentText() == "SIMULATION"
        assert window._worker is None
        window.close()

    def test_open_video_file_bad_path_fails_cleanly(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        ok = window._open_video_file("/nonexistent/nope.mp4")
        assert ok is False
        assert any("VIDEO failed" in e.message for e in window._state.events)
        window.close()

    def test_start_video_uses_last_path(self, qapp):
        import os

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        mp4 = os.path.join("test_videos", "test_target.mp4")
        window._last_video_path = mp4
        window._control_panel._mode_combo.setCurrentText("VIDEO")
        window._on_start()
        assert window._worker is not None
        window._worker.stop_run()
        window.close()

    def test_start_video_without_path_prompts(self, qapp, monkeypatch):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._last_video_path = None
        window._control_panel._mode_combo.setCurrentText("VIDEO")
        monkeypatch.setattr(
            "fsoc_tracker.gui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        window._on_start()
        assert window._worker is None
        assert any("video file" in e.message for e in window._state.events)
        window.close()

    def test_sidebar_mode_mirror(self, qapp, tmp_path, monkeypatch):
        import os

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        mp4 = os.path.join("test_videos", "test_target.mp4")
        monkeypatch.setattr(
            "fsoc_tracker.gui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (mp4, ""),
        )
        window._mission_mode_combo.setCurrentText("VIDEO")
        qapp.processEvents()
        assert window._worker is not None
        assert window._src_combo.currentText() == "VIDEO"
        window._worker.stop_run()
        window.close()

    def test_top_combo_syncs_sidebar(self, qapp, tmp_path, monkeypatch):
        import os

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        mp4 = os.path.join("test_videos", "test_target.mp4")
        monkeypatch.setattr(
            "fsoc_tracker.gui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (mp4, ""),
        )
        window._src_combo.setCurrentText("VIDEO")
        qapp.processEvents()
        assert window._mission_mode_combo.currentText() == "VIDEO"
        assert window._worker is not None
        window._worker.stop_run()
        window.close()


class TestShapeAndDisturbanceTuning:
    """PS compliance: shape selection + numeric disturbance tuning."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_shape_label_mapping(self):
        from fsoc_tracker.gui.controls import _shape_label_to_value
        assert _shape_label_to_value("Spot (soft)") == "spot"
        assert _shape_label_to_value("Square") == "square"
        assert _shape_label_to_value("Circular") == "circular"
        assert _shape_label_to_value("") == "spot"

    def test_sensor_shape_kwargs(self):
        from fsoc_tracker.gui.worker import _sensor_shape_kwargs
        assert _sensor_shape_kwargs("spot") == {}
        assert _sensor_shape_kwargs("square") == {
            "beacon_shape": "square", "beacon_soft_edges": False}
        assert _sensor_shape_kwargs("circular") == {
            "beacon_shape": "circular", "beacon_soft_edges": False}

    def test_deposit_shape_override_default_unchanged(self):
        import numpy as np

        from fsoc_tracker.simulation.sensor.beacon import deposit_beacon
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        cfg = SensorConfig()
        a = np.zeros((64, 64)); b = np.zeros((64, 64))
        deposit_beacon(a, 32.0, 32.0, 10.0, 200.0, cfg)
        deposit_beacon(b, 32.0, 32.0, 10.0, 200.0, cfg,
                       shape_override="square")
        assert np.array_equal(a, b)

    def test_deposit_hard_square_differs_from_spot(self):
        import numpy as np

        from fsoc_tracker.simulation.sensor.beacon import deposit_beacon
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        cfg = SensorConfig(beacon_soft_edges=False)
        a = np.zeros((64, 64)); b = np.zeros((64, 64))
        deposit_beacon(a, 32.0, 32.0, 10.0, 200.0, cfg)
        deposit_beacon(b, 32.0, 32.0, 10.0, 200.0, cfg,
                       shape_override="circular")
        assert not np.array_equal(a, b)

    def test_beacon_config_shape_default(self):
        from fsoc_tracker.simulation.scenario import BeaconConfig
        assert BeaconConfig().shape == "square"

    def test_control_panel_shape_and_tuning_keys(self, qapp):
        from fsoc_tracker.gui.controls import ControlPanel
        panel = ControlPanel()
        cfg = panel.get_config()
        assert cfg["target_shape"] == "spot"
        assert cfg["disturbance_noise_sigma"] == 5.0
        assert cfg["disturbance_noise_density"] == 0.05
        assert cfg["disturbance_jitter_amp"] == 5.0
        assert cfg["disturbance_platform_type"] == "linear"
        assert cfg["disturbance_effects"]["noise_gaussian"] is True
        assert cfg["disturbance_effects"]["noise_salt_pepper"] is True
        assert cfg["disturbance_effects"]["noise_poisson"] is True
        panel.close()

    def test_main_window_tuning_keys(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        tuning = window._disturbance_tuning()
        assert tuning["disturbance_noise_sigma"] == 5.0
        assert tuning["disturbance_jitter_amp"] == 5.0
        assert tuning["disturbance_platform_type"] == "linear"
        cfg = window._get_unified_config()
        assert cfg["target_shape"] == "spot"
        assert cfg["disturbance_noise_sigma"] == 5.0
        window.close()

    def test_apply_overrides_tuning(self, qapp):
        from fsoc_tracker.disturbances.config import (
            DisturbanceConfig,
            JitterConfig,
            NoiseConfig,
            PlatformMotionConfig,
            PlatformMotionType,
        )
        from fsoc_tracker.gui.worker import ProcessingWorker
        w = ProcessingWorker()
        dist = DisturbanceConfig(
            noise=NoiseConfig(), jitter=JitterConfig(),
            platform_motion=PlatformMotionConfig())
        w._apply_effect_overrides(
            dist,
            {"noise": True, "noise_gaussian": True,
             "noise_salt_pepper": False, "noise_poisson": True,
             "jitter": True, "platform_motion": True},
            0.5,
            tuning={"disturbance_noise_sigma": 12.0,
                    "disturbance_noise_density": 0.1,
                    "disturbance_jitter_amp": 30.0,  # over PS max: clamped
                    "disturbance_platform_type": "circular"},
        )
        assert dist.noise.enabled is True
        assert dist.noise.gaussian_sigma == 12.0
        assert dist.noise.salt_pepper_density == 0.0  # subtype off
        assert dist.noise.poisson_enabled is True
        assert dist.jitter.amplitude_px == 20.0  # clamped to PS max
        assert dist.platform_motion.type == PlatformMotionType.CIRCULAR

    def test_apply_overrides_bad_platform_falls_back(self, qapp):
        from fsoc_tracker.disturbances.config import (
            DisturbanceConfig,
            PlatformMotionConfig,
            PlatformMotionType,
        )
        from fsoc_tracker.gui.worker import ProcessingWorker
        w = ProcessingWorker()
        dist = DisturbanceConfig(platform_motion=PlatformMotionConfig())
        w._apply_effect_overrides(
            dist, {"platform_motion": True}, 0.5,
            tuning={"disturbance_platform_type": "nonsense"})
        assert dist.platform_motion.type == PlatformMotionType.LINEAR

    def test_apply_overrides_legacy_unchanged(self, qapp):
        from fsoc_tracker.disturbances.config import (
            DisturbanceConfig,
            JitterConfig,
            NoiseConfig,
        )
        from fsoc_tracker.gui.worker import ProcessingWorker
        w = ProcessingWorker()
        dist = DisturbanceConfig(noise=NoiseConfig(), jitter=JitterConfig())
        w._apply_effect_overrides(
            dist, {"noise": True, "jitter": True}, 0.5)
        assert dist.noise.gaussian_sigma == 2.5
        assert dist.noise.salt_pepper_density == 0.025
        assert dist.jitter.amplitude_px == 2.5


class TestBeaconControl:
    """Simulation beacon control: trajectory option, keys, guidance."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_traj_combo_has_user_controlled(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        items = [window._traj_combo.itemText(i)
                 for i in range(window._traj_combo.count())]
        assert "user_controlled" in items
        window.close()

    def test_add_user_controlled_and_drive_with_key(self, qapp):
        import time

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._traj_combo.setCurrentText("user_controlled")
        window._on_add_beacon()
        window._on_start()
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.05)
        eng = window._worker._sim_engine
        uc = [t for t in eng.get_state().targets
              if t.trajectory_type == "user_controlled"]
        assert len(uc) == 1
        window.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_D,
                                       Qt.NoModifier))
        time.sleep(0.4)
        qapp.processEvents()
        aft = [t for t in eng.get_state().targets
               if t.target_id == uc[0].target_id][0]
        assert aft.x > uc[0].x
        window._worker.stop_run()
        window.close()

    def test_move_key_without_uc_beacon_warns(self, qapp):
        import time

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._traj_combo.setCurrentText("circular")
        window._on_add_beacon()
        window._on_start()
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.05)
        window.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_D,
                                       Qt.NoModifier))
        qapp.processEvents()
        assert any("user-controlled beacon" in e.message
                   for e in window._state.events)
        window._worker.stop_run()
        window.close()

    def test_is_beacon_move_key(self, qapp):
        from PySide6.QtCore import Qt

        from fsoc_tracker.gui.main_window import MainWindow
        assert MainWindow._is_beacon_move_key(Qt.Key_W) is True
        assert MainWindow._is_beacon_move_key(Qt.Key_F1) is False
        window = MainWindow()
        window.close()

    def test_add_beacon_default_random_placement(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._traj_combo.setCurrentText("straight_line")
        window._seed_spin.setValue(42)
        window._on_add_beacon()
        window._on_add_beacon()
        wc = window._world_config
        pos = [(b.x0, b.y0, b.z0) for b in wc.beacons]
        assert len(pos) == 2
        # Near centre (inside first-acquisition reach), not identical,
        # reproducible
        for x, y, z in pos:
            assert 985.0 <= x <= 1015.0
            assert 985.0 <= y <= 1015.0
            assert z == 500.0
        assert pos[0][:2] != pos[1][:2]
        window2 = MainWindow()
        window2._world_combo.setCurrentText("New empty world")
        window2._traj_combo.setCurrentText("straight_line")
        window2._seed_spin.setValue(42)
        window2._on_add_beacon()
        pos2 = [(b.x0, b.y0, b.z0) for b in window2._world_config.beacons]
        assert pos2[0] == pos[0]
        window.close()
        window2.close()


class TestWorldSelectionAndEditing:
    """3D selection of beacons/terminal + per-object editing."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_staged_world_mirrored_pre_start(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        assert window._state.targets_all == []
        window._world_combo.setCurrentText("New empty world")
        window._traj_combo.setCurrentText("circular")
        window._on_add_beacon()
        shown = window._state.targets_all
        assert len(shown) == 1
        assert shown[0].trajectory_type == "circular"
        assert shown[0].visible is True
        assert window._state.terminal_a.active is True
        assert window._state.terminal_a.world_x == 1000.0
        window.close()

    def test_terminal_select_and_move_staged(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._setup_world_view.terminal_selected.emit()
        assert window._state.selected_terminal_a is True
        assert window._state.selected_object_id is None
        window._term_x.setValue(1200.0)
        window._on_apply_terminal_edit()
        assert window._world_config.terminal.x == 1200.0
        assert window._state.terminal_a.world_x == 1200.0
        window.close()

    def test_beacon_select_and_edit_staged(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_target_selected(0)
        assert window._state.selected_object_id == 0
        assert window._state.selected_terminal_a is False
        window._edit_traj.setCurrentText("figure_8")
        window._edit_size.setValue(20.0)
        window._on_apply_beacon_edit()
        b = window._world_config.beacons[0]
        assert b.trajectory == "figure_8"
        assert b.size_px == 20.0
        window.close()

    def test_beacon_edit_live_engine(self, qapp):
        import time

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_start()
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.05)
        window._on_target_selected(0)
        window._edit_traj.setCurrentText("circular")
        window._edit_size.setValue(15.0)
        window._on_apply_beacon_edit()
        eng = window._worker._sim_engine
        t = [x for x in eng.get_state().targets if x.target_id == 0][0]
        assert t.trajectory_type == "circular"
        assert t.size_px == 15.0
        window._worker.stop_run()
        window.close()

    def test_remove_clears_selection_ui(self, qapp):
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_target_selected(0)
        window._on_remove_beacon()
        assert window._state.selected_object_id is None
        assert window._state.selected_terminal_a is False
        assert window._state.targets_all == []
        window.close()


class TestMotionTrails:
    """Camera sweep trail + target trails (ported from Astrionics review)."""

    @pytest.fixture
    def qapp(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_trails_populate_and_bounded(self, qapp):
        import time

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._traj_combo.setCurrentText("circular")
        window._on_add_beacon()
        window._on_start()
        for _ in range(15):
            qapp.processEvents()
            time.sleep(0.05)
        s = window._state
        assert len(s.camera_trail) > 0
        assert len(s.camera_trail) <= 120
        assert len(s.targets_all) == 1
        assert len(s.targets_all[0].trail) > 0
        assert len(s.targets_all[0].trail) <= 120
        assert all(len(p) == 3 for p in s.camera_trail)
        window._worker.stop_run()
        window.close()

    def test_trails_cleared_on_restart(self, qapp):
        import time

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_start()
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.05)
        assert len(window._state.camera_trail) > 0
        window._worker.stop_run()
        window._on_start()
        assert len(window._worker._camera_trail) == 0
        assert window._worker._target_trails == {}
        window._worker.stop_run()
        window.close()

    def test_world_view_paints_trails(self, qapp):
        import time

        from PySide6.QtGui import QPixmap

        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_start()
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.05)
        window._navigate_to(0)
        qapp.processEvents()
        wv = window._setup_world_view
        pm = QPixmap(max(wv.width(), 100), max(wv.height(), 100))
        wv.render(pm)  # must not raise with trails present
        window._state.show_trail = False
        wv.render(pm)  # toggle-off path
        window._worker.stop_run()
        window.close()

    def test_show_trail_checkbox_flows(self, qapp):
        from fsoc_tracker.gui.controls import ControlPanel
        panel = ControlPanel()
        assert panel.get_config()["show_trail"] is True
        panel._show_trail.setChecked(False)
        assert panel.get_config()["show_trail"] is False
        panel.close()


class TestPerceptionBackendSwitch:
    """Backend selector builds working detectors with safe fallback."""

    @pytest.fixture
    def qapp(self):
        import sys
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def _worker(self):
        from fsoc_tracker.gui.worker import ProcessingWorker
        return ProcessingWorker()

    def test_ai_backend_loads_trained(self, qapp):
        w = self._worker()
        logs = []
        w.log.connect(lambda lv, m: logs.append((lv, m)))
        from fsoc_tracker.perception.config import PerceptionConfig
        det = w._build_detector({"perception_backend": "ai"},
                                PerceptionConfig.for_video())
        assert det.name == "ai_heatmap_cnn"
        assert det.model.trained is True
        w.close() if hasattr(w, "close") else None

    def test_hybrid_backend_ai_available(self, qapp):
        w = self._worker()
        from fsoc_tracker.perception.config import PerceptionConfig
        det = w._build_detector({"perception_backend": "hybrid"},
                                PerceptionConfig.for_video())
        assert det.ai_available is True
        import numpy as np
        img = np.full((240, 320), 5, dtype=np.uint8)
        yy, xx = np.ogrid[:240, :320]
        img[(xx - 160) ** 2 + (yy - 120) ** 2 <= 25] = 230
        r = det.detect(img, 0.0, 0)
        assert r.status is not None

    def test_unknown_backend_falls_back_classical(self, qapp):
        w = self._worker()
        from fsoc_tracker.perception.config import PerceptionConfig
        det = w._build_detector({"perception_backend": "bogus"},
                                PerceptionConfig())
        assert det.name == "classical_bright_spot"

    def test_default_backend_classical(self, qapp):
        w = self._worker()
        from fsoc_tracker.perception.config import PerceptionConfig
        det = w._build_detector({}, PerceptionConfig())
        assert det.name == "classical_bright_spot"

    def test_apply_toggles_switches_backend_live(self, qapp):
        import time
        from fsoc_tracker.gui.main_window import MainWindow
        window = MainWindow()
        window._world_combo.setCurrentText("New empty world")
        window._on_add_beacon()
        window._on_start()
        for _ in range(8):
            qapp.processEvents()
            time.sleep(0.05)
        assert window._worker._detector.name == "classical_bright_spot"
        window._worker.apply_runtime_toggles(
            {"perception_backend": "hybrid", "kalman_enabled": True,
             "ai_brain_enabled": True})
        qapp.processEvents()
        assert "hybrid" in window._worker._detector.name
        window._worker.apply_runtime_toggles(
            {"perception_backend": "classical", "kalman_enabled": True,
             "ai_brain_enabled": True})
        qapp.processEvents()
        assert window._worker._detector.name == "classical_bright_spot"
        window._worker.stop_run()
        window.close()


class TestVideoEndurance:
    """Long-video robustness: transient read gaps must not kill tracking."""

    @pytest.fixture
    def qapp(self):
        import sys
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def _flaky_worker(self, qapp, fail_runs):
        """Worker in VIDEO mode with scripted read-failure runs.

        fail_runs: list of (start_step, length) frame-index ranges that
        return None, simulating decoder hiccups on long files.
        """
        import numpy as np
        from fsoc_tracker.core.interfaces import FrameSource
        from fsoc_tracker.core.models import ColorModel, Frame, SourceType
        from fsoc_tracker.gui.worker import ProcessingWorker

        img = np.full((480, 640, 3), 5, dtype=np.uint8)
        img[238:243, 318:323] = [230, 230, 230]

        class FlakySource(FrameSource):
            def __init__(self):
                self.n = 0

            def open(self):
                pass

            def is_open(self):
                return True

            def release(self):
                pass

            @property
            def source_id(self):
                return "flaky"

            def read(self):
                for start, length in fail_runs:
                    if start <= self.n < start + length:
                        self.n += 1
                        return None
                self.n += 1
                return Frame(image=img.copy(), width=640, height=480,
                             channels=3, color_model=ColorModel.BGR,
                             source_id="flaky", source_type=SourceType.VIDEO,
                             frame_index=self.n,
                             timestamp_s=self.n / 30.0, nominal_fps=30.0)

        w = ProcessingWorker()
        w.configure({"mode": "video", "video_path": "flaky.mp4"})
        w._init_pipeline()
        w._source = FlakySource()
        w._source.open()
        w._ensure_tracking_pipeline()
        w._running = True
        return w

    def test_transient_gaps_tolerated(self, qapp):
        w = self._flaky_worker(qapp, [(50, 5), (120, 10)])
        for _ in range(200):
            w._run_one_step()
            qapp.processEvents()
        assert w._running is True
        assert w._video_miss_streak == 0
        assert w._frames_processed == 200 - 15
        w.stop_run()
        w.close() if hasattr(w, "close") else None

    def test_sustained_failure_stops_cleanly(self, qapp):
        w = self._flaky_worker(qapp, [(10, 1000)])
        for _ in range(60):
            w._run_one_step()
            qapp.processEvents()
        assert w._running is False
        assert w._frames_processed == 10
        w.close() if hasattr(w, "close") else None


class TestEventLogRollover:
    """Event widget stays live + bounded on long runs."""

    @pytest.fixture
    def qapp(self):
        import sys
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def test_rollover_keeps_displaying(self, qapp):
        from fsoc_tracker.gui.event_log import EventLogPanel
        from fsoc_tracker.gui.state import ApplicationViewState
        panel = EventLogPanel()
        s = ApplicationViewState()
        for i in range(600):
            s.add_event(f"msg-{i}", "INFO")
        panel.update_state(s)
        assert panel._text.document().blockCount() <= 500
        first_blocks = panel._text.toPlainText()
        for i in range(600, 610):
            s.add_event(f"msg-{i}", "INFO")
        panel.update_state(s)
        text = panel._text.toPlainText()
        assert "msg-609" in text
        assert panel._text.document().blockCount() <= 500
        panel.close()
