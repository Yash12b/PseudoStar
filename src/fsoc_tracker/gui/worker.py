"""Processing worker thread.

Runs the perception/tracking/control pipeline in a background thread.
Communicates with the GUI via Qt signals.

Architecture:
    FrameSource → Frame → Perception → Tracking → Control → Camera

All three modes (SIMULATION, VIDEO, LIVE) produce Frame objects through
the same FrameSource interface. The downstream pipeline NEVER branches
on source type for perception/tracking/control logic.

Ground truth is NEVER consumed by the runtime AI layer. Simulation
frames never carry ground truth; offline scoring reads it from an
EvalSink side channel attached to the simulation source.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fsoc_tracker.ai.mission import (
    AIMissionBrain,
    MissionAction,
    MissionObservation,
    ObservationFeatures,
    Situation,
)
from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import Frame
from fsoc_tracker.disturbances.config import DisturbanceConfig
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.gui.state import (
    ApplicationViewState,
    SystemMode,
    SystemState,
    TargetView,
)
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.pipeline.eval import EvalSink
from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.pipeline.sources import (
    EquirectangularFrameSource,
    LiveSource,
    LiveViewportSource,
    VideoSource,
    VirtualSimulationSource,
)
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.tracker import KalmanTracker

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    from PyQt5.QtCore import QThread, Signal  # type: ignore


# Video-decoder recovery policy (VIDEO mode only).
#
# OpenCV's ffmpeg backend goes sticky-EOF after a damaged packet:
# read() returns None forever even though later frames are perfectly
# decodable. Without recovery, one corrupt packet in a 5-minute file
# stops the whole run ("Video ended after N frames" far short of the
# real duration). So on sustained misses the worker re-opens the file
# and seeks past the damage, skipping further ahead with each attempt.
# The run stops only on confirmed end-of-file or exhausted recovery.
_VIDEO_RECOVER_FIRST_AT = 5      # misses before the first recovery attempt
_VIDEO_RECOVER_EVERY = 10        # misses between further attempts
_VIDEO_RECOVER_MAX_ATTEMPTS = 8  # give up recovery after this many failures
_VIDEO_RECOVER_SKIP_FRAMES = 30  # seek this many frames further per attempt
_VIDEO_HARD_STOP_STREAK = 300    # backstop: never spin forever (~10 s)


# Handoff readiness: coarse alignment counts as stable enough for a
# future fine-pointing stage when lock + low uncertainty/residual hold
# for this many consecutive measured frames (all GT-free observables).
_HANDOFF_STREAK_FRAMES = 15
_HANDOFF_UNCERTAINTY_SUM_PX = 20.0
_HANDOFF_RESIDUAL_PX = 5.0


def _sensor_shape_kwargs(shape: str) -> dict:
    """Map a GUI target-shape value to SensorConfig overrides.

    "spot" keeps the legacy soft-Gaussian rendering (config defaults —
    existing behavior unchanged). "square"/"circular" render hard-edged
    spots of the selected geometry.
    """
    shape = (shape or "spot").lower()
    if shape == "square":
        return {"beacon_shape": "square", "beacon_soft_edges": False}
    if shape == "circular":
        return {"beacon_shape": "circular", "beacon_soft_edges": False}
    return {}


class WorkerSignals:
    """Defined on the worker for cross-thread communication."""

    class _Cls:
        state_updated = Signal(object)
        error = Signal(str)
        log = Signal(str, str)

    instance = _Cls()


class ProcessingWorker(QThread):
    """Background processing thread.

    Architecture: FrameSource → Frame → Perception → Tracking → Control

    All three modes produce Frame objects through the same FrameSource
    interface. The pipeline NEVER branches on source type for
    perception/tracking/control logic.
    """

    state_updated = Signal(object)
    error = Signal(str)
    log = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._paused = False
        self._step = False
        self._mode = SystemMode.SIMULATION
        # Consecutive unreadable-video-frames tolerance: a single bad
        # read (common on long/real-world files) must not kill the run;
        # only a sustained failure means EOF/device loss.
        self._video_miss_streak = 0
        self._video_miss_limit = 30
        self._video_recover_attempts = 0
        self._state = ApplicationViewState()
        self._config: dict[str, Any] = {}
        # Runtime ablation toggles (live from GUI; defaults = full system).
        self._ai_enabled = True
        self._use_kalman_tracker = True

        # Unified frame source — ALL modes go through this
        self._source: FrameSource | None = None

        # Simulation-only: engine, sensor, disturbance (for GT eval + scene mgmt)
        self._sim_engine: SimulationEngine | None = None
        self._camera: VirtualCamera | None = None
        self._sensor: VirtualSensorRenderer | None = None
        self._detector: ClassicalBeaconDetector | None = None
        self._tracker: KalmanTracker | None = None
        self._controller: CoarsePointingController | None = None
        self._actuator = CameraActuator()
        # Single tracking loop: worker delegates perception → tracking →
        # search → control → camera to ONE TrackingPipeline built from the
        # components above. Rebuilt on every _init_pipeline.
        self._pipeline: TrackingPipeline | None = None
        self._disturbance: DisturbancePipeline | None = None
        self._last_frame: np.ndarray | None = None
        self._shutdown = False
        self._det_result_prev = None

        self._sim_dt: float = 1.0 / 30.0
        self._frame_index: int = 0
        self._sim_time: float = 0.0
        self._wall_start: float | None = None
        self._frames_processed = 0
        self._eval_sink: EvalSink | None = None
        self._mission_policy = AIMissionBrain()
        self._last_mission_decision = None
        # Trail history (bounded): per-target world positions for the 3D
        # target trails, plus camera boresight aim points for the search
        # sweep trail. Replaces the unused _target_trail/_max_trail stubs.
        from collections import deque as _deque
        self._target_trails: dict[int, _deque] = {}
        self._trail_len = 120
        self._camera_trail: _deque = _deque(maxlen=120)
        self._trail_depth = 550.0

        # Search controller
        from fsoc_tracker.tracking.search import SearchConfig, SearchController
        self._search_controller = SearchController(SearchConfig(
            image_width=640, image_height=480,
        ))
        self._search_active = False
        self._last_tracking_state = "NO_TRACK"

        # Adaptive ROI
        from fsoc_tracker.perception.adaptive_roi import AdaptiveROI, ROIConfig
        self._adaptive_roi = AdaptiveROI(ROIConfig(
            image_width=640, image_height=480,
        ))

        # Failure predictor (trained weights when shipped, else heuristic)
        from fsoc_tracker.ai.failure_predictor import FailurePredictor
        self._failure_predictor = FailurePredictor()
        try:
            self._failure_predictor = FailurePredictor.load(
                "artifacts/models/failure-v1/failure-v1.npz")
            self.log.emit("INFO", "Failure predictor: trained weights loaded")
        except Exception:
            pass
        from collections import deque
        self._failure_window: deque = deque(maxlen=10)

        # Search metrics
        self._search_start_time_s: float | None = None
        self._last_search_strategy: str | None = None

        # Beacon AI engine (GT-free — consumes only TrackingState + intrinsics)
        from fsoc_tracker.ai.beacon_ai import FSocBeaconAI
        self._beacon_ai = FSocBeaconAI()

        # Scorecard tracking
        self._acquisition_time_s: float | None = None
        self._first_detection_s: float | None = None
        self._loss_count: int = 0
        self._total_frames_tracked: int = 0
        self._reacquisition_time_s: float | None = None
        self._last_loss_s: float | None = None
        self._was_tracking: bool = False
        # Handoff readiness (GT-free coarse-stability signal)
        self._handoff_streak: int = 0
        self._handoff_streak_start_s: float = 0.0
        self._handoff_ready: bool = False

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        try:
            self._mode = SystemMode(config.get("mode", "simulation"))
        except ValueError as exc:
            self.error.emit(f"Unsupported input mode: {config.get('mode')}")
            raise ValueError(f"Unsupported input mode: {config.get('mode')}") from exc
        self._state.system_mode = self._mode
        self._state.show_ground_truth = bool(config.get("show_ground_truth", False))
        self._state.show_trail = bool(config.get("show_trail", True))
        self._state.source_info = str(config.get("video_path", "")) if self._mode == SystemMode.VIDEO else self._mode.value
        self._sim_dt = config.get("sim_dt", 1.0 / 30.0)
        self._active_world = None
        self._ai_enabled = bool(config.get("ai_brain_enabled", True))
        self._use_kalman_tracker = bool(config.get("kalman_enabled", True))
        if self._pipeline is not None:
            self._pipeline.set_tracker_enabled(self._use_kalman_tracker)
        try:
            self._mission_policy.prediction_horizon_s = float(
                config.get("prediction_horizon_s", 0.1))
        except (ValueError, TypeError):
            pass

    @property
    def latest_frame(self) -> np.ndarray | None:
        return self._last_frame

    @property
    def view_state(self) -> ApplicationViewState:
        return self._state

    @property
    def comm_engine(self):
        if not hasattr(self, "_comm_engine"):
            from fsoc_tracker.simulation.communication import CommunicationEngine
            self._comm_engine = CommunicationEngine()
        return self._comm_engine

    def inject_beacon(self, x: float, y: float, z: float, trajectory_type: str = "random",
                      trajectory_params: dict | None = None, seed: int = 42,
                      shape: str = "square", size_px: float | None = None,
                      brightness: float | None = None) -> int:
        """Inject a beacon target into the simulation engine.

        Returns the new target id, or -1 when no engine is available.
        Caller-supplied params win; per-type sane defaults fill the rest.
        """
        if self._sim_engine is None:
            return -1
        params = {"x0": x, "y0": y, "z0": z}
        if trajectory_type == "random":
            params["max_step"] = 1.5
            params["segment_duration"] = 0.5
        elif trajectory_type == "straight_line":
            params["vx"] = 0.1
            params["vy"] = 0.02
        elif trajectory_type in ("circular", "figure_8", "sinusoidal",
                                 "spiral"):
            # These types center on cx/cy/cz, not x0/y0/z0.
            params["cx"] = params.pop("x0")
            params["cy"] = params.pop("y0")
            params["cz"] = params.pop("z0")
            if trajectory_type == "circular":
                params["radius"] = 2.0
                params["angular_speed_rad_s"] = 0.3
            elif trajectory_type == "figure_8":
                params["amplitude_x"] = 8.0
                params["amplitude_y"] = 5.0
            elif trajectory_type == "sinusoidal":
                params["amplitude_x"] = 8.0
                params["amplitude_y"] = 5.0
                params["freq_x"] = 0.3
                params["freq_y"] = 0.2
            elif trajectory_type == "spiral":
                params["radius_start"] = 2.0
                params["radius_growth"] = 0.5
                params["angular_speed_rad_s"] = 0.3
        if trajectory_params:
            params.update(trajectory_params)
        if trajectory_type in ("random", "random_walk"):
            params.setdefault("seed", seed)
        if trajectory_params:
            params.update(trajectory_params)
        if trajectory_type in ("random", "random_walk"):
            params.setdefault("seed", seed)
        target = self._sim_engine.add_target(
            trajectory_type=trajectory_type, trajectory_params=params)
        if shape and hasattr(target, "shape"):
            target.shape = str(shape)
        if size_px is not None and hasattr(target, "size_px"):
            target.size_px = float(size_px)
        if brightness is not None and hasattr(target, "brightness"):
            target.brightness = float(brightness)
        self.log.emit("INFO", f"Injected beacon {target.target_id} at ({x:.1f}, {y:.1f}, {z:.1f}) type={trajectory_type}")
        return target.target_id

    def nudge_beacon(self, dx: float, dy: float, dz: float, target_id: int = 0) -> bool:
        """Manually displace a user-controlled beacon (AI-blind challenge).

        The keyboard command never reaches perception/tracking/AI — only
        the resulting rendered pixels do. Returns True if accepted.
        """
        if self._sim_engine is None or self._mode != SystemMode.SIMULATION:
            return False
        ok = self._sim_engine.nudge_target(target_id, dx, dy, dz)
        if ok:
            self.log.emit("INFO", f"Beacon {target_id} nudged by ({dx:+.0f}, {dy:+.0f}, {dz:+.0f})")
        return ok

    def reset_beacon(self, target_id: int = 0) -> bool:
        """Return a user-controlled beacon to its spawn position."""
        if self._sim_engine is None or self._mode != SystemMode.SIMULATION:
            return False
        ok = self._sim_engine.reset_target(target_id)
        if ok:
            self.log.emit("INFO", f"Beacon {target_id} reset to spawn")
        return ok

    def random_beacon_maneuver(self, target_id: int = 0, magnitude_m: float = 15.0) -> bool:
        """Apply a random maneuver to a user-controlled beacon."""
        import math
        import random
        angle = random.uniform(0.0, 2.0 * math.pi)
        return self.nudge_beacon(
            magnitude_m * math.cos(angle), 0.0,
            magnitude_m * math.sin(angle), target_id,
        )

    def _rebase_platform(self) -> None:
        """Re-baseline strapdown carry after manual placement.

        Manual actions teleport the platform AND the camera together;
        without re-baselining, the next frame would apply the teleport
        as a motion delta a second time.
        """
        try:
            sync = getattr(self._source, "sync_platform_baseline", None)
            if callable(sync):
                sync()
        except Exception:
            pass

    def set_terminal_pose(self, x: float, y: float, z: float) -> bool:
        """Place Terminal A (move the virtual camera platform).

        Wired to the 3D world-view "Place Terminal A Here" action.
        Moves the engine platform and the camera together (they are
        bolted to each other), then re-baselines strapdown carry.
        Returns True if applied.
        """
        if self._camera is None:
            return False
        self._camera.state.position_x = float(x)
        self._camera.state.position_y = float(y)
        self._camera.state.position_z = float(z)
        try:
            if self._sim_engine is not None:
                self._sim_engine.set_platform_pose(
                    float(x), float(y), float(z),
                    self._state.terminal_a.yaw_deg,
                    self._state.terminal_a.pitch_deg)
        except Exception:
            pass
        self._rebase_platform()
        self._state.terminal_a.world_x = float(x)
        self._state.terminal_a.world_y = float(y)
        self._state.terminal_a.world_z = float(z)
        self.log.emit("INFO", f"Terminal A placed at ({x:.0f}, {y:.0f}, {z:.0f})")
        return True

    def set_terminal_pose_full(
        self, x: float, y: float, z: float,
        yaw_deg: float | None = None, pitch_deg: float | None = None,
    ) -> bool:
        """Place Terminal A with orientation (position + yaw/pitch).

        Extends set_terminal_pose (which stays for the 3D-view action).
        Orientation applies to the engine platform and the view state.
        """
        if not self.set_terminal_pose(x, y, z):
            return False
        if yaw_deg is not None:
            self._state.terminal_a.yaw_deg = float(yaw_deg)
        if pitch_deg is not None:
            self._state.terminal_a.pitch_deg = float(pitch_deg)
        try:
            if self._sim_engine is not None:
                self._sim_engine.set_platform_pose(
                    float(x), float(y), float(z),
                    self._state.terminal_a.yaw_deg,
                    self._state.terminal_a.pitch_deg)
        except Exception:
            pass
        self._rebase_platform()
        return True

    def send_message_to_beacon(self, payload: str) -> None:
        """Send a message to the connected beacon."""
        dest = "BEACON"
        self._beacon_ai.send_message(dest, payload)
        self.log.emit("INFO", f"Message sent to {dest}: {payload[:50]}...")

        import threading
        def _send_response():
            time.sleep(0.5)
            resp = self._beacon_ai.simulate_beacon_response()
            if resp:
                self.log.emit("INFO", f"Response from {resp.source}: {resp.payload}")
        threading.Thread(target=_send_response, daemon=True).start()

    def start_run(self) -> None:
        self._shutdown = False
        self._running = True
        self._paused = False
        self._frame_index = 0
        self._sim_time = 0.0
        self._prev_ts = None
        self._wall_start = time.perf_counter()
        self._frames_processed = 0
        self._video_miss_streak = 0
        self._video_recover_attempts = 0
        self._target_trails = {}
        self._camera_trail.clear()
        self._state.system_state = SystemState.RUNNING
        self._state.run_state = "PLAYING"

        self._acquisition_time_s = None
        self._first_detection_s = None
        self._loss_count = 0
        self._total_frames_tracked = 0
        self._reacquisition_time_s = None
        self._last_loss_s = None
        self._was_tracking = False
        self._handoff_streak = 0
        self._handoff_ready = False
        self._det_result_prev = None
        self._search_start_time_s = None
        self._last_search_strategy = None
        self._state.scorecard.acquisition_s = None
        self._state.scorecard.rmse_px = None
        self._state.scorecard.loss_percent = None
        self._state.scorecard.reacq_s = None
        if self._eval_sink is not None:
            self._eval_sink.clear()

        self._init_pipeline()
        if not self.isRunning():
            self.start()

    def stop_run(self) -> None:
        self._running = False
        self._paused = False
        self._shutdown = True
        self._state.system_state = SystemState.IDLE
        self._state.run_state = "STOPPED"

        if self._frames_processed > 0:
            try:
                import json

                from fsoc_tracker.benchmark.report import (
                    generate_performance_report,
                    generate_text_summary,
                    verify_performance_report,
                )
                raw_errors = None
                try:
                    if self._pipeline is not None:
                        raw_errors = list(self._pipeline.state.errors)
                except Exception:
                    raw_errors = None
                report_path = generate_performance_report(
                    self._state, output_dir="artifacts/reports",
                    filename=f"report_{int(time.time())}.json",
                    raw_errors=raw_errors,
                )
                with open(report_path, encoding="utf-8") as f:
                    report = json.loads(f.read())
                summary = generate_text_summary(report)
                summary_path = report_path.replace(".json", ".txt")
                with open(summary_path, "w") as f:
                    f.write(summary)
                self.log.emit("INFO", f"Performance report saved: {report_path}")
                if raw_errors:
                    verdict = verify_performance_report(report_path)
                    self.log.emit(
                        "INFO" if verdict["ok"] else "ERROR",
                        f"Report recompute-verify: {'PASS' if verdict['ok'] else 'FAIL'} "
                        f"({verdict.get('frames', 0)} raw frames checked)",
                    )
            except Exception as e:
                self.log.emit("ERROR", f"Failed to generate report: {e}")

        self._release_resources()
        self._emit_state()

    def _world_disturbance_base(self, gui_preset: str) -> Any:
        """Base pipeline disturbance config: world preset, GUI overrides.

        The world's disturbance preset applies unless the live GUI preset
        is explicitly set (non-clear). Numeric world overrides
        (jitter/noise) always apply on top.
        """
        from fsoc_tracker.disturbances.config import (
            DisturbanceMode,
            get_preset_config,
        )
        world = getattr(self, "_active_world", None)
        world_preset = "clear"
        if world is not None:
            world_preset = getattr(world.disturbances, "preset", "clear") or "clear"
        chosen = gui_preset if gui_preset != "clear" else world_preset
        try:
            base = get_preset_config(DisturbanceMode(chosen))
        except (ValueError, KeyError):
            base = get_preset_config(DisturbanceMode.CLEAR)
        if world is not None:
            wd = world.disturbances
            if wd.enabled:
                base.enabled = True
            if wd.jitter_px and wd.jitter_px > 0:
                base.jitter.enabled = True
                base.jitter.amplitude_px = float(wd.jitter_px)
            if wd.noise_sigma and wd.noise_sigma > 0:
                base.noise.enabled = True
                base.noise.gaussian_sigma = float(wd.noise_sigma)
        return base

    def update_disturbance(self, config: dict[str, Any]) -> None:
        """Hot-reload disturbance pipeline from GUI config without restarting."""
        if self._mode != SystemMode.SIMULATION:
            return
        try:
            dist_cfg = self._world_disturbance_base(
                config.get("disturbance_preset", "clear"))
            if "disturbance_enabled" in config:
                dist_cfg.enabled = bool(config["disturbance_enabled"])
            effects = config.get("disturbance_effects", {})
            if effects:
                intensity = config.get("disturbance_intensity", 0.5)
                self._apply_effect_overrides(dist_cfg, effects, intensity,
                                             tuning=config)
            self._disturbance = DisturbancePipeline(dist_cfg)
            if self._source is not None and hasattr(self._source, "_disturbance"):
                self._source._disturbance = self._disturbance
            self.log.emit("INFO", f"Disturbance updated: enabled={dist_cfg.enabled}, "
                          f"preset={config.get('disturbance_preset')}")
        except Exception as e:
            self.log.emit("ERROR", f"Failed to update disturbance: {e}")

    def update_perception(self, config: dict[str, Any]) -> None:
        """Hot-reload perception confidence threshold from GUI."""
        if self._detector is None:
            return
        gui_conf = config.get("confidence_threshold")
        if gui_conf is not None:
            try:
                self._detector._config = self._detector._config.model_copy(
                    update={"min_confidence": float(gui_conf)})
            except (ValueError, TypeError):
                pass

    def pause(self) -> None:
        self._paused = True
        self._state.system_state = SystemState.PAUSED
        self._state.run_state = "PAUSED"
        self._emit_state()

    def resume(self) -> None:
        self._paused = False
        self._state.system_state = SystemState.RUNNING
        self._state.run_state = "PLAYING"
        self._emit_state()

    def step(self) -> None:
        self._step = True
        if self._paused:
            self._paused = False
            self._run_one_step()
            self._paused = True
            self._state.system_state = SystemState.PAUSED

    def _build_detector(self, cfg: dict, perc_cfg: PerceptionConfig):
        """Build the perception backend selected in the GUI.

        Delegates to perception.backends (trained weights or classical
        fallback, never silent random weights).
        """
        from fsoc_tracker.perception.backends import build_backend
        backend = str(cfg.get("perception_backend", "classical")).lower()
        det, _, _ = build_backend(backend, perc_cfg, log=self)
        return det

    def _init_pipeline(self) -> None:
        cfg = self._config
        # Fresh components AND fresh loop state on every (re)init.
        self._pipeline = None
        try:
            # Build perception config: video/live get relaxed thresholds
            # tuned for real-world contrast; simulation keeps stock defaults.
            perc_cfg = PerceptionConfig()
            if self._mode in (SystemMode.VIDEO, SystemMode.LIVE):
                perc_cfg = PerceptionConfig.for_video()
            # Apply GUI confidence threshold override
            gui_conf = cfg.get("confidence_threshold")
            if gui_conf is not None:
                try:
                    perc_cfg = perc_cfg.model_copy(
                        update={"min_confidence": float(gui_conf)})
                except (ValueError, TypeError):
                    pass
            self._detector = self._build_detector(cfg, perc_cfg)
            self._tracker = KalmanTracker()
            self._controller = CoarsePointingController()

            if self._mode == SystemMode.VIDEO:
                path = str(cfg.get("video_path", "")).strip()
                if not path:
                    raise ValueError("A video file must be selected in VIDEO mode")
                video_src = VideoSource(path)
                video_src.open()
                info = video_src.get_info()
                if video_src.nominal_fps:
                    self._sim_dt = 1.0 / video_src.nominal_fps
                self._state.camera.image_width = int(info["width"])
                self._state.camera.image_height = int(info["height"])
                self._state.source_info = f"VIDEO: {path.split('/')[-1]}"

                # Track video metadata for GUI display
                self._video_total_frames = info.get("total_frames", 0)
                self._video_duration_s = info.get("duration_s", 0.0)
                self._video_fps = info.get("fps", None)
                self._video_path = path
                self._video_is_equirectangular = info.get("is_equirectangular", False)

                # Populate state for GUI
                self._state.video_total_frames = self._video_total_frames
                self._state.video_duration_s = self._video_duration_s
                self._state.video_fps = self._video_fps or 0.0
                self._state.video_is_equirectangular = self._video_is_equirectangular
                self._state.video_distance_status = "N/A"

                # Auto-wrap equirectangular/360 video with viewport extractor
                if info.get("is_equirectangular", False):
                    self._init_camera_from_config(cfg)
                    self._source = EquirectangularFrameSource(
                        video_src,
                        self._camera,
                        output_width=cfg.get("camera_width", 640),
                        output_height=cfg.get("camera_height", 480),
                        hfov_deg=cfg.get("hfov", 4.0),
                        vfov_deg=cfg.get("vfov", 3.0),
                    )
                    self._source.open()
                    self.log.emit("INFO", "Equirectangular video detected — viewport extraction active")
                else:
                    self._source = video_src

                self._init_camera_from_config(cfg)
                self.log.emit("INFO", f"Video loaded: {path.split('/')[-1]} "
                              f"({info['width']}x{info['height']} @ {info.get('fps', '?')} FPS, "
                              f"{info.get('total_frames', '?')} frames)")
                return
            if self._mode == SystemMode.LIVE:
                camera_id = int(cfg.get("camera_id", 0))
                self._source = LiveSource(
                    camera_id=camera_id,
                    width=cfg.get("camera_width", 640),
                    height=cfg.get("camera_height", 480),
                )
                try:
                    self._source.open()
                except Exception as e:
                    self._state.source_info = f"LIVE CAMERA {camera_id} - ERROR"
                    self._state.live_camera_status = "error"
                    self._state.live_error_message = str(e)
                    self.error.emit(f"Cannot open camera {camera_id}: {e}")
                    return
                info = self._source.get_info()
                self._state.source_info = f"LIVE CAMERA {info['camera_id']}"
                # Software virtual viewport: the fixed live feed is steered
                # by pan/tilt commands (no physical PTZ hardware assumed).
                self._source = LiveViewportSource(
                    self._source,
                    output_width=cfg.get("camera_width", 640),
                    output_height=cfg.get("camera_height", 480),
                    hfov_deg=cfg.get("hfov", 4.0),
                    vfov_deg=cfg.get("vfov", 3.0),
                )
                self._state.camera.image_width = int(info["width"])
                self._state.camera.image_height = int(info["height"])
                self._state.live_camera_id = info["camera_id"]
                self._state.live_camera_name = f"Camera {info['camera_id']}"
                self._state.live_resolution = f"{info['width']}x{info['height']}"
                self._state.live_fps = info.get("actual_fps", 0.0)
                self._state.live_camera_status = "available"
                self._state.live_distance_status = "UNAVAILABLE"  # Monocular, no depth
                self._init_camera_from_config(cfg)
                return

            # --- SIMULATION MODE (generic world, no scene presets) ---
            wc = WorldConfig(
                width=cfg.get("world_width", 2000.0),
                height=cfg.get("world_height", 2000.0),
                random_seed=cfg.get("seed", 42),
            )
            self._sim_engine = SimulationEngine(wc)

            scenario = cfg.get("scenario", None)
            if scenario is None:
                # Legacy/config-key fallback: single beacon from trajectory
                # keys (used by tests and headless harnesses).
                from fsoc_tracker.simulation.world_builder import (
                    add_beacon,
                    create_world,
                    set_primary_beacon,
                )
                scenario = create_world(
                    width=wc.width, height=wc.height, depth=wc.depth,
                    seed=cfg.get("seed", 42), name="Default World")
                traj_type = cfg.get("trajectory", "straight_line")
                trajectory_params = dict(cfg.get("trajectory_params", {}))
                if not trajectory_params and traj_type == "straight_line":
                    trajectory_params = {
                        "x0": wc.width / 2.0,
                        "y0": wc.height / 2.0,
                        "z0": 500.0,
                        "vx": 0.3,
                        "vy": 0.2,
                    }
                if not trajectory_params and traj_type == "user_controlled":
                    trajectory_params = {
                        "x0": wc.width / 2.0,
                        "y0": wc.height / 2.0,
                        "z0": 500.0,
                    }
                beacon = add_beacon(
                    scenario, wc.width / 2.0, wc.height / 2.0, 500.0,
                    motion=traj_type, motion_params=trajectory_params,
                    seed=cfg.get("seed", 42),
                    size_px=cfg.get("target_size", 10.0))
                set_primary_beacon(scenario, beacon.beacon_id)
            self._active_world = scenario
            self._sim_engine.load_scenario(scenario)
            self.log.emit("INFO", f"Loaded world: {scenario.name} "
                                  f"({len(scenario.beacons)} beacons)")
            # Initialize designation from the world's primary beacon.
            if scenario.primary_beacon_id is not None:
                self._state.designated_beacon_id = scenario.primary_beacon_id
                self._state.tracked_target_id = scenario.primary_beacon_id

            self._init_camera_from_config(cfg)

            sc = SensorConfig(
                width=cfg.get("camera_width", 640),
                height=cfg.get("camera_height", 480),
                beacon_default_size_px=cfg.get("target_size", 10.0),
                **_sensor_shape_kwargs(str(cfg.get("target_shape", "spot"))),
            )
            self._sensor = VirtualSensorRenderer(sc)

            dist_cfg = self._world_disturbance_base(
                cfg.get("disturbance_preset", "clear"))
            if "disturbance_enabled" in cfg:
                dist_cfg.enabled = bool(cfg["disturbance_enabled"])
            # Apply custom effect overrides from GUI checkboxes
            effects = cfg.get("disturbance_effects", {})
            if effects:
                intensity = cfg.get("disturbance_intensity", 0.5)
                self._apply_effect_overrides(dist_cfg, effects, intensity,
                                             tuning=cfg)
            self._disturbance = DisturbancePipeline(dist_cfg)

            # Create VirtualSimulationSource with an EvalSink side channel.
            # Frames never carry ground truth; scoring reads the sink.
            self._eval_sink = EvalSink()
            self._source = VirtualSimulationSource(
                sim_engine=self._sim_engine,
                camera=self._camera,
                sensor=self._sensor,
                disturbance=self._disturbance,
                dt=self._sim_dt,
                eval_sink=self._eval_sink,
            )
            self._source.open()

            self._state.perception.backend = "classical_bright_spot"
            self._state.target.trajectory_type = traj_type

            self.log.emit("INFO", "Pipeline initialized")
        except Exception as e:
            self.error.emit(str(e))
            self.log.emit("ERROR", f"Init failed: {e}")

    def _init_camera_from_config(self, cfg: dict[str, Any]) -> None:
        """Initialize camera and search controller from config (shared by all modes).

        In simulation mode the initial pose comes from the world's
        Terminal A configuration — never auto-aimed at any beacon.
        """
        terminal = None
        world = getattr(self, "_active_world", None)
        if world is not None:
            terminal = world.terminal
        cam_state = CameraState(
            horizontal_fov_deg=cfg.get("hfov", 4.0),
            vertical_fov_deg=cfg.get("vfov", 3.0),
            width=cfg.get("camera_width", 640),
            height=cfg.get("camera_height", 480),
            max_pan_speed_deg_s=cfg.get("max_pan_rate", 5.0),
            max_tilt_speed_deg_s=cfg.get("max_tilt_rate", 5.0),
            position_x=terminal.x if terminal is not None else 1000.0,
            position_y=terminal.y if terminal is not None else 1000.0,
            position_z=terminal.z if terminal is not None else 50.0,
            pan_deg=terminal.yaw_deg if terminal is not None else 0.0,
            tilt_deg=terminal.pitch_deg if terminal is not None else 0.0,
        )
        self._camera = VirtualCamera(cam_state)

        self._state.camera.image_width = cfg.get("camera_width", 640)
        self._state.camera.image_height = cfg.get("camera_height", 480)
        self._state.camera.hfov_deg = cfg.get("hfov", 4.0)
        self._state.camera.vfov_deg = cfg.get("vfov", 3.0)

        from fsoc_tracker.tracking.search import SearchConfig, SearchController
        self._search_controller = SearchController(SearchConfig(
            image_width=cfg.get("camera_width", 640),
            image_height=cfg.get("camera_height", 480),
        ))
        self._search_active = False

    def _apply_effect_overrides(
        self,
        cfg: DisturbanceConfig,
        effects: dict[str, bool],
        intensity: float,
        tuning: dict | None = None,
    ) -> None:
        """Apply individual effect toggles and intensity scaling.

        The optional tuning dict carries PS numeric controls from the GUI
        (noise sigma/density, jitter amplitude, platform type). Values are
        clamped to the PS maxima (sigma/jitter/platform amplitude ≤ 20).
        Absent keys fall back to the legacy intensity-scaled defaults.
        """

        def scale(val: float, factor: float) -> float:
            return val * factor

        def clamp(v: float, lo: float, hi: float) -> float:
            try:
                return max(lo, min(hi, float(v)))
            except (TypeError, ValueError):
                return lo

        f = intensity  # 0..1
        tuning = tuning or {}

        cfg.noise.enabled = effects.get("noise", False)
        if cfg.noise.enabled:
            use_gauss = effects.get("noise_gaussian", True)
            use_sp = effects.get("noise_salt_pepper", True)
            use_pois = effects.get("noise_poisson", True)
            if "disturbance_noise_sigma" in tuning:
                # Explicit per-type user selection (PS: one or more).
                cfg.noise.gaussian_sigma = (
                    clamp(tuning["disturbance_noise_sigma"], 0.0, 20.0)
                    if use_gauss else 0.0)
                cfg.noise.salt_pepper_density = (
                    clamp(tuning.get("disturbance_noise_density", 0.05), 0.0, 1.0)
                    if use_sp else 0.0)
                cfg.noise.poisson_enabled = bool(use_pois)
            else:
                # Legacy: enabling on a clear base must do something
                # (scaling 0.0 would be a silent no-op).
                cfg.noise.gaussian_sigma = scale(cfg.noise.gaussian_sigma or 5.0, f)
                cfg.noise.salt_pepper_density = scale(
                    cfg.noise.salt_pepper_density or 0.05, f)

        cfg.atmosphere.enabled = any(effects.get(k, False) for k in ("fog", "haze", "rain", "low_light"))
        if effects.get("fog", False):
            cfg.atmosphere.fog_strength = scale(cfg.atmosphere.fog_strength or 0.3, f)
        if effects.get("haze", False):
            cfg.atmosphere.haze_strength = scale(cfg.atmosphere.haze_strength or 0.2, f)
        if effects.get("rain", False):
            cfg.atmosphere.rain_density = scale(cfg.atmosphere.rain_density or 0.3, f)
        if effects.get("low_light", False):
            cfg.atmosphere.low_light_factor = max(0.1, scale(cfg.atmosphere.low_light_factor or 0.7, f))

        cfg.jitter.enabled = effects.get("jitter", False)
        if cfg.jitter.enabled:
            if "disturbance_jitter_amp" in tuning:
                cfg.jitter.amplitude_px = clamp(
                    tuning["disturbance_jitter_amp"], 0.0, 20.0)
            else:
                cfg.jitter.amplitude_px = min(
                    20.0, scale(cfg.jitter.amplitude_px or 5.0, f))

        cfg.platform_motion.enabled = effects.get("platform_motion", False)
        if cfg.platform_motion.enabled:
            from fsoc_tracker.disturbances.config import PlatformMotionType
            try:
                cfg.platform_motion.type = PlatformMotionType(
                    str(tuning.get("disturbance_platform_type",
                                   cfg.platform_motion.type.value)).lower())
            except ValueError:
                cfg.platform_motion.type = PlatformMotionType.LINEAR
            cfg.platform_motion.amplitude_x_px = min(
                20.0, scale(cfg.platform_motion.amplitude_x_px or 5.0, f))
            cfg.platform_motion.amplitude_y_px = min(
                20.0, scale(cfg.platform_motion.amplitude_y_px or 5.0, f))

        cfg.turbulence.enabled = effects.get("turbulence", False)
        if cfg.turbulence.enabled:
            cfg.turbulence.strength = scale(cfg.turbulence.strength or 1.0, f)

        cfg.blur.enabled = effects.get("blur", False)
        if cfg.blur.enabled:
            cfg.blur.sigma = scale(cfg.blur.sigma or 0.5, f)

        cfg.motion_blur.enabled = effects.get("motion_blur", False)
        if cfg.motion_blur.enabled:
            cfg.motion_blur.kernel_size = max(
                3, int(scale(cfg.motion_blur.kernel_size or 15, f)))

        cfg.brightness_contrast.enabled = effects.get("brightness_contrast", False)
        if cfg.brightness_contrast.enabled:
            cfg.brightness_contrast.brightness = max(0.1, 1.0 - scale(0.5, f))

        cfg.target_disappearance.enabled = effects.get("target_disappearance", False)
        if cfg.target_disappearance.enabled:
            cfg.target_disappearance.probability_per_frame = scale(0.02, f)

        cfg.distractors.enabled = effects.get("distractors", False)

    def seek_video(self, frame_index: int) -> bool:
        """Seek the video source to a specific frame. Only works in VIDEO mode."""
        if self._mode != SystemMode.VIDEO or self._source is None:
            return False
        # Walk up equirectangular wrapper if present
        source = self._source
        if hasattr(source, '_source'):
            source = source._source
        if hasattr(source, 'seek'):
            success = source.seek(frame_index)
            if success:
                self._frame_index = frame_index
                self.log.emit("INFO", f"Seeked to frame {frame_index}")
            return success
        return False

    def _ensure_tracking_pipeline(self) -> TrackingPipeline | None:
        """Build (once) the single loop from this worker's components.

        The pipeline shares the worker's detector/tracker/controller/
        search/ROI objects, so tunings stay identical and HUD code keeps
        reading live controller state. The pipeline owns perception →
        tracking → search → control → camera application; the worker
        keeps view-state, AI-brain HUD, link/comm, and scorecard.
        """
        if self._source is None:
            return None
        if (self._detector is None or self._tracker is None
                or self._controller is None):
            return None
        if self._pipeline is None:
            self._pipeline = TrackingPipeline(
                perception=self._detector,
                tracker=self._tracker,
                controller=self._controller,
                actuator=self._actuator,
                search_controller=self._search_controller,
                adaptive_roi=self._adaptive_roi,
                eval_sink=self._eval_sink,
            )
            self._pipeline.set_source(self._source)
            self._pipeline.set_tracker_enabled(self._use_kalman_tracker)
        return self._pipeline

    def apply_runtime_toggles(self, config: dict[str, Any]) -> None:
        """Apply live ablation toggles (Kalman tracker, AI brain) and
        perception backend switches.

        Safe mid-run: no stateful component is rebuilt except the
        stateless detector. Disabling Kalman resets it on re-enable
        for a clean handoff.
        """
        self._ai_enabled = bool(config.get("ai_brain_enabled", True))
        use_kalman = bool(config.get("kalman_enabled", True))
        if use_kalman != self._use_kalman_tracker and use_kalman:
            self._use_kalman_tracker = True
            if self._pipeline is not None:
                self._pipeline.set_tracker_enabled(True)
            try:
                self._tracker.reset(self._sim_time)
            except Exception:
                pass
            self.log.emit("INFO", "Kalman tracker re-enabled (reset)")
        else:
            self._use_kalman_tracker = use_kalman
            if self._pipeline is not None:
                self._pipeline.set_tracker_enabled(use_kalman)
            if not use_kalman:
                self.log.emit("INFO", "Kalman tracker bypassed: raw detections")
        if not self._ai_enabled:
            try:
                self._controller.set_lead_enabled(False)
            except Exception:
                pass
        # Live backend switch (stateless detector: safe mid-run).
        backend = str(config.get("perception_backend", "classical")).lower()
        current = str(getattr(self._detector, "name", "classical")).lower()
        current_backend = (
            "ai" if current.startswith("ai_")
            else "hybrid" if current.startswith("hybrid")
            else "classical")
        if backend != current_backend:
            from fsoc_tracker.perception.config import PerceptionConfig as _PC
            base = getattr(self._detector, "_config", None)
            try:
                perc_cfg = base.model_copy() if base is not None else _PC()
            except Exception:
                perc_cfg = _PC()
            self._detector = self._build_detector(
                {"perception_backend": backend}, perc_cfg)
            if self._pipeline is not None:
                try:
                    self._pipeline.set_perception(self._detector)
                except Exception:
                    pass
            self.log.emit("INFO", f"Perception backend -> {backend}")

    def _release_resources(self) -> None:
        if self._source is not None:
            self._source.release()
            self._source = None

    def _run_one_step(self) -> None:
        """Read one Frame from the source and process it."""
        if self._source is None:
            return

        # Handle pending world changes (simulation only)
        if self._mode == SystemMode.SIMULATION:
            if (hasattr(self._state, '_pending_world_config')
                    and self._state._pending_world_config is not None):
                self._handle_world_change()

        try:
            viewport = self._source
            if isinstance(viewport, LiveViewportSource) and self._camera is not None:
                viewport.set_viewport_center(
                    self._camera.state.pan_deg, self._camera.state.tilt_deg)
            frame = self._source.read()
            if frame is None:
                if self._mode == SystemMode.VIDEO:
                    self._handle_video_miss()
                return
            self._video_miss_streak = 0
            self._video_recover_attempts = 0
            self._run_frame(frame)
        except Exception as e:
            self.error.emit(str(e))
            self.log.emit("ERROR", f"Frame read failed: {e}")

    def _video_source_total(self) -> int:
        """Container-reported frame total, 0 when unknown/unreliable.

        Named to avoid the pre-existing ``_video_total_frames`` int
        attribute set at init (GUI display metadata).
        """
        src = self._source
        if hasattr(src, "_source"):
            src = src._source  # unwrap equirectangular viewport wrapper
        try:
            total = src.frame_count
            total = int(total()) if callable(total) else int(total)
        except Exception:
            return 0
        return max(0, total or 0)

    def _video_position_str(self) -> str:
        """Best-effort '(decoder pos X / total Y)' diagnostic suffix."""
        src = self._source
        if hasattr(src, "_source"):
            src = src._source
        pos = None
        try:
            getter = getattr(src, "position_frames", None)
            pos = int(getter()) if callable(getter) else (
                int(getter) if getter is not None else None)
        except Exception:
            pos = None
        total = self._video_source_total()
        if pos is None and total <= 0:
            return ""
        return f" (decoder pos {pos if pos is not None else '?'} / total {total if total > 0 else '?'})"

    def _handle_video_miss(self) -> None:
        """Handle one unreadable video frame.

        Brief gaps are tolerated silently. Sustained misses trigger
        decoder recovery (re-open + seek past the damage) when the
        source supports it. The run stops only on confirmed
        end-of-file, exhausted recovery, or the hard-stop backstop —
        never on a single corrupt packet mid-file.
        """
        self._video_miss_streak += 1
        streak = self._video_miss_streak
        if streak == 1:
            self.log.emit("WARNING", "Video frame unreadable; tolerating brief gaps")
        # _run_frame leaves _frame_index one past the last processed frame
        # (fi, then +1), so it doubles as the next-wanted resume point.
        resume_at = max(self._frame_index, 0)
        total = self._video_source_total()
        if total > 0 and resume_at >= total:
            self.log.emit(
                "INFO",
                f"Video ended after {self._frames_processed} frames ({total} total)",
            )
            self.stop_run()
            return
        recover = getattr(self._source, "reopen_and_seek", None)
        if callable(recover):
            if (streak >= _VIDEO_RECOVER_FIRST_AT
                    and (streak - _VIDEO_RECOVER_FIRST_AT) % _VIDEO_RECOVER_EVERY == 0
                    and self._video_recover_attempts < _VIDEO_RECOVER_MAX_ATTEMPTS):
                target = resume_at + self._video_recover_attempts * _VIDEO_RECOVER_SKIP_FRAMES
                if total > 0 and target >= total:
                    self.log.emit(
                        "INFO",
                        f"Video ended after {self._frames_processed} frames ({total} total)",
                    )
                    self.stop_run()
                    return
                frame = None
                try:
                    if bool(recover(int(target))):
                        frame = self._source.read()
                except Exception as e:
                    self.log.emit("WARNING", f"Video recovery attempt failed: {e}")
                if frame is not None:
                    skipped = int(target) - int(resume_at)
                    self._video_miss_streak = 0
                    self._video_recover_attempts = 0
                    if skipped > 0:
                        self.log.emit(
                            "WARNING",
                            f"Video decoder recovered at frame {target} "
                            f"(skipped {skipped} unreadable frames)",
                        )
                    else:
                        self.log.emit("INFO", f"Video decoder recovered at frame {target}")
                    self._run_frame(frame)
                    return
                self._video_recover_attempts += 1
                self.log.emit(
                    "WARNING",
                    f"Video decoder stall at frame {resume_at}; "
                    f"recovery attempt {self._video_recover_attempts} failed",
                )
                if self._video_recover_attempts >= _VIDEO_RECOVER_MAX_ATTEMPTS:
                    pos = self._video_position_str()
                    self.log.emit(
                        "ERROR",
                        f"Video decoder stalled at frame {resume_at}{pos}; giving up — "
                        "file may be truncated or damaged past this point",
                    )
                    self.stop_run()
                    return
        elif streak > self._video_miss_limit:
            self.log.emit("INFO", f"Video ended after {self._frames_processed} frames")
            self.stop_run()
            return
        if streak > _VIDEO_HARD_STOP_STREAK:
            pos = self._video_position_str()
            self.log.emit(
                "ERROR",
                f"Video stalled for {streak} frames at {resume_at}{pos}; stopping",
            )
            self.stop_run()

    def _handle_world_change(self) -> None:
        """Load a pending generic world config live (simulation only)."""
        world = self._state._pending_world_config
        self._state._pending_world_config = None
        if self._sim_engine is None or world is None:
            return
        try:
            self._active_world = world
            self._sim_engine.load_scenario(world)
            self.log.emit("INFO", f"Loaded world: {world.name}")
            if self._disturbance is not None:
                base = self._world_disturbance_base("clear")
                base.enabled = self._disturbance.config.enabled
                self._disturbance.set_config(base)
            self._sim_time = 0.0
            self._frame_index = 0
            self._frames_processed = 0
            self._target_trails = {}
            self._camera_trail.clear()
            self._tracker.reset(0.0)
            self._search_active = False
            self._failure_window.clear()
            if self._controller is not None:
                self._controller.reset()
            ws = self._sim_engine.get_state()
            active = ws.get_active_targets()
            if active:
                designated = world.primary_beacon_id
                if designated is None or not any(
                        t.target_id == designated for t in active):
                    designated = active[0].target_id
                self._state.designated_beacon_id = designated
                self._state.tracked_target_id = designated
        except Exception as e:
            self.error.emit(f"Failed to load world: {e}")

    def _run_frame(self, frame: Frame) -> None:
        """Unified processing pipeline — consumes a generic Frame.

        ALL inputs are observable quantities. Ground truth (if present in
        frame.metadata) is used ONLY for benchmark scoring, never by
        the AI, controller, or search subsystems.
        """
        t_start = time.perf_counter()

        image = frame.image
        ts = frame.timestamp_s
        fi = frame.frame_index
        dt = ts - self._sim_time if self._sim_time > 0 else self._sim_dt
        # Scoring truth comes from the EvalSink side channel keyed by
        # frame index — never from frame metadata (Frames carry none).
        gt = (self._eval_sink.primary_for(frame.frame_index)
              if getattr(self, "_eval_sink", None) is not None else None)

        self._sim_time = ts
        self._frame_index = fi
        self._sim_dt = dt

        try:
            # --- Single loop: perception → tracking → search → control
            # --- (TrackingPipeline owns the shared components)
            pipeline = self._ensure_tracking_pipeline()
            if pipeline is None:
                return
            result = pipeline.process_frame(frame)
            det_result = result.perception
            trk_state = result.tracking
            camera_cmd = result.command
            if det_result is None or trk_state is None:
                return
            dt = result.dt
            self._sim_dt = dt
            tracking_state = trk_state.state.name
            error_x, error_y, error_px = (
                result.error_x, result.error_y, result.error_px)

            # Apply the command to the virtual mount when the source has
            # no camera of its own (live-mode viewport steering). Sim
            # sources apply inside the pipeline through the actuator.
            if (camera_cmd is not None and self._camera is not None
                    and getattr(self._source, "camera", None) is None):
                self._actuator.apply_command(self._camera, camera_cmd, dt)

            # Mirror search activity for HUD/metrics. The pipeline owns
            # the shared SearchController; ROI resets on reacquire.
            if result.search_reacquired:
                self._adaptive_roi.reset()
            if result.search_strategy:
                self._last_search_strategy = result.search_strategy
            self._search_active = bool(result.search_active)
            if self._search_active and self._search_start_time_s is None:
                self._search_start_time_s = ts
            try:
                search_region = self._search_controller.get_search_region()
            except Exception:
                search_region = None

            self._last_frame = image.copy()

            # --- Beacon AI (GT-free: consumes TrackingState + intrinsics only) ---
            if self._beacon_ai is not None and self._camera is not None:
                detected = bool(
                    det_result.primary_detection and det_result.primary_detection.detected
                )
                conf = det_result.primary_detection.confidence if det_result.primary_detection else 0.0

                ai_state = self._beacon_ai.update(
                    dt=dt,
                    tracking_state=tracking_state,
                    detected=detected,
                    estimated_x=trk_state.estimated_x,
                    estimated_y=trk_state.estimated_y,
                    velocity_x=trk_state.velocity_x,
                    velocity_y=trk_state.velocity_y,
                    uncertainty_x=trk_state.uncertainty_x,
                    uncertainty_y=trk_state.uncertainty_y,
                    confidence=conf,
                    time_since_detection=trk_state.time_since_last_detection_s,
                    intrinsics=self._camera.intrinsics,
                    image=image,
                )
                self._state.beacon_connected = ai_state["connected"]
                self._state.atmospheric_disturbance = ai_state["atmospheric"]

            # --- Scoring errors come from the pipeline (EvalSink side
            # --- channel); the view layer only displays them.

            # --- Update view state ---
            self._update_state_from_pipeline(
                image, det_result, trk_state, gt,
                camera_cmd, error_x, error_y, error_px,
                search_region=search_region,
                search_active=self._search_active,
                processing_ms=(time.perf_counter() - t_start) * 1000.0,
                fi=fi,
                ts=ts,
            )

            self._sim_time += dt
            self._frame_index += 1

        except Exception as e:
            self.error.emit(str(e))
            self.log.emit("ERROR", f"Frame failed: {e}")

    def _apply_brain_gating(self, decision) -> None:
        """Turn mission-brain decisions into behavior (safety-gated).

        Currently: lead-angle compensation engages only when the brain
        explicitly calls for predictive tracking of a fast target with
        confidence and safety approval. All other situations keep pure
        estimate tracking.
        """
        try:
            self._controller.set_lead_enabled(bool(
                decision.action == MissionAction.TRACK_PREDICTIVE
                and decision.situation == Situation.FAST_TARGET_MOTION
                and decision.confidence > 0.5
                and decision.safety.approved
            ))
        except Exception:
            pass

    def _update_state_from_pipeline(
        self, image, det_result, trk_state, gt, camera_cmd,
        error_x, error_y, error_px, processing_ms: float = 0.0,
        search_region=None, search_active: bool = False,
        fi: int = 0, ts: float = 0.0,
    ) -> None:
        s = self._state
        if self._wall_start is None:
            self._wall_start = time.perf_counter() - max(self._sim_dt, 1e-6)
        self._frames_processed += 1
        wall_time = (
            time.perf_counter() - self._wall_start
            if self._wall_start is not None else 0.0
        )
        s.elapsed_s = self._sim_time

        s.camera.pan_deg = self._camera.state.pan_deg if self._camera else 0.0
        s.camera.tilt_deg = self._camera.state.tilt_deg if self._camera else 0.0
        s.camera.fps = 1.0 / max(self._sim_dt, 1e-6)
        s.camera.timestamp_s = self._sim_time
        s.performance.processing_ms = processing_ms
        s.performance.frames_processed = self._frames_processed
        s.performance.wall_time_s = wall_time
        s.performance.pipeline_fps = (
            self._frames_processed / wall_time if wall_time > 0 else 0.0
        )
        s.performance.perception_ms = det_result.processing_time_ms
        s.performance.tracking_ms = 0.0
        s.performance.control_ms = 0.0

        # Video mode metadata (populated each frame)
        if self._mode == SystemMode.VIDEO:
            s.video_current_frame = fi
            s.video_timestamp_s = ts
            s.video_processing_fps = s.performance.pipeline_fps

        # Live camera metadata (populated each frame)
        if self._mode == SystemMode.LIVE and self._source:
            source_info = self._source.get_info()
            s.live_fps = source_info.get("actual_fps", 0.0)
            s.live_camera_status = "available" if self._source.is_open() else "unavailable"
            s.live_resolution = f"{source_info.get('width', 0)}x{source_info.get('height', 0)}"
            s.live_distance_status = "UNAVAILABLE"  # Monocular, no depth

        s.perception.detected = det_result.detected
        s.perception.confidence = det_result.primary_detection.confidence if det_result.primary_detection else 0.0
        s.perception.detection_x = det_result.primary_detection.center_x if det_result.primary_detection else 0.0
        s.perception.detection_y = det_result.primary_detection.center_y if det_result.primary_detection else 0.0
        s.perception.processing_ms = det_result.processing_time_ms
        s.perception.num_candidates = det_result.num_candidates

        s.tracking.state = trk_state.state.name
        s.tracking.locked = trk_state.locked
        s.tracking.estimated_x = trk_state.estimated_x
        s.tracking.estimated_y = trk_state.estimated_y
        s.tracking.velocity_x = trk_state.velocity_x
        s.tracking.velocity_y = trk_state.velocity_y
        s.tracking.uncertainty_x = trk_state.uncertainty_x
        s.tracking.uncertainty_y = trk_state.uncertainty_y
        s.tracking.residual = trk_state.residual_magnitude
        s.tracking.prediction_only = trk_state.prediction_only
        s.tracking.time_since_detection_s = trk_state.time_since_last_detection_s

        primary = det_result.primary_detection
        mission_features = ObservationFeatures(
            timestamp_s=self._sim_time,
            detected=bool(primary and primary.detected),
            confidence=primary.confidence if primary else 0.0,
            residual_px=trk_state.residual_magnitude,
            uncertainty_x_px=trk_state.uncertainty_x,
            uncertainty_y_px=trk_state.uncertainty_y,
            velocity_x_px_s=trk_state.velocity_x,
            velocity_y_px_s=trk_state.velocity_y,
            distance_from_center_px=0.0,
            time_since_detection_s=trk_state.time_since_last_detection_s,
            latency_ms=processing_ms,
            source_fps=s.camera.fps,
            processing_fps=s.performance.pipeline_fps,
            candidate_count=det_result.num_candidates,
        )
        if self._ai_enabled:
            self._run_brain_step(mission_features, processing_ms)
        else:
            ai = s.ai_state
            ai.situation = "DISABLED"
            ai.action = "track"
            ai.confidence = 0.0
            ai.explanation = "AI brain disabled by operator (ablation)"
            ai.fallback_active = True
        self._update_handoff(trk_state)
        self._update_view_tail(search_active, camera_cmd, gt,
                               error_x, error_y, error_px)

    def _run_brain_step(
        self,
        mission_features: ObservationFeatures,
        processing_ms: float,
    ) -> None:
        """Mission-brain decision + safety gating + failure risk.

        Split out of _update_state_from_pipeline so the AI layer can be
        bypassed live (ablation) without touching the track/control path.
        """
        s = self._state
        self._last_mission_decision = self._mission_policy.decide(
            MissionObservation(
                features=mission_features,
                camera_pan_deg=s.camera.pan_deg,
                camera_tilt_deg=s.camera.tilt_deg,
            )
        )
        decision = self._last_mission_decision
        ai = s.ai_state
        ai.situation = decision.situation.value
        ai.action = decision.action.value
        ai.confidence = decision.confidence
        ai.explanation = decision.reason
        if decision.prediction is not None:
            ai.prediction_dx = decision.prediction.mean_x_px
            ai.prediction_dy = decision.prediction.mean_y_px
            ai.prediction_uncertainty_x = decision.prediction.uncertainty_x_px
            ai.prediction_uncertainty_y = decision.prediction.uncertainty_y_px
            ai.prediction_horizon_s = decision.prediction.horizon_s
        ai.fallback_active = not decision.safety.approved
        self._apply_brain_gating(decision)

        # --- Failure risk from the trained predictor (sliding window) ---
        try:
            self._failure_window.append(mission_features)
            if (self._failure_predictor.trained
                    and len(self._failure_window) >= 10):
                ai.failure_risk = float(self._failure_predictor.predict(
                    list(self._failure_window)).risk_score)
                # Preemptive widening: act on risk before misses accumulate.
                self._adaptive_roi.expand_for_risk(ai.failure_risk)
        except Exception:
            pass

    def _update_handoff(self, trk_state: Any) -> None:
        """GT-free coarse-stability signal for a fine-pointing stage.

        Handoff becomes READY after 15 consecutive measured frames that
        are TRACKING + locked + low uncertainty + low residual. Any
        unstable frame clears it (re-arming logs again on recovery).
        """
        fresh = (trk_state.time_since_last_detection_s
                 <= max(self._sim_dt, 1e-6) * 1.5)
        stable = (
            trk_state.state.name == "TRACKING"
            and bool(trk_state.locked)
            and not bool(trk_state.prediction_only)
            and fresh
            and (trk_state.uncertainty_x + trk_state.uncertainty_y)
            < _HANDOFF_UNCERTAINTY_SUM_PX
            and trk_state.residual_magnitude < _HANDOFF_RESIDUAL_PX
        )
        if stable:
            if self._handoff_streak == 0:
                self._handoff_streak_start_s = self._sim_time
            self._handoff_streak += 1
        else:
            self._handoff_streak = 0
            self._handoff_ready = False
        if (self._handoff_streak >= _HANDOFF_STREAK_FRAMES
                and not self._handoff_ready):
            self._handoff_ready = True
            self.log.emit(
                "INFO",
                "HANDOFF READY: coarse alignment stable — "
                "suitable for fine-pointing stage",
            )
        self._state.handoff_ready = self._handoff_ready
        self._state.handoff_stable_s = (
            (self._sim_time - self._handoff_streak_start_s)
            if self._handoff_streak else 0.0
        )

    def _update_view_tail(
        self,
        search_active: bool,
        camera_cmd: Any,
        gt: Any,
        error_x: float | None,
        error_y: float | None,
        error_px: float | None,
    ) -> None:
        """Non-AI view-state tail: ROI, search HUD, targets, trails,
        terminal, disturbance, scorecard, emit. Runs every frame
        regardless of the AI ablation toggle."""
        s = self._state
        ai = s.ai_state

        # --- ROI + model provenance straight from runtime objects ---
        try:
            ai.roi_mode = self._adaptive_roi.state_name
            region = self._adaptive_roi.region
            if region is not None:
                x1, y1, x2, y2 = region
                ai.roi_radius_px = 0.5 * max(x2 - x1, y2 - y1)
            else:
                import math
                ai.roi_radius_px = 0.5 * math.hypot(
                    s.camera.image_width, s.camera.image_height)
        except Exception:
            pass
        try:
            ai.model_status = self._mission_policy.model_status
        except Exception:
            pass

        # --- Search state ---
        ai.search_active = search_active
        if search_active and self._search_controller:
            ss = self._search_controller.state
            ai.search_phase = ss.phase.name
            ai.search_strategy = ss.strategy.value
            ai.search_center_x = ss.search_center_x
            ai.search_center_y = ss.search_center_y
            ai.search_radius_px = ss.search_radius_px
            ai.search_frames = ss.frames_in_search
            ai.search_time_s = ss.total_search_time_s

        # --- Search metrics ---
        if self._search_start_time_s is not None and not search_active:
            ai.search_duration_s = self._sim_time - self._search_start_time_s
            ai.search_strategy_used = self._last_search_strategy or "none"
            self._search_start_time_s = None


        # --- Populate targets_all from simulation ---
        if self._sim_engine:
            world = self._sim_engine.get_state()
            new_targets = []
            for wt in world.targets:
                tv = TargetView(
                    visible=wt.active,
                    world_x=wt.x, world_y=wt.y, world_z=wt.z,
                    trajectory_type=wt.trajectory_type,
                    size_px=10.0,
                )
                tv.target_id = wt.target_id
                # Per-target motion trail (bounded deque per id).
                if wt.active:
                    trail = self._target_trails.get(wt.target_id)
                    if trail is None:
                        from collections import deque as _dq
                        trail = _dq(maxlen=self._trail_len)
                        self._target_trails[wt.target_id] = trail
                    trail.append((float(wt.x), float(wt.y), float(wt.z)))
                    tv.trail = list(trail)
                new_targets.append(tv)
            # Drop trails of removed targets.
            live_ids = {wt.target_id for wt in world.targets}
            for tid in list(self._target_trails):
                if tid not in live_ids:
                    del self._target_trails[tid]
            s.targets_all = new_targets

        # --- Camera boresight trail (search sweep path) ---
        try:
            if self._camera is not None:
                from fsoc_tracker.simulation.camera.geometry import (
                    camera_rotation_matrix as _rot,
                )
                cst = self._camera.state
                depth = self._trail_depth
                desig = s.designated_beacon_id
                for t in s.targets_all:
                    if t.target_id == desig and t.visible:
                        depth = max(1.0, float(
                            ((t.world_x - cst.position_x) ** 2
                             + (t.world_y - cst.position_y) ** 2
                             + (t.world_z - cst.position_z) ** 2) ** 0.5))
                        break
                self._trail_depth = depth
                R = _rot(cst.pan_deg, cst.tilt_deg, cst.roll_deg)
                fx, fy, fz = R[0][2], R[1][2], R[2][2]
                self._camera_trail.append((
                    float(cst.position_x + fx * depth),
                    float(cst.position_y + fy * depth),
                    float(cst.position_z + fz * depth),
                ))
                s.camera_trail = list(self._camera_trail)
        except Exception:
            pass

        # --- Terminal A = tracking camera position/orientation ---
        # The optical axis comes from the camera's current pan/tilt.
        # This is the beam direction — NOT where the beacon actually is.
        # Position is read from the live camera (set at init from the
        # world's Terminal A pose and by place-terminal actions) — never
        # a hard-coded constant.
        s.terminal_a.active = True
        s.terminal_a.terminal_id = "TERM_A"
        s.terminal_a.yaw_deg = s.camera.pan_deg
        s.terminal_a.pitch_deg = s.camera.tilt_deg
        if self._camera is not None:
            s.terminal_a.world_x = self._camera.state.position_x
            s.terminal_a.world_y = self._camera.state.position_y
            s.terminal_a.world_z = self._camera.state.position_z
        s.terminal_a.hfov_deg = s.camera.hfov_deg
        s.terminal_a.vfov_deg = s.camera.vfov_deg

        # --- Terminal B = designated beacon (position from simulation) ---
        # Genuine 3D position exists ONLY in SIMULATION mode (sim engine).
        # In VIDEO/LIVE there is no monocular depth, so Terminal B stays
        # inactive and the link reports NO_LINK (no fabricated range).
        # Terminal B's orientation is not used by the optical link model.
        # The angular error is purely between Terminal A's optical axis
        # and the direction TO Terminal B.
        if s.designated_beacon_id is not None and self._sim_engine:
            world = self._sim_engine.get_state()
            for wt in world.targets:
                if wt.target_id == s.designated_beacon_id:
                    s.terminal_b.active = True
                    s.terminal_b.terminal_id = f"BEACON_{wt.target_id}"
                    s.terminal_b.world_x = wt.x
                    s.terminal_b.world_y = wt.y
                    s.terminal_b.world_z = wt.z
                    break
            else:
                s.terminal_b.active = False
        else:
            s.terminal_b.active = False

        # --- Optical link computation ---
        # Beam direction = Terminal A's optical axis (camera pan/tilt).
        # Angular error = angle between optical axis and direction to beacon.
        # Link quality = f(alignment, distance, atmosphere).
        # Always updated (even when a terminal is inactive) so the view
        # never shows stale status/alignment/range from a previous mode.
        from fsoc_tracker.simulation.optical_link import OpticalLinkEngine
        from fsoc_tracker.simulation.terminal import TerminalState

        ta = TerminalState(
            x=s.terminal_a.world_x, y=s.terminal_a.world_y, z=s.terminal_a.world_z,
            yaw_deg=s.terminal_a.yaw_deg, pitch_deg=s.terminal_a.pitch_deg,
            active=s.terminal_a.active,
        )
        tb = TerminalState(
            x=s.terminal_b.world_x, y=s.terminal_b.world_y, z=s.terminal_b.world_z,
            active=s.terminal_b.active,
        )

        if not hasattr(self, "_link_engine"):
            self._link_engine = OpticalLinkEngine()

        # Atmospheric attenuation from disturbance config.
        # Heuristic only: fog/rain/haze reduce link quality
        # multiplicatively. Coefficients are placeholders, not physics.
        from fsoc_tracker.simulation.optical_link import (
            atmospheric_attenuation_from_disturbance,
        )
        dist_pipeline = getattr(self, "_disturbance", None)
        cfg = dist_pipeline.config if dist_pipeline is not None else None
        atmo_atten = atmospheric_attenuation_from_disturbance(cfg)

        link = self._link_engine.update(ta, tb, atmospheric_attenuation=atmo_atten)
        s.optical_link.status = link.status.value
        s.optical_link.beam_alignment_percent = link.beam_alignment_percent
        s.optical_link.range_m = link.range_m
        s.optical_link.angular_error_deg = link.angular_error_deg
        s.optical_link.link_quality_percent = link.link_quality_percent

        if not hasattr(self, "_comm_engine"):
            from fsoc_tracker.simulation.communication import CommunicationEngine
            self._comm_engine = CommunicationEngine()

        self._comm_engine.update(link, self._sim_time)
        from fsoc_tracker.gui.state import MessageView
        s.messages = [
            MessageView(
                message_id=m.message_id, source=m.source, destination=m.destination,
                timestamp_s=m.timestamp_s, payload=m.payload, status=m.status.value,
            )
            for m in self._comm_engine.message_log
        ]

        if camera_cmd is not None:
            s.control.pan_deg = self._camera.state.pan_deg if self._camera else 0.0
            s.control.tilt_deg = self._camera.state.tilt_deg if self._camera else 0.0
            s.control.pan_command = camera_cmd.pan_rate_deg_s
            s.control.tilt_command = camera_cmd.tilt_rate_deg_s
            s.control.pan_saturated = camera_cmd.pan_saturated
            s.control.tilt_saturated = camera_cmd.tilt_saturated
            s.control.control_mode = camera_cmd.control_mode.value

        if gt is not None:
            s.target.visible = gt.target_visible
            s.target.pixel_x = gt.target_pixel_x
            s.target.pixel_y = gt.target_pixel_y
            s.target.size_px = gt.target_size_px
            if self._sim_engine:
                ts = self._sim_engine.get_state()
                if ts.targets:
                    s.target.world_x = ts.targets[0].x
                    s.target.world_y = ts.targets[0].y

        if error_px is not None:
            s.errors.timestamps.append(self._sim_time)
            s.errors.errors_x.append(error_x)
            s.errors.errors_y.append(error_y)
            s.errors.errors_euclidean.append(error_px)
            s.errors.fps_history.append(s.camera.fps)
            n = s.errors.max_len
            if len(s.errors.timestamps) > n:
                s.errors.timestamps = s.errors.timestamps[-n:]
                s.errors.errors_x = s.errors.errors_x[-n:]
                s.errors.errors_y = s.errors.errors_y[-n:]
                s.errors.errors_euclidean = s.errors.errors_euclidean[-n:]
                s.errors.fps_history = s.errors.fps_history[-n:]

        s.scorecard.fps = s.camera.fps
        if error_px is not None:
            s.scorecard.rmse_px = error_px

        # Scorecard tracking: acquisition time, loss rate, reacquisition
        is_tracking = s.tracking.state == "TRACKING"
        is_detected = s.perception.detected

        if is_detected and self._first_detection_s is None:
            self._first_detection_s = self._sim_time

        if is_tracking and self._acquisition_time_s is None and self._first_detection_s is not None:
            self._acquisition_time_s = self._sim_time - self._first_detection_s
            s.scorecard.acquisition_s = self._acquisition_time_s

        if not is_tracking and self._was_tracking:
            self._loss_count += 1
            self._last_loss_s = self._sim_time

        if is_tracking and not self._was_tracking and self._last_loss_s is not None:
            reacq = self._sim_time - self._last_loss_s
            self._reacquisition_time_s = reacq
            s.scorecard.reacq_s = reacq

        self._was_tracking = is_tracking
        if is_tracking:
            self._total_frames_tracked += 1

        if self._frame_index > 0:
            loss_pct = (self._loss_count / max(self._frame_index, 1)) * 100.0
            s.scorecard.loss_percent = loss_pct
            s.scorecard.lock_retention_pct = (
                self._total_frames_tracked / max(self._frame_index, 1)
            ) * 100.0

        # Populate disturbance view
        if self._disturbance is not None:
            s.disturbances.enabled = self._disturbance.config.enabled
            profile = getattr(self._disturbance.config, "profile", None)
            if profile is None:
                atmosphere_mode = getattr(self._disturbance.config.atmosphere, "mode", None)
                profile = atmosphere_mode.value if getattr(atmosphere_mode, "value", None) is not None else "clear"
            s.disturbances.profile = str(profile)

        self._emit_state()

    def _emit_state(self) -> None:
        self.state_updated.emit(self._state)

    def run(self) -> None:
        while not self._shutdown:
            if not self._running:
                self.msleep(50)
                continue
            if self._paused:
                self.msleep(50)
                continue
            self._run_one_step()
            self.msleep(1)
