"""Closed-loop simulation benchmark — one real execution per method.

Each benchmark method runs the SAME closed-loop harness (simulation +
render + perception + estimation + control) with a genuinely different
estimation/decision stack. Nothing is stubbed: trajectories, metric
values, link states, and delivered messages all come from executed
frames. Ground truth is used for SCORING ONLY (error metrics, FOV
retention, link geometry) — never by perception, tracking, control,
or the mission brain.

Method stacks:
- CLASSICAL_PID: raw detection centroid drives a proportional law
  (no Kalman filter, no mission brain).
- KALMAN_EXPERT: Kalman-filtered track drives PID (engineered stack,
  no learned components).
- LEARNED_TEMPORAL_EXPERT: Kalman + PID + mission brain with a
  trained linear motion model (ROI control, predictive control
  during outages, search). Expert policy.
- LEARNED_TEMPORAL_LEARNED_POLICY: as above with the trained policy
  classifier selecting safety-gated actions. Refuses to run when the
  policy weights are missing or stale (see benchmark.methods).
- FULL_AI_MISSION: the production TrackingPipeline with the full
  adaptive stack (mission brain, adaptive ROI/Kalman/controller,
  failure predictor, search controller).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fsoc_tracker.benchmark.methods import (
    METHOD_CLASSICAL_PID,
    METHOD_FULL_AI_MISSION,
    METHOD_KALMAN_EXPERT,
    METHOD_LABELS,
    METHOD_LEARNED_TEMPORAL_EXPERT,
    METHOD_LEARNED_TEMPORAL_LEARNED_POLICY,
    MethodUnavailableError,
    check_method_availability,
    load_method_components,
)
from fsoc_tracker.control.controller import CoarsePointingController
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.pipeline.eval import EvalSink
from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.simulation.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor import SensorConfig, VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.config import AssociationMethod, TrackerConfig
from fsoc_tracker.tracking.tracker import KalmanTracker


class BenchmarkMode(str, Enum):
    CLASSICAL_PID = "classical_pid"
    KALMAN_EXPERT = "kalman_expert"
    LEARNED_TEMPORAL_EXPERT = "learned_temporal_expert"
    LEARNED_TEMPORAL_LEARNED_POLICY = "learned_temporal_learned_policy"
    FULL_AI_MISSION = "full_ai_mission"


MODE_TO_METHOD: dict[BenchmarkMode, str] = {
    BenchmarkMode.CLASSICAL_PID: METHOD_CLASSICAL_PID,
    BenchmarkMode.KALMAN_EXPERT: METHOD_KALMAN_EXPERT,
    BenchmarkMode.LEARNED_TEMPORAL_EXPERT: METHOD_LEARNED_TEMPORAL_EXPERT,
    BenchmarkMode.LEARNED_TEMPORAL_LEARNED_POLICY: METHOD_LEARNED_TEMPORAL_LEARNED_POLICY,
    BenchmarkMode.FULL_AI_MISSION: METHOD_FULL_AI_MISSION,
}

ProgressCallback = Callable[[int, int], None] | None
LogCallback = Callable[[str, str], None] | None

# Link states during which the channel is unusable (downtime accounting).
LINK_DOWN_STATES = ("LOST", "NO_LINK", "REACQUIRING")

# Frame columns for the per-frame CSV log.
FRAME_LOG_COLUMNS: tuple[str, ...] = (
    "frame_index", "timestamp_s", "detected", "confidence",
    "track_x", "track_y", "track_state", "locked",
    "error_px", "gt_visible", "fov_inside",
    "pan_deg", "tilt_deg",
    "link_status", "angular_error_deg", "alignment_pct",
    "link_quality_pct", "range_m",
    "ai_action", "ai_situation", "ai_approved",
    "processing_ms",
)


@dataclass
class BenchmarkRunConfig:
    mode: BenchmarkMode = BenchmarkMode.KALMAN_EXPERT
    # Generated world profile (see simulation.world_builder) + seed.
    # Replaces the retired Scene 1-15 preset IDs.
    world_profile: str = "nominal"
    seed: int = 42
    max_frames: int = 300
    sim_dt: float = 1.0 / 30.0
    output_dir: str = "logs/benchmark"
    disturbance_preset: str = "clear"
    assoc_gate_px: float | None = None
    assoc_method: str | None = None
    assoc_appearance: float | None = None
    assoc_identity: float | None = None
    identity_code: str | None = None
    lead_compensation: bool = False
    lead_time_s: float | None = None


# Disturbance shorthand per generated world profile. Mirrors the retired
# scene definitions so benchmark behavior is preserved across the cutover.
_WORLD_PROFILE_DISTURBANCES: dict[str, dict] = {
    "nominal": {},
    "multi": {},
    "distractor": {"distractors": True},
    "loss": {"target_disappearance": True},
    "noise": {"noise": 0.5},
    "fog": {"fog": 0.8},
    "jitter": {"jitter_px": 5.0},
}


@dataclass
class BenchmarkMetrics:
    """Per-run record. Every field comes from executed frames."""

    method: str = ""
    scenario: str = ""
    seed: int = 42
    source: str = "simulation"
    duration_s: float = 0.0
    acquisition_s: float | None = None
    rmse_px: float = 0.0
    mae_px: float = 0.0
    p95_px: float = 0.0
    max_error_px: float = 0.0
    loss_rate_pct: float = 0.0
    reacquisition_s: float | None = None
    reacquisition_from_reappearance_s: float | None = None
    lock_retention_pct: float = 0.0
    fov_retention_pct: float = 0.0
    search_duration_s: float = 0.0
    fps: float = 0.0
    latency_ms: float = 0.0
    latency_p95_ms: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    target_losses: int = 0
    safety_events: int = 0
    safety_note: str = ""
    link_downtime_s: float = 0.0
    link_note: str = ""
    messages_delivered: int = 0
    messages_total: int = 0
    ai_action_histogram: dict[str, int] = field(default_factory=dict)
    prediction_rmse_px: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    log_json_path: str = ""
    log_csv_path: str = ""
    repro_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scenario": self.scenario,
            "seed": self.seed,
            "source": self.source,
            "duration_s": round(self.duration_s, 3),
            "acquisition_s": self.acquisition_s,
            "rmse_px": round(self.rmse_px, 2),
            "mae_px": round(self.mae_px, 2),
            "p95_px": round(self.p95_px, 2),
            "max_error_px": round(self.max_error_px, 2),
            "loss_rate_pct": round(self.loss_rate_pct, 1),
            "reacquisition_s": self.reacquisition_s,
            "reacquisition_from_reappearance_s": self.reacquisition_from_reappearance_s,
            "lock_retention_pct": round(self.lock_retention_pct, 1),
            "fov_retention_pct": round(self.fov_retention_pct, 1),
            "search_duration_s": round(self.search_duration_s, 3),
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "target_losses": self.target_losses,
            "safety_events": self.safety_events,
            "safety_note": self.safety_note,
            "link_downtime_s": round(self.link_downtime_s, 3),
            "link_note": self.link_note,
            "messages_delivered": self.messages_delivered,
            "messages_total": self.messages_total,
            "ai_action_histogram": dict(self.ai_action_histogram),
            "prediction_rmse_px": (
                round(self.prediction_rmse_px, 2)
                if self.prediction_rmse_px is not None else None
            ),
            "provenance": dict(self.provenance),
            "completed": self.completed,
            "log_json_path": self.log_json_path,
            "log_csv_path": self.log_csv_path,
            "repro_command": self.repro_command,
        }

    def summary_table(self) -> str:
        def fmt_acq(value: float | None) -> str:
            return f"{value:.3f}s" if value is not None else "N/A"

        rows = [
            ("Method", self.method),
            ("Scenario / seed", f"{self.scenario} / {self.seed}"),
            ("Duration", f"{self.duration_s:.2f}s ({self.frames_processed} frames)"),
            ("Acquisition", fmt_acq(self.acquisition_s)),
            ("RMSE / MAE", f"{self.rmse_px:.2f} / {self.mae_px:.2f} px"),
            ("P95 / Max", f"{self.p95_px:.2f} / {self.max_error_px:.2f} px"),
            ("Target loss", f"{self.loss_rate_pct:.1f}% ({self.target_losses} events)"),
            ("Reacquisition", fmt_acq(self.reacquisition_s)),
            ("Reacq from reappearance", fmt_acq(self.reacquisition_from_reappearance_s)),
            ("Lock retention", f"{self.lock_retention_pct:.1f}%"),
            ("FOV retention", f"{self.fov_retention_pct:.1f}%"),
            ("Search duration", f"{self.search_duration_s:.3f}s"),
            ("FPS / latency", f"{self.fps:.1f} / {self.latency_ms:.2f} ms"),
            ("Link downtime", f"{self.link_downtime_s:.3f}s"),
            ("Messages", f"{self.messages_delivered}/{self.messages_total} delivered"),
            ("Safety events", str(self.safety_events)),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}s} : {value}" for name, value in rows)


def build_repro_command(config: BenchmarkRunConfig) -> str:
    mode_value = config.mode.value if isinstance(config.mode, BenchmarkMode) else str(config.mode)
    cmd = (
        "python -m fsoc_tracker.cli.run_benchmark sim"
        f" --method {mode_value} --world {config.world_profile}"
        f" --seed {config.seed} --frames {config.max_frames}"
        f" --output {config.output_dir}"
    )
    # Every tuning flag that changes results must round-trip here, or
    # the recorded command silently reproduces different numbers.
    if config.assoc_gate_px is not None:
        cmd += f" --assoc-gate-px {config.assoc_gate_px}"
    if config.assoc_method is not None:
        cmd += f" --assoc-method {config.assoc_method}"
    if config.assoc_appearance is not None:
        cmd += f" --assoc-appearance {config.assoc_appearance}"
    if config.assoc_identity is not None:
        cmd += f" --assoc-identity {config.assoc_identity}"
    if config.identity_code:
        cmd += f" --identity-code {config.identity_code}"
    if config.lead_compensation:
        cmd += " --lead"
    if config.lead_time_s is not None:
        cmd += f" --lead-time {config.lead_time_s}"
    return cmd


class BenchmarkRunner:
    """Runs one closed-loop benchmark. Blocks until complete or stopped.

    Thread-safety: run() executes wherever it is called. The GUI calls
    it from a QThread worker so the interface stays responsive.
    """

    def __init__(self) -> None:
        self._running = False
        self._stop_requested = False

    @property
    def running(self) -> bool:
        return self._running

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        config: BenchmarkRunConfig,
        progress_callback: ProgressCallback = None,
        log_callback: LogCallback = None,
    ) -> BenchmarkMetrics:
        """Run a single benchmark. Blocks until complete or stopped."""
        method = MODE_TO_METHOD.get(config.mode, str(config.mode))
        availability = check_method_availability(method)
        if not availability.available:
            # Never fabricate results for an unavailable method.
            raise MethodUnavailableError(
                f"Benchmark method '{METHOD_LABELS.get(method, method)}' NOT AVAILABLE: "
                f"{availability.reason}"
            )
        if config.mode == BenchmarkMode.FULL_AI_MISSION:
            return self._run_full_ai_mission(config, progress_callback, log_callback)
        return self._run_explicit_loop(config, progress_callback, log_callback)

    # ------------------------------------------------------------------
    # Explicit closed-loop harness (methods 1-4)
    # ------------------------------------------------------------------
    def _run_explicit_loop(
        self,
        config: BenchmarkRunConfig,
        progress_callback: ProgressCallback = None,
        log_callback: LogCallback = None,
    ) -> BenchmarkMetrics:
        from fsoc_tracker.simulation.communication import CommunicationEngine
        from fsoc_tracker.simulation.optical_link import OpticalLinkEngine
        from fsoc_tracker.simulation.terminal import TerminalState
        from fsoc_tracker.tracking.search import SearchConfig, SearchController

        method = MODE_TO_METHOD.get(config.mode, str(config.mode))
        metrics = BenchmarkMetrics(
            method=method,
            scenario=f"world_{config.world_profile}",
            seed=config.seed,
            source="simulation",
            repro_command=build_repro_command(config),
        )
        frame_rows: list[dict[str, Any]] = []
        self._running = True
        self._stop_requested = False

        def log(level: str, msg: str) -> None:
            if log_callback:
                log_callback(level, msg)

        try:
            use_kalman = config.mode != BenchmarkMode.CLASSICAL_PID
            use_brain = config.mode in (
                BenchmarkMode.LEARNED_TEMPORAL_EXPERT,
                BenchmarkMode.LEARNED_TEMPORAL_LEARNED_POLICY,
            )
            components: dict[str, Any] = {}
            if use_brain:
                components = load_method_components(method)
            metrics.provenance = {
                "harness": "explicit_closed_loop",
                "tracker": "kalman" if use_kalman else "raw_detection",
                "controller": "pid" if use_kalman else "proportional_raw",
                "mission_brain": "expert+learned_motion"
                if method == METHOD_LEARNED_TEMPORAL_EXPERT
                else ("expert+learned_motion+learned_policy"
                      if method == METHOD_LEARNED_TEMPORAL_LEARNED_POLICY else "none"),
                **components.get("provenance", {}),
            }

            wc = WorldConfig(width=2000.0, height=2000.0, random_seed=config.seed)
            engine = SimulationEngine(wc)
            world = None
            try:
                from fsoc_tracker.simulation.world_builder import (
                    build_benchmark_world,
                )
                world = build_benchmark_world(config.world_profile,
                                              seed=config.seed)
                engine.load_scenario(world)
                metrics.scenario = world.name
            except Exception as e:
                log("WARN", f"World build failed: {e}")
            from fsoc_tracker.disturbances.config import (
                DisturbanceConfig,
                apply_world_disturbances,
                get_preset_config,
            )
            from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
            _dist_base = (
                get_preset_config(config.disturbance_preset)
                if config.disturbance_preset != "clear"
                else DisturbanceConfig()
            )
            apply_world_disturbances(
                _dist_base,
                _WORLD_PROFILE_DISTURBANCES.get(config.world_profile, {}),
                seed=config.seed)
            disturbance = (
                DisturbancePipeline(_dist_base)
                if _dist_base.enabled else None
            )

            camera = VirtualCamera(CameraState(
                horizontal_fov_deg=4.0,
                vertical_fov_deg=3.0,
                width=640,
                height=480,
                max_pan_speed_deg_s=5.0,
                max_tilt_speed_deg_s=5.0,
            ))
            sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))
            detector = ClassicalBeaconDetector()
            tracker = None
            if use_kalman:
                tracker_cfg = TrackerConfig()
                if config.assoc_gate_px is not None:
                    tracker_cfg.association_gate_px = float(config.assoc_gate_px)
                if config.assoc_method is not None:
                    tracker_cfg.association_method = AssociationMethod(
                        config.assoc_method)
                if config.assoc_appearance is not None:
                    tracker_cfg.appearance_weight = float(config.assoc_appearance)
                if config.assoc_identity is not None:
                    tracker_cfg.identity_weight = float(config.assoc_identity)
                if config.identity_code:
                    tracker_cfg.identity_expected_code = str(config.identity_code)
                tracker = KalmanTracker(tracker_cfg)
            controller = CoarsePointingController()
            if config.lead_compensation:
                controller._config.lead_compensation_enabled = True
                if config.lead_time_s is not None:
                    controller._config.lead_time_s = float(config.lead_time_s)
            search = SearchController(SearchConfig()) if use_kalman else None
            link_engine = OpticalLinkEngine()
            comm = CommunicationEngine()

            brain = None
            if use_brain:
                from fsoc_tracker.ai.mission import (
                    AIMissionBrain,
                    MissionObservation,
                    ObservationFeatures,
                )
                brain_components: dict[str, Any] = {}
                if components.get("learned_motion") is not None:
                    brain_components["learned_motion"] = components["learned_motion"]
                if components.get("learned_policy") is not None:
                    brain_components["learned_policy"] = components["learned_policy"]
                brain = AIMissionBrain(**brain_components)

            errors: list[float] = []
            pred_errors: list[float] = []
            reacquisition_times: list[float] = []
            reacquisition_from_reappearance: list[float] = []
            reappear_s: float | None = None
            frame_times: list[float] = []
            fov_inside_count = 0
            fov_total_count = 0
            locked_time_s = 0.0
            loss_count = 0
            search_time_s = 0.0
            acquisition_s: float | None = None
            first_lock_seen = False
            was_locked = False
            loss_start_s: float | None = None
            search_active = False
            last_est = (320.0, 240.0)
            last_detect_s: float | None = None
            roi_radius: float | None = None  # None = full frame
            roi_frames = 0
            safety_events = 0
            action_histogram: dict[str, int] = {}
            link_downtime_s = 0.0
            next_probe_s = 1.0
            probe_id = 0
            sim_time = 0.0
            prev_proc_ms = 0.0
            prev_platform = engine.get_terminal_a_state()

            for frame_idx in range(config.max_frames):
                if self._stop_requested:
                    log("INFO", "Benchmark stopped by user")
                    break
                t_start = time.perf_counter()

                engine.step(config.sim_dt)
                # Strapdown carry (Terminal A itself moves): the mount
                # rides the platform; gimbal commands operate on top.
                try:
                    from fsoc_tracker.simulation.platform import (
                        apply_platform_delta_to_camera,
                    )
                    cur_platform = engine.get_terminal_a_state()
                    apply_platform_delta_to_camera(
                        camera.state, prev_platform, cur_platform)
                    prev_platform = cur_platform
                except Exception:
                    pass
                active_targets = engine.get_state().get_active_targets()
                render_camera = camera
                disturbance_context = None
                if disturbance is not None and disturbance.config.enabled:
                    from dataclasses import replace

                    from fsoc_tracker.disturbances.context import (
                        CameraPoseContext,
                        DisturbanceContext,
                    )
                    disturbance_context = DisturbanceContext(
                        timestamp_s=sim_time,
                        dt=config.sim_dt,
                        frame_index=frame_idx,
                        image_width=640,
                        image_height=480,
                        camera_pose=CameraPoseContext(
                            position_x=camera.state.position_x,
                            position_y=camera.state.position_y,
                            position_z=camera.state.position_z,
                            pan_deg=camera.state.pan_deg,
                            tilt_deg=camera.state.tilt_deg,
                            roll_deg=camera.state.roll_deg,
                        ),
                    )
                    if disturbance.should_suppress_target(sim_time):
                        active_targets = []
                    effective_pose = disturbance.compute_effective_pose(
                        disturbance_context.camera_pose,
                        disturbance_context,
                    )
                    render_camera = VirtualCamera(replace(
                        camera.state,
                        position_x=effective_pose.position_x,
                        position_y=effective_pose.position_y,
                        position_z=effective_pose.position_z,
                        pan_deg=effective_pose.pan_deg,
                        tilt_deg=effective_pose.tilt_deg,
                        roll_deg=effective_pose.roll_deg,
                    ))
                rendered = sensor.render(render_camera, active_targets,
                                         sim_time, frame_idx)
                image = rendered.image.copy()
                if disturbance_context is not None and disturbance is not None:
                    image = disturbance.apply_to_image(
                        image, disturbance_context)

                # --- Perception (ROI crop when the brain selects it) ---
                detect_image = image
                roi_offset = (0, 0)
                if roi_radius is not None:
                    h, w = image.shape[:2]
                    cx = int(min(max(last_est[0], 0), w - 1))
                    cy = int(min(max(last_est[1], 0), h - 1))
                    r = int(roi_radius)
                    x1, y1 = max(0, cx - r), max(0, cy - r)
                    x2, y2 = min(w, cx + r), min(h, cy + r)
                    if x2 > x1 and y2 > y1:
                        detect_image = image[y1:y2, x1:x2]
                        roi_offset = (x1, y1)
                        roi_frames += 1
                det_result = detector.detect(detect_image, sim_time, frame_idx)
                if roi_offset != (0, 0) and det_result.detections:
                    ox, oy = roi_offset
                    for cand in det_result.detections:
                        cand.center_x += ox
                        cand.center_y += oy
                primary = det_result.primary_detection
                detected = bool(primary and primary.detected)
                if detected:
                    last_detect_s = sim_time

                # --- Estimation ---
                track_state_name = ""
                locked = False
                if use_kalman:
                    assert tracker is not None
                    # Full list (mirrors pipeline.py): the tracker itself
                    # splits above-threshold association from the rescue
                    # pass; feeding only the primary hid all competition.
                    detections = list(det_result.detections)
                    trk = tracker.update(detections, sim_time)
                    est_x, est_y = trk.estimated_x, trk.estimated_y
                    track_state_name = trk.state.name
                    locked = bool(trk.locked)
                    res_px = trk.residual_magnitude
                    unc_x, unc_y = trk.uncertainty_x, trk.uncertainty_y
                    vel_x, vel_y = trk.velocity_x, trk.velocity_y
                else:
                    # CLASSICAL: no filter — raw centroid or hold last.
                    if detected:
                        assert primary is not None
                        est_x, est_y = primary.center_x, primary.center_y
                    else:
                        est_x, est_y = last_est
                    track_state_name = "CLASSICAL_HOLD" if not detected else "CLASSICAL_TRACK"
                    locked = detected
                    res_px = 0.0 if detected else float(np.hypot(est_x - 320.0, est_y - 240.0))
                    unc_x = unc_y = 0.0
                    vel_x = vel_y = 0.0
                last_est = (est_x, est_y)

                # --- Mission brain (AI modes): real decisions on observables ---
                # Runs BEFORE search/control so the selected action drives
                # this frame's camera motion (ROI applies next frame).
                ai_action = ""
                ai_situation = ""
                ai_approved: Any = ""
                brain_prediction = None
                if brain is not None:
                    features = ObservationFeatures(
                        timestamp_s=sim_time,
                        detected=detected,
                        confidence=primary.confidence if primary else 0.0,
                        residual_px=res_px,
                        uncertainty_x_px=unc_x,
                        uncertainty_y_px=unc_y,
                        velocity_x_px_s=vel_x,
                        velocity_y_px_s=vel_y,
                        distance_from_center_px=float(
                            np.hypot(est_x - 320.0, est_y - 240.0)),
                        time_since_detection_s=(
                            (sim_time - last_detect_s)
                            if last_detect_s is not None else 0.0),
                        latency_ms=prev_proc_ms,
                        source_fps=1.0 / config.sim_dt,
                        processing_fps=(1000.0 / prev_proc_ms) if prev_proc_ms > 0 else 0.0,
                        candidate_count=det_result.num_candidates,
                        roi_radius_px=roi_radius or 0.0,
                    )
                    decision = brain.decide(MissionObservation(
                        features=features,
                        camera_pan_deg=camera.state.pan_deg,
                        camera_tilt_deg=camera.state.tilt_deg,
                    ))
                    ai_action = decision.action.value
                    ai_situation = decision.situation.value
                    ai_approved = bool(decision.safety.approved)
                    action_histogram[ai_action] = action_histogram.get(ai_action, 0) + 1
                    if not decision.safety.approved:
                        safety_events += 1
                    # ROI actuation for the NEXT frame.
                    roi_radius = 80.0 if ai_action == "use_roi" else None
                    brain_prediction = decision.prediction

                # --- Search ---
                # Engineered stack: search whenever the tracker is lost
                # (production behavior). Brain modes: search only when the
                # safety-approved action selects it, so policy choices
                # genuinely move the camera.
                track_lost = track_state_name in ("LOST", "NO_TRACK")
                want_search = track_lost and (
                    brain is None
                    or ai_action in ("local_search", "global_search", "reacquire")
                )
                if search is not None:
                    if want_search:
                        if not search_active:
                            search_active = True
                            search.begin_search(
                                est_x, est_y, vel_x, vel_y, sim_time,
                                uncertainty_x=unc_x, uncertainty_y=unc_y,
                                current_pan_deg=camera.state.pan_deg,
                                current_tilt_deg=camera.state.tilt_deg,
                            )
                        else:
                            search.update(
                                config.sim_dt,
                                current_pan_deg=camera.state.pan_deg,
                                current_tilt_deg=camera.state.tilt_deg,
                            )
                        camera.set_target_pan_tilt(
                            camera.state.pan_deg + search.pan_rate_deg_s * config.sim_dt,
                            camera.state.tilt_deg + search.tilt_rate_deg_s * config.sim_dt,
                        )
                        search_time_s += config.sim_dt
                        if (detected and primary is not None and search.process_detection(
                            primary.center_x, primary.center_y,
                            primary.confidence, sim_time,
                        )):
                            search_active = False
                            search.reset()
                    elif search_active:
                        search_active = False
                        search.reset()

                # --- Control ---
                predictive_case = (
                    brain_prediction is not None
                    and getattr(brain_prediction, "method", "") == "learned_motion"
                    and not detected
                    and ai_action in ("track", "track_predictive")
                )
                if not search_active:
                    if ai_action in ("hold", "safe_stop"):
                        # Safety hold: freeze the platform.
                        camera.set_target_pan_tilt(
                            camera.state.pan_deg, camera.state.tilt_deg)
                    elif predictive_case:
                        # Predictive control during outages: steer toward the
                        # learned-displacement projection of the last estimate.
                        # Documented harness law: proportional gain 3.0,
                        # clamped to the 5 deg/s platform limit.
                        assert brain_prediction is not None
                        pred_x = last_est[0] + brain_prediction.mean_x_px
                        pred_y = last_est[1] + brain_prediction.mean_y_px
                        err_x = (pred_x - 320.0) / 320.0
                        err_y = (pred_y - 240.0) / 240.0
                        camera.set_target_pan_tilt(
                            camera.state.pan_deg
                            + max(-5.0, min(5.0, err_x * 3.0)) * config.sim_dt,
                            camera.state.tilt_deg
                            + max(-5.0, min(5.0, err_y * 3.0)) * config.sim_dt,
                        )
                    elif use_kalman:
                        camera_cmd, _ = controller.compute(
                            trk, camera.intrinsics, config.sim_dt, sim_time,
                        )
                        camera.set_target_pan_tilt(
                            camera.state.pan_deg + camera_cmd.pan_rate_deg_s * config.sim_dt,
                            camera.state.tilt_deg + camera_cmd.tilt_rate_deg_s * config.sim_dt,
                        )
                    else:
                        # CLASSICAL: proportional law on the raw centroid,
                        # same documented gain/limit as the predictive case.
                        err_x = (est_x - 320.0) / 320.0
                        err_y = (est_y - 240.0) / 240.0
                        camera.set_target_pan_tilt(
                            camera.state.pan_deg
                            + max(-5.0, min(5.0, err_x * 3.0)) * config.sim_dt,
                            camera.state.tilt_deg
                            + max(-5.0, min(5.0, err_y * 3.0)) * config.sim_dt,
                        )
                camera.update(config.sim_dt)

                # --- Scoring (ground truth ONLY: error metrics, FOV, link) ---
                gt = rendered.truth_for(engine.primary_beacon_id)
                gt_visible = bool(gt is not None and gt.target_visible)
                error: float | None = None
                fov_inside = False
                if gt_visible:
                    assert gt is not None
                    error = float(np.hypot(est_x - gt.target_pixel_x,
                                           est_y - gt.target_pixel_y))
                    errors.append(error)
                    fov_total_count += 1
                    fov_inside = (0 <= gt.target_pixel_x < 640
                                  and 0 <= gt.target_pixel_y < 480)
                    if fov_inside:
                        fov_inside_count += 1
                    if (brain_prediction is not None
                            and getattr(brain_prediction, "method", "") == "learned_motion"):
                        pred_errors.append(float(np.hypot(
                            (est_x + brain_prediction.mean_x_px) - gt.target_pixel_x,
                            (est_y + brain_prediction.mean_y_px) - gt.target_pixel_y,
                        )))

                # --- Lock / loss / acquisition accounting (source time) ---
                evaluable_time_s_total = sim_time + config.sim_dt
                if locked:
                    locked_time_s += config.sim_dt
                    if not first_lock_seen:
                        first_lock_seen = True
                        acquisition_s = sim_time
                    if loss_start_s is not None:
                        reacquisition_times.append(sim_time - loss_start_s)
                        if reappear_s is not None:
                            reacquisition_from_reappearance.append(
                                sim_time - reappear_s)
                        loss_start_s = None
                        reappear_s = None
                else:
                    if was_locked and loss_start_s is None:
                        loss_count += 1
                        loss_start_s = sim_time
                        reappear_s = None
                    if loss_start_s is not None and reappear_s is None and gt_visible:
                        reappear_s = sim_time
                was_locked = locked

                # --- Optical link + comm (SIM geometry; scoring/model use) ---
                world_targets = engine.get_state().get_active_targets()
                tb_active = len(world_targets) > 0
                ta_state = TerminalState(
                    x=1000.0, y=1000.0, z=50.0,
                    yaw_deg=camera.state.pan_deg,
                    pitch_deg=camera.state.tilt_deg,
                    active=True,
                )
                tb_state = TerminalState(
                    x=world_targets[0].x if tb_active else 0.0,
                    y=world_targets[0].y if tb_active else 0.0,
                    z=world_targets[0].z if tb_active else 0.0,
                    active=tb_active,
                )
                link = link_engine.update(ta_state, tb_state)
                if link.status.value in LINK_DOWN_STATES:
                    link_downtime_s += config.sim_dt
                while sim_time >= next_probe_s:
                    probe = comm.create_message("TERM_A", "TERM_B", f"PROBE-{probe_id}", sim_time)
                    comm.send_message(probe)
                    probe_id += 1
                    next_probe_s += 1.0
                comm.update(link, sim_time)

                t_end = time.perf_counter()
                proc_ms = (t_end - t_start) * 1000.0
                frame_times.append(proc_ms)
                prev_proc_ms = proc_ms

                frame_rows.append({
                    "frame_index": frame_idx,
                    "timestamp_s": round(sim_time, 4),
                    "detected": detected,
                    "confidence": round(primary.confidence, 3) if primary else 0.0,
                    "track_x": round(est_x, 2),
                    "track_y": round(est_y, 2),
                    "track_state": track_state_name,
                    "locked": locked,
                    "error_px": round(error, 2) if error is not None else "",
                    "gt_visible": gt_visible,
                    "fov_inside": fov_inside,
                    "pan_deg": round(camera.state.pan_deg, 3),
                    "tilt_deg": round(camera.state.tilt_deg, 3),
                    "link_status": link.status.value,
                    "angular_error_deg": round(link.angular_error_deg, 3),
                    "alignment_pct": round(link.beam_alignment_percent, 1),
                    "link_quality_pct": round(link.link_quality_percent, 1),
                    "range_m": round(link.range_m, 1),
                    "ai_action": ai_action,
                    "ai_situation": ai_situation,
                    "ai_approved": ai_approved,
                    "processing_ms": round(proc_ms, 3),
                })

                if progress_callback:
                    progress_callback(frame_idx + 1, config.max_frames)
                sim_time += config.sim_dt

            # --- Finalize record (all values from executed frames) ---
            frames_run = len(frame_rows)
            metrics.frames_processed = frames_run
            metrics.total_frames = config.max_frames
            metrics.duration_s = sim_time
            metrics.target_losses = loss_count
            if errors:
                arr = np.array(errors)
                metrics.rmse_px = float(np.sqrt(np.mean(arr ** 2)))
                metrics.mae_px = float(np.mean(np.abs(arr)))
                metrics.p95_px = float(np.percentile(arr, 95))
                metrics.max_error_px = float(np.max(arr))
            if evaluable_time_s_total > 0:
                retention = min(locked_time_s / evaluable_time_s_total, 1.0)
                metrics.lock_retention_pct = retention * 100.0
                metrics.loss_rate_pct = (1.0 - retention) * 100.0
            metrics.acquisition_s = acquisition_s
            metrics.reacquisition_s = (
                float(np.mean(reacquisition_times)) if reacquisition_times else None
            )
            metrics.reacquisition_from_reappearance_s = (
                float(np.mean(reacquisition_from_reappearance))
                if reacquisition_from_reappearance else None
            )
            metrics.search_duration_s = search_time_s
            metrics.fov_retention_pct = (
                (fov_inside_count / fov_total_count * 100.0) if fov_total_count else 0.0
            )
            if frame_times:
                metrics.latency_ms = float(np.mean(frame_times))
                metrics.latency_p95_ms = float(np.percentile(frame_times, 95))
                metrics.fps = 1000.0 / max(float(np.mean(frame_times)), 1e-6)
            metrics.prediction_rmse_px = (
                float(np.sqrt(np.mean(np.array(pred_errors) ** 2))) if pred_errors else None
            )
            metrics.link_downtime_s = link_downtime_s
            metrics.messages_total = probe_id
            # Only fully completed transmissions count as delivered;
            # in-flight probes at teardown are neither delivered nor failed.
            metrics.messages_delivered = sum(
                1 for m in comm.message_log if m.status.value == "DELIVERED"
            )
            metrics.completed = not self._stop_requested
            metrics.safety_events = safety_events
            metrics.safety_note = (
                "safety vetoes counted from mission-brain approvals"
                if use_brain else "no mission brain in loop"
            )
            metrics.ai_action_histogram = action_histogram
            metrics.link_note = "SIM closed loop: camera axis vs simulated beacon position"
            self._write_logs(metrics, frame_rows, config, log)
            log(
                "INFO",
                f"Benchmark complete: method={method} "
                f"RMSE={metrics.rmse_px:.2f}px FPS={metrics.fps:.1f}",
            )
        except Exception as e:
            log("ERROR", f"Benchmark failed: {e}")
            raise
        finally:
            self._running = False
        return metrics

    # ------------------------------------------------------------------
    # Production-stack harness (FULL AI MISSION)
    # ------------------------------------------------------------------
    def _run_full_ai_mission(
        self,
        config: BenchmarkRunConfig,
        progress_callback: ProgressCallback = None,
        log_callback: LogCallback = None,
    ) -> BenchmarkMetrics:
        """Run the production TrackingPipeline with the full adaptive stack.

        Same record schema as the explicit loop; provenance records the
        harness so cross-method comparisons stay honest.
        """
        from fsoc_tracker.ai.adaptive_roi import AdaptiveROI
        from fsoc_tracker.ai.failure_predictor import FailurePredictor
        from fsoc_tracker.ai.mission import AIMissionBrain
        from fsoc_tracker.control.adaptive import AdaptiveController
        from fsoc_tracker.disturbances.config import get_preset_config
        from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
        from fsoc_tracker.pipeline.sources import VirtualSimulationSource
        from fsoc_tracker.simulation.communication import CommunicationEngine
        from fsoc_tracker.simulation.optical_link import OpticalLinkEngine
        from fsoc_tracker.simulation.terminal import TerminalState
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        from fsoc_tracker.tracking.search import SearchController

        method = METHOD_FULL_AI_MISSION
        metrics = BenchmarkMetrics(
            method=method,
            scenario=f"world_{config.world_profile}",
            seed=config.seed,
            source="simulation",
            repro_command=build_repro_command(config),
        )
        frame_rows: list[dict[str, Any]] = []
        self._running = True
        self._stop_requested = False

        def log(level: str, msg: str) -> None:
            if log_callback:
                log_callback(level, msg)

        try:
            components = load_method_components(method)
            brain = AIMissionBrain(
                learned_motion=components.get("learned_motion"),
                learned_policy=components.get("learned_policy"),
            )
            metrics.provenance = {
                "harness": "tracking_pipeline_full_stack",
                "adaptive_roi": True,
                "adaptive_kalman": True,
                "adaptive_controller": True,
                "failure_predictor": True,
                "search_controller": True,
                **components.get("provenance", {}),
            }

            wc = WorldConfig(width=2000.0, height=2000.0, random_seed=config.seed)
            engine = SimulationEngine(wc)
            try:
                from fsoc_tracker.simulation.world_builder import (
                    build_benchmark_world,
                )
                world = build_benchmark_world(config.world_profile,
                                              seed=config.seed)
                engine.load_scenario(world)
                metrics.scenario = world.name
            except Exception as e:
                log("WARN", f"World build failed: {e}")

            camera = VirtualCamera(CameraState(
                horizontal_fov_deg=4.0,
                vertical_fov_deg=3.0,
                width=640,
                height=480,
                max_pan_speed_deg_s=5.0,
                max_tilt_speed_deg_s=5.0,
            ))
            sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480))
            disturbance = None
            if config.disturbance_preset != "clear":
                disturbance = DisturbancePipeline(get_preset_config(config.disturbance_preset))
            from fsoc_tracker.disturbances.config import (
                DisturbanceConfig as _DisturbanceConfig,
            )
            from fsoc_tracker.disturbances.config import (
                apply_world_disturbances as _apply_world,
            )
            if disturbance is None:
                disturbance = DisturbancePipeline(_DisturbanceConfig())
            _apply_world(disturbance.config,
                         _WORLD_PROFILE_DISTURBANCES.get(
                             config.world_profile, {}),
                         seed=config.seed)
            if not disturbance.config.enabled:
                disturbance = None
            source = VirtualSimulationSource(
                engine, camera, sensor, disturbance, config.sim_dt,
                eval_sink=EvalSink(),
            )
            pipeline = TrackingPipeline(
                ai_brain=brain,
                adaptive_roi=AdaptiveROI(),
                failure_predictor=FailurePredictor(),
                adaptive_kalman=AdaptiveKalmanManager(),
                adaptive_controller=AdaptiveController(),
                search_controller=SearchController(),
            )
            pipeline.reset()
            pipeline.set_source(source)
            pipeline.start()

            link_engine = OpticalLinkEngine()
            comm = CommunicationEngine()
            errors: list[float] = []
            pred_errors: list[float] = []
            reacquisition_times: list[float] = []
            reacquisition_from_reappearance: list[float] = []
            reappear_s: float | None = None
            frame_times: list[float] = []
            fov_inside_count = 0
            fov_total_count = 0
            locked_time_s = 0.0
            loss_count = 0
            search_time_s = 0.0
            acquisition_s: float | None = None
            first_lock_seen = False
            was_locked = False
            loss_start_s: float | None = None
            safety_events = 0
            action_histogram: dict[str, int] = {}
            link_downtime_s = 0.0
            next_probe_s = 1.0
            probe_id = 0
            sim_time = 0.0

            try:
                while len(frame_rows) < config.max_frames:
                    if self._stop_requested:
                        log("INFO", "Benchmark stopped by user")
                        break
                    result = pipeline.step()
                    if result is None:
                        break
                    t_result = result.timestamp_s
                    dt = result.dt if result.dt > 0 else config.sim_dt
                    locked = bool(result.tracking and result.tracking.locked)
                    state_name = result.tracking.state.name if result.tracking else ""
                    if state_name in ("LOST", "SEARCHING"):
                        search_time_s += dt

                    error = result.error_px
                    gt = result.ground_truth
                    gt_visible = bool(
                        gt is not None and getattr(gt, "target_visible", False)
                    )
                    if error is not None and gt_visible:
                        errors.append(error)
                        fov_total_count += 1
                        if result.fov_inside:
                            fov_inside_count += 1

                    ai_action = ""
                    ai_situation = ""
                    ai_approved: Any = ""
                    decision = result.ai_decision
                    if decision is not None:
                        action = getattr(decision, "action", None)
                        ai_action = getattr(action, "value", str(action)) if action else ""
                        situation = getattr(decision, "situation", None)
                        ai_situation = (
                            getattr(situation, "value", str(situation)) if situation else ""
                        )
                        safety = getattr(decision, "safety", None)
                        ai_approved = bool(getattr(safety, "approved", True))
                        if action:
                            action_histogram[ai_action] = action_histogram.get(ai_action, 0) + 1
                        if not ai_approved:
                            safety_events += 1
                        prediction = getattr(decision, "prediction", None)
                        if (prediction is not None
                                and getattr(prediction, "method", "") == "learned_motion"
                                and gt_visible and error is not None
                                and result.tracking is not None):
                            pred_errors.append(float(np.hypot(
                                (result.tracking.estimated_x + prediction.mean_x_px)
                                - gt.target_pixel_x,
                                (result.tracking.estimated_y + prediction.mean_y_px)
                                - gt.target_pixel_y,
                            )))

                    if locked:
                        locked_time_s += dt
                        if not first_lock_seen:
                            first_lock_seen = True
                            acquisition_s = t_result
                        if loss_start_s is not None:
                            reacquisition_times.append(t_result - loss_start_s)
                            if reappear_s is not None:
                                reacquisition_from_reappearance.append(
                                    t_result - reappear_s)
                            loss_start_s = None
                            reappear_s = None
                    else:
                        if was_locked and loss_start_s is None:
                            loss_count += 1
                            loss_start_s = t_result
                            reappear_s = None
                        if (loss_start_s is not None and reappear_s is None
                                and gt_visible):
                            reappear_s = t_result
                    was_locked = locked

                    world_targets = engine.get_state().get_active_targets()
                    tb_active = len(world_targets) > 0
                    link = link_engine.update(
                        TerminalState(
                            x=1000.0, y=1000.0, z=50.0,
                            yaw_deg=camera.state.pan_deg,
                            pitch_deg=camera.state.tilt_deg,
                            active=True,
                        ),
                        TerminalState(
                            x=world_targets[0].x if tb_active else 0.0,
                            y=world_targets[0].y if tb_active else 0.0,
                            z=world_targets[0].z if tb_active else 0.0,
                            active=tb_active,
                        ),
                    )
                    if link.status.value in LINK_DOWN_STATES:
                        link_downtime_s += dt
                    while t_result >= next_probe_s:
                        probe = comm.create_message(
                            "TERM_A", "TERM_B", f"PROBE-{probe_id}", t_result
                        )
                        comm.send_message(probe)
                        probe_id += 1
                        next_probe_s += 1.0
                    comm.update(link, t_result)

                    frame_times.append(result.total_ms)
                    frame_rows.append({
                        "frame_index": result.frame_index,
                        "timestamp_s": round(t_result, 4),
                        "detected": bool(result.perception and result.perception.detected),
                        "confidence": round(result.perception.primary_detection.confidence, 3)
                        if result.perception and result.perception.primary_detection else 0.0,
                        "track_x": round(result.tracking.estimated_x, 2)
                        if result.tracking else 0.0,
                        "track_y": round(result.tracking.estimated_y, 2)
                        if result.tracking else 0.0,
                        "track_state": state_name,
                        "locked": locked,
                        "error_px": round(error, 2) if error is not None else "",
                        "gt_visible": gt_visible,
                        "fov_inside": bool(result.fov_inside),
                        "pan_deg": round(camera.state.pan_deg, 3),
                        "tilt_deg": round(camera.state.tilt_deg, 3),
                        "link_status": link.status.value,
                        "angular_error_deg": round(link.angular_error_deg, 3),
                        "alignment_pct": round(link.beam_alignment_percent, 1),
                        "link_quality_pct": round(link.link_quality_percent, 1),
                        "range_m": round(link.range_m, 1),
                        "ai_action": ai_action,
                        "ai_situation": ai_situation,
                        "ai_approved": ai_approved,
                        "processing_ms": round(result.total_ms, 3),
                    })
                    sim_time = t_result
                    if progress_callback:
                        progress_callback(len(frame_rows), config.max_frames)
            finally:
                pipeline.stop()

            metrics.frames_processed = len(frame_rows)
            metrics.total_frames = config.max_frames
            metrics.duration_s = sim_time
            metrics.target_losses = loss_count
            if errors:
                arr = np.array(errors)
                metrics.rmse_px = float(np.sqrt(np.mean(arr ** 2)))
                metrics.mae_px = float(np.mean(np.abs(arr)))
                metrics.p95_px = float(np.percentile(arr, 95))
                metrics.max_error_px = float(np.max(arr))
            if sim_time > 0:
                metrics.lock_retention_pct = min(locked_time_s / sim_time, 1.0) * 100.0
                metrics.loss_rate_pct = (1.0 - min(locked_time_s / sim_time, 1.0)) * 100.0
            metrics.acquisition_s = acquisition_s
            metrics.reacquisition_s = (
                float(np.mean(reacquisition_times)) if reacquisition_times else None
            )
            metrics.reacquisition_from_reappearance_s = (
                float(np.mean(reacquisition_from_reappearance))
                if reacquisition_from_reappearance else None
            )
            metrics.search_duration_s = search_time_s
            metrics.fov_retention_pct = (
                (fov_inside_count / fov_total_count * 100.0) if fov_total_count else 0.0
            )
            if frame_times:
                metrics.latency_ms = float(np.mean(frame_times))
                metrics.latency_p95_ms = float(np.percentile(frame_times, 95))
                metrics.fps = 1000.0 / max(float(np.mean(frame_times)), 1e-6)
            metrics.prediction_rmse_px = (
                float(np.sqrt(np.mean(np.array(pred_errors) ** 2))) if pred_errors else None
            )
            metrics.link_downtime_s = link_downtime_s
            metrics.messages_total = probe_id
            metrics.messages_delivered = sum(
                1 for m in comm.message_log if m.status.value == "DELIVERED"
            )
            metrics.completed = not self._stop_requested
            metrics.safety_events = safety_events
            metrics.safety_note = "safety vetoes counted from mission-brain approvals"
            metrics.ai_action_histogram = action_histogram
            metrics.link_note = "SIM closed loop: camera axis vs simulated beacon position"
            self._write_logs(metrics, frame_rows, config, log)
            log(
                "INFO",
                f"Benchmark complete: method={method} "
                f"RMSE={metrics.rmse_px:.2f}px FPS={metrics.fps:.1f}",
            )
        except Exception as e:
            log("ERROR", f"Benchmark failed: {e}")
            raise
        finally:
            self._running = False
        return metrics

    def _write_logs(
        self,
        metrics: BenchmarkMetrics,
        frame_rows: list[dict[str, Any]],
        config: BenchmarkRunConfig,
        log: LogCallback,
    ) -> None:
        from fsoc_tracker.benchmark.export import export_run_json
        try:
            json_path, csv_path = export_run_json(metrics, frame_rows, config.output_dir)
            metrics.log_json_path = json_path
            metrics.log_csv_path = csv_path
            log("INFO", f"Logs written: {json_path}, {csv_path}")
        except Exception as e:
            log("ERROR", f"Log write failed: {e}")
