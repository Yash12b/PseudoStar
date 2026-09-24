"""Application view state — read-only UI snapshot.

The GUI renders this state. Never read mutable subsystem objects
from widgets concurrently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SystemMode(str, Enum):
    SIMULATION = "simulation"
    VIDEO = "video"
    LIVE = "live"


class SystemState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    BENCHMARK = "BENCHMARK"
    ERROR = "ERROR"


class RunState(str, Enum):
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    STEP = "STEP"


@dataclass
class CameraViewState:
    image_width: int = 640
    image_height: int = 480
    hfov_deg: float = 4.0
    vfov_deg: float = 3.0
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    fps: float = 0.0
    timestamp_s: float = 0.0


@dataclass
class PerceptionState:
    backend: str = "classical"
    detected: bool = False
    confidence: float = 0.0
    detection_x: float = 0.0
    detection_y: float = 0.0
    num_candidates: int = 0
    processing_ms: float = 0.0
    model_status: str = ""


@dataclass
class TrackingStateView:
    state: str = "NO_TRACK"
    locked: bool = False
    estimated_x: float = 0.0
    estimated_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    uncertainty_x: float = 0.0
    uncertainty_y: float = 0.0
    residual: float = 0.0
    prediction_only: bool = False
    time_since_detection_s: float = 0.0


@dataclass
class ControlState:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    pan_command: float = 0.0
    tilt_command: float = 0.0
    pan_saturated: bool = False
    tilt_saturated: bool = False
    control_mode: str = "disabled"


@dataclass
class TargetView:
    target_id: int = -1
    visible: bool = False
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    size_px: float = 10.0
    trajectory_type: str = ""
    trail: list[tuple[float, ...]] = field(default_factory=list)


@dataclass
class DisturbanceView:
    enabled: bool = False
    profile: str = "clear"
    noise_type: str = "none"
    noise_sigma: float = 0.0
    fog: float = 0.0
    haze: float = 0.0
    rain: float = 0.0
    low_light: float = 0.0
    jitter_px: float = 0.0


@dataclass
class PerformanceView:
    pipeline_fps: float = 0.0
    processing_ms: float = 0.0
    perception_ms: float = 0.0
    tracking_ms: float = 0.0
    control_ms: float = 0.0
    frames_processed: int = 0
    wall_time_s: float = 0.0


@dataclass
class ErrorHistory:
    timestamps: list[float] = field(default_factory=list)
    errors_x: list[float] = field(default_factory=list)
    errors_y: list[float] = field(default_factory=list)
    errors_euclidean: list[float] = field(default_factory=list)
    fps_history: list[float] = field(default_factory=list)
    max_len: int = 300


@dataclass
class BenchmarkView:
    active: bool = False
    source_name: str = ""
    progress: float = 0.0
    frame_count: int = 0
    total_frames: int = 0
    acquisition_s: float | None = None
    rmse_px: float | None = None
    loss_rate: float | None = None
    reacq_s: float | None = None
    processing_fps: float | None = None


@dataclass
class SIHScorecard:
    acquisition_s: float | None = None
    acquisition_threshold: float = 2.0
    rmse_px: float | None = None
    rmse_threshold: float = 10.0
    loss_percent: float | None = None
    loss_threshold: float = 5.0
    reacq_s: float | None = None
    reacq_threshold: float = 1.0
    fps: float | None = None
    fps_threshold: float = 20.0
    lock_retention_pct: float | None = None


@dataclass
class EventLogEntry:
    timestamp_s: float = 0.0
    level: str = "INFO"
    message: str = ""


@dataclass
class AIState:
    situation: str = "normal_tracking"
    action: str = "use_roi"
    confidence: float = 0.0
    failure_risk: float = 0.0
    prediction_dx: float = 0.0
    prediction_dy: float = 0.0
    prediction_uncertainty_x: float = 0.0
    prediction_uncertainty_y: float = 0.0
    prediction_horizon_s: float = 0.1
    roi_mode: str = "fixed"
    roi_radius_px: float = 80.0
    model_status: str = "expert_only"
    fallback_active: bool = False
    explanation: str = ""
    ai_ms: float = 0.0

    # Search state
    search_active: bool = False
    search_phase: str = ""
    search_strategy: str = ""
    search_center_x: float = 0.0
    search_center_y: float = 0.0
    search_radius_px: float = 0.0
    search_frames: int = 0
    search_time_s: float = 0.0
    search_duration_s: float = 0.0
    search_strategy_used: str = ""


@dataclass
class TerminalView:
    terminal_id: str = ""
    active: bool = False
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    hfov_deg: float = 4.0
    vfov_deg: float = 3.0

    @property
    def optical_axis(self) -> tuple[float, float, float]:
        """Unit optical-axis vector from yaw/pitch.

        Single source of truth for Terminal A's beam direction in the
        GUI layer. Must match TerminalState.optical_axis convention:
        yaw=0/pitch=0 looks along +Z.
        """
        yaw_rad = math.radians(self.yaw_deg)
        pitch_rad = math.radians(self.pitch_deg)
        return (
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
        )

@dataclass
class OpticalLinkView:
    status: str = "NO_LINK"
    angular_error_deg: float = 0.0
    beam_alignment_percent: float = 0.0
    range_m: float = 0.0
    link_quality_percent: float = 0.0

@dataclass
class MessageView:
    message_id: int = 0
    source: str = ""
    destination: str = ""
    timestamp_s: float = 0.0
    payload: str = ""
    status: str = ""

@dataclass
class ApplicationViewState:
    system_mode: SystemMode = SystemMode.SIMULATION
    system_state: SystemState = SystemState.IDLE
    run_state: RunState = RunState.STOPPED

    camera: CameraViewState = field(default_factory=CameraViewState)
    perception: PerceptionState = field(default_factory=PerceptionState)
    tracking: TrackingStateView = field(default_factory=TrackingStateView)
    control: ControlState = field(default_factory=ControlState)
    target: TargetView = field(default_factory=TargetView)
    disturbances: DisturbanceView = field(default_factory=DisturbanceView)
    performance: PerformanceView = field(default_factory=PerformanceView)
    errors: ErrorHistory = field(default_factory=ErrorHistory)
    benchmark: BenchmarkView = field(default_factory=BenchmarkView)
    scorecard: SIHScorecard = field(default_factory=SIHScorecard)
    ai_state: AIState = field(default_factory=AIState)

    session_id: str = ""
    elapsed_s: float = 0.0
    source_info: str = ""
    events: list[EventLogEntry] = field(default_factory=list)

    # Handoff readiness: coarse alignment stable enough for a future
    # fine-pointing stage (GT-free: lock + low uncertainty/residual held
    # for a consecutive streak). Set by the worker every frame.
    handoff_ready: bool = False
    handoff_stable_s: float = 0.0

    terminal_a: TerminalView = field(default_factory=TerminalView)
    terminal_b: TerminalView = field(default_factory=TerminalView)
    optical_link: OpticalLinkView = field(default_factory=OpticalLinkView)
    messages: list[MessageView] = field(default_factory=list)
    targets_all: list[TargetView] = field(default_factory=list)

    # Separate concepts
    selected_object_id: int | None = None
    selected_terminal_a: bool = False
    designated_beacon_id: int | None = None
    tracked_target_id: int | None = None
    terminal_a_id: str = "TERM_A"
    terminal_b_id: str = "TERM_B"

    layout: str = "IMMERSIVE EXPLORER"
    camera_mode: str = "FREE"

    show_ground_truth: bool = False
    show_debug_overlay: bool = True
    show_trail: bool = True
    view_mode: str = "raw"
    # Camera boresight aim-point history (world x,y,z), oldest first.
    # Bounded in the worker; rendered as the search sweep trail.
    camera_trail: list[tuple[float, float, float]] = field(default_factory=list)

    # Beacon AI state
    scan_active: bool = False
    scan_radius: float = 0.0
    scan_beacons_found: int = 0
    beacon_connected: bool = False
    beacon_distance_m: float = 0.0
    beacon_azimuth_deg: float = 0.0
    beacon_elevation_deg: float = 0.0
    beacon_speed_ms: float = 0.0
    beacon_predicted_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    link_locked: bool = False
    atmospheric_disturbance: str = "clear"
    atmospheric_visibility_km: float = 10.0

    # Beacon drag state
    dragging_beacon: bool = False
    drag_beacon_id: int | None = None

    # Video source metadata (MP4 mode)
    video_total_frames: int = 0
    video_duration_s: float = 0.0
    video_fps: float = 0.0
    video_current_frame: int = 0
    video_timestamp_s: float = 0.0
    video_processing_fps: float = 0.0
    video_is_equirectangular: bool = False
    video_distance_status: str = "N/A"  # "N/A" for monocular, "ESTIMATED" if depth available

    # Live camera metadata
    live_camera_id: int = 0
    live_camera_name: str = ""
    live_available_cameras: list[dict] = field(default_factory=list)
    live_resolution: str = ""
    live_fps: float = 0.0
    live_camera_status: str = "unavailable"  # "available", "unavailable", "error"
    live_error_message: str = ""
    live_distance_status: str = "UNAVAILABLE"  # "UNAVAILABLE", "ESTIMATED", "OBSERVED"

    # Pending generic world config for a live world switch (replaces the
    # retired scene-ID mechanism). Holds a ScenarioConfig or None.
    _pending_world_config: Any | None = None

    @property
    def error_history(self) -> ErrorHistory:
        return self.errors

    def add_event(self, message: str, level: str = "INFO") -> None:
        self.events.append(EventLogEntry(
            timestamp_s=self.elapsed_s, level=level, message=message,
        ))
        if len(self.events) > 500:
            self.events = self.events[-500:]