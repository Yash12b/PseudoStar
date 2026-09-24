"""Intelligent search and reacquisition.

Five bounded search strategies when tracking is lost:

1. LOCAL_LAST_KNOWN — search around last known position + velocity projection
2. UNCERTAINTY_REGION — expand search proportional to Kalman uncertainty
3. PREDICTIVE_SEARCH — extrapolate along estimated velocity vector
4. EXPANDING_SPIRAL — Archimedean spiral from last known position
5. GLOBAL_SWEEP — full-frame scan when all else fails

Progression: LOCAL → UNCERTAINTY → PREDICTIVE → SPIRAL → GLOBAL
Each phase has a configurable frame timeout.

NO ground truth is used. All inputs are runtime-observable.
Camera motion is bounded by safety limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


class SearchStrategy(Enum):
    """The 5 search strategies."""
    LOCAL_LAST_KNOWN = "local_last_known"
    UNCERTAINTY_REGION = "uncertainty_region"
    PREDICTIVE_SEARCH = "predictive_search"
    EXPANDING_SPIRAL = "expanding_spiral"
    GLOBAL_SWEEP = "global_sweep"


class SearchPhase(Enum):
    """Internal phase tracking (maps to strategies)."""
    LOCAL_LAST_KNOWN = auto()
    UNCERTAINTY_REGION = auto()
    PREDICTIVE_SEARCH = auto()
    EXPANDING_SPIRAL = auto()
    GLOBAL_SWEEP = auto()
    COMPLETE = auto()


@dataclass
class SearchState:
    """Current search state and diagnostics."""
    phase: SearchPhase = SearchPhase.LOCAL_LAST_KNOWN
    strategy: SearchStrategy = SearchStrategy.LOCAL_LAST_KNOWN
    search_center_x: float = 0.0
    search_center_y: float = 0.0
    search_radius_px: float = 0.0
    search_direction_rad: float = 0.0
    frames_in_search: int = 0
    total_search_time_s: float = 0.0
    pattern_angle_rad: float = 0.0
    pattern_step: int = 0
    cell_frames: int = 0
    spiral_radius_px: float = 0.0
    detection_during_search: bool = False
    reacquired: bool = False

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "phase": self.phase.name,
            "center": [round(self.search_center_x, 1), round(self.search_center_y, 1)],
            "radius_px": round(self.search_radius_px, 1),
            "frames": self.frames_in_search,
            "time_s": round(self.total_search_time_s, 3),
            "step": self.pattern_step,
            "cell_frames": self.cell_frames,
            "reacquired": self.reacquired,
        }


@dataclass
class SearchConfig:
    """Configuration for search strategy."""
    # Phase timeouts (frames)
    local_timeout_frames: int = 15
    uncertainty_timeout_frames: int = 20
    predictive_timeout_frames: int = 20
    spiral_timeout_frames: int = 40
    global_timeout_frames: int = 60

    # Local search
    local_radius_px: float = 50.0
    velocity_projection_s: float = 0.3

    # Uncertainty region
    uncertainty_scale: float = 3.0
    min_uncertainty_radius_px: float = 30.0
    max_uncertainty_radius_px: float = 250.0

    # Predictive search
    predictive_horizon_s: float = 0.5
    predictive_radius_px: float = 80.0

    # Spiral (Archimedean, constant linear speed along the path).
    # Pitch matches one viewport height minus overlap per revolution;
    # the per-frame angle step dθ = S/sqrt(b²+r²) sweeps equal path
    # length per frame, unlike fixed angular steps whose outer loops
    # race ahead (uniform sky coverage at any mount rate limit).
    spiral_max_radius_px: float = 250.0
    spiral_pitch_px: float = 336.0
    spiral_arc_step_px: float = 12.0

    # Global sweep
    global_sweep_rows: int = 4
    global_sweep_cols: int = 5
    # Half-span of the global raster in degrees around the search origin.
    # The sweep must cover real sky (not just the current frame), so the
    # span is much wider than one FOV.
    sweep_span_deg: float = 30.0
    # Dwell: advance to the next raster cell once the mount is within
    # this tolerance of the current cell target (a fixed frame cadence
    # outruns the 5 deg/s mount, so the sweep would jitter around the
    # origin forever instead of covering sky).
    global_arrive_tol_deg: float = 2.0
    # Backstop: advance anyway after this many frames on one cell, so a
    # cell can never stall the sweep.
    global_cell_dwell_max_frames: int = 90

    # Safety
    # Sweep-cycle length: on expiry the sweep recycles at EXPANDING_SPIRAL
    # (never parks the camera — a static SEARCHING display is a failure).
    max_search_duration_s: float = 5.0
    max_pan_rate_deg_s: float = 5.0
    max_tilt_rate_deg_s: float = 5.0

    # Image dimensions
    image_width: int = 640
    image_height: int = 480

    @property
    def phase_timeouts(self) -> dict[SearchPhase, int]:
        return {
            SearchPhase.LOCAL_LAST_KNOWN: self.local_timeout_frames,
            SearchPhase.UNCERTAINTY_REGION: self.uncertainty_timeout_frames,
            SearchPhase.PREDICTIVE_SEARCH: self.predictive_timeout_frames,
            SearchPhase.EXPANDING_SPIRAL: self.spiral_timeout_frames,
            SearchPhase.GLOBAL_SWEEP: self.global_timeout_frames,
        }


class SearchController:
    """Manages intelligent search when tracking is lost.

    Search progression:
        LOCAL_LAST_KNOWN → UNCERTAINTY_REGION → PREDICTIVE_SEARCH
        → EXPANDING_SPIRAL → GLOBAL_SWEEP

    Each phase has a configurable frame timeout. Camera motion commands
    are bounded by safety limits. NO ground truth is used.
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self._config = config or SearchConfig()
        self._state = SearchState()
        self._last_known_x: float = 0.0
        self._last_known_y: float = 0.0
        self._last_known_vx: float = 0.0
        self._last_known_vy: float = 0.0
        self._last_known_uncertainty_x: float = 5.0
        self._last_known_uncertainty_y: float = 5.0
        self._lost_timestamp_s: float = 0.0
        self._current_pan_deg: float = 0.0
        self._current_tilt_deg: float = 0.0
        # Search origin: mount orientation when the search began. All
        # phase waypoints are computed relative to this origin so the
        # sweep covers actual sky instead of collapsing onto boresight.
        self._origin_pan_deg: float = 0.0
        self._origin_tilt_deg: float = 0.0
        self._pan_rate_deg_s: float = 0.0
        self._tilt_rate_deg_s: float = 0.0

    @property
    def state(self) -> SearchState:
        return self._state

    @property
    def pan_rate_deg_s(self) -> float:
        """Current pan rate command from search (bounded)."""
        return self._pan_rate_deg_s

    @property
    def tilt_rate_deg_s(self) -> float:
        """Current tilt rate command from search (bounded)."""
        return self._tilt_rate_deg_s

    def begin_search(
        self,
        last_known_x: float,
        last_known_y: float,
        velocity_x: float,
        velocity_y: float,
        timestamp_s: float,
        uncertainty_x: float = 5.0,
        uncertainty_y: float = 5.0,
        current_pan_deg: float = 0.0,
        current_tilt_deg: float = 0.0,
    ) -> SearchState:
        """Begin a new search from last known target state."""
        self._last_known_x = last_known_x
        self._last_known_y = last_known_y
        self._last_known_vx = velocity_x
        self._last_known_vy = velocity_y
        self._last_known_uncertainty_x = max(1.0, uncertainty_x)
        self._last_known_uncertainty_y = max(1.0, uncertainty_y)
        self._lost_timestamp_s = timestamp_s
        self._current_pan_deg = current_pan_deg
        self._current_tilt_deg = current_tilt_deg
        self._origin_pan_deg = current_pan_deg
        self._origin_tilt_deg = current_tilt_deg

        # Phase 1: LOCAL_LAST_KNOWN
        proj_x = last_known_x + velocity_x * self._config.velocity_projection_s
        proj_y = last_known_y + velocity_y * self._config.velocity_projection_s

        self._state = SearchState(
            phase=SearchPhase.LOCAL_LAST_KNOWN,
            strategy=SearchStrategy.LOCAL_LAST_KNOWN,
            search_center_x=proj_x,
            search_center_y=proj_y,
            search_radius_px=self._config.local_radius_px,
            search_direction_rad=math.atan2(velocity_y, velocity_x),
            frames_in_search=0,
            total_search_time_s=0.0,
            pattern_step=0,
        )
        return self._state

    def update(
        self,
        dt: float,
        current_pan_deg: float = 0.0,
        current_tilt_deg: float = 0.0,
    ) -> SearchState:
        """Update search state each frame. Returns camera pan/tilt rates."""
        self._state.frames_in_search += 1
        self._state.total_search_time_s += dt
        self._current_pan_deg = current_pan_deg
        self._current_tilt_deg = current_tilt_deg

        cfg = self._config
        s = self._state

        # Recycle on expiry: a lost beacon may reappear at any time, so
        # the sweep restarts at EXPANDING_SPIRAL instead of parking the
        # camera with zero rates forever (a static "SEARCHING" display
        # with no camera motion is a system failure, not a search).
        if s.total_search_time_s > cfg.max_search_duration_s:
            s.phase = SearchPhase.EXPANDING_SPIRAL
            s.strategy = SearchStrategy.EXPANDING_SPIRAL
            s.frames_in_search = 0
            # pattern_step preserved (spiral angle continues; zeroing it
            # would restart the global raster every cycle).
            s.pattern_angle_rad = 0.0
            s.search_center_x = self._last_known_x
            s.search_center_y = self._last_known_y
            s.search_radius_px = cfg.spiral_max_radius_px
            s.spiral_radius_px = 0.0
            s.total_search_time_s = 0.0

        # Check phase timeout
        timeout = cfg.phase_timeouts.get(s.phase, 30)
        if s.frames_in_search >= timeout:
            self._advance_phase()

        # Compute camera motion for current phase
        self._compute_search_motion(dt)

        return s

    def get_search_region(self) -> tuple[int, int, int, int]:
        """Get current search bounding box (x1, y1, x2, y2)."""
        s = self._state
        r = s.search_radius_px
        cx, cy = s.search_center_x, s.search_center_y

        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(self._config.image_width, int(cx + r))
        y2 = min(self._config.image_height, int(cy + r))

        return (x1, y1, x2, y2)

    def process_detection(
        self,
        det_x: float,
        det_y: float,
        confidence: float,
        timestamp_s: float,
    ) -> bool:
        """Process a detection found during search.

        Returns True if detection is accepted as valid reacquisition.
        """
        if confidence < 0.2:
            return False

        s = self._state
        dist = math.sqrt(
            (det_x - s.search_center_x) ** 2
            + (det_y - s.search_center_y) ** 2
        )

        # Relaxed acceptance in later phases
        max_dist = s.search_radius_px * 1.5
        if s.phase == SearchPhase.GLOBAL_SWEEP:
            max_dist = float(self._config.image_width)
        elif s.phase == SearchPhase.EXPANDING_SPIRAL:
            max_dist = s.search_radius_px * 2.0

        if dist > max_dist:
            return False

        # Velocity consistency check (only early in search)
        if s.frames_in_search < 5 and self._lost_timestamp_s > 0:
            elapsed = max(timestamp_s - self._lost_timestamp_s, 0.001)
            det_speed = math.sqrt(
                (det_x - self._last_known_x) ** 2
                + (det_y - self._last_known_y) ** 2
            ) / elapsed
            vel_mag = math.sqrt(self._last_known_vx ** 2 + self._last_known_vy ** 2)
            if vel_mag > 0 and det_speed > vel_mag * 5.0:
                return False

        s.detection_during_search = True
        s.reacquired = True
        return True

    def _advance_phase(self) -> None:
        """Advance to next search phase."""
        s = self._state
        cfg = self._config
        w, h = cfg.image_width, cfg.image_height

        phase_order = [
            SearchPhase.LOCAL_LAST_KNOWN,
            SearchPhase.UNCERTAINTY_REGION,
            SearchPhase.PREDICTIVE_SEARCH,
            SearchPhase.EXPANDING_SPIRAL,
            SearchPhase.GLOBAL_SWEEP,
        ]
        strategy_map = {
            SearchPhase.LOCAL_LAST_KNOWN: SearchStrategy.LOCAL_LAST_KNOWN,
            SearchPhase.UNCERTAINTY_REGION: SearchStrategy.UNCERTAINTY_REGION,
            SearchPhase.PREDICTIVE_SEARCH: SearchStrategy.PREDICTIVE_SEARCH,
            SearchPhase.EXPANDING_SPIRAL: SearchStrategy.EXPANDING_SPIRAL,
            SearchPhase.GLOBAL_SWEEP: SearchStrategy.GLOBAL_SWEEP,
        }

        idx = phase_order.index(s.phase) if s.phase in phase_order else len(phase_order) - 1
        next_idx = min(idx + 1, len(phase_order) - 1)
        next_phase = phase_order[next_idx]

        s.phase = next_phase
        s.strategy = strategy_map[next_phase]
        s.frames_in_search = 0
        # NOTE: pattern_step is deliberately NOT reset here (except for
        # the spiral branch below): the global raster must keep its progress
        # across phase visits, otherwise the sweep restarts at cell 0 every
        # visit and never covers the sky.
        s.pattern_angle_rad = 0.0
        s.cell_frames = 0

        # Set initial state for each phase
        if next_phase == SearchPhase.UNCERTAINTY_REGION:
            unc_r = max(
                cfg.min_uncertainty_radius_px,
                min(
                    cfg.max_uncertainty_radius_px,
                    cfg.uncertainty_scale * max(
                        self._last_known_uncertainty_x,
                        self._last_known_uncertainty_y,
                    ),
                ),
            )
            s.search_center_x = self._last_known_x
            s.search_center_y = self._last_known_y
            s.search_radius_px = unc_r

        elif next_phase == SearchPhase.PREDICTIVE_SEARCH:
            proj_x = self._last_known_x + self._last_known_vx * cfg.predictive_horizon_s
            proj_y = self._last_known_y + self._last_known_vy * cfg.predictive_horizon_s
            s.search_center_x = proj_x
            s.search_center_y = proj_y
            s.search_radius_px = cfg.predictive_radius_px
            s.search_direction_rad = math.atan2(
                self._last_known_vy, self._last_known_vx
            )

        elif next_phase == SearchPhase.EXPANDING_SPIRAL:
            s.search_center_x = self._last_known_x
            s.search_center_y = self._last_known_y
            s.search_radius_px = cfg.spiral_max_radius_px
            s.spiral_radius_px = 0.0
            # pattern_step intentionally preserved (spiral angle continues;
            # zeroing it here would restart the global raster every cycle).

        elif next_phase == SearchPhase.GLOBAL_SWEEP:
            s.search_center_x = w / 2.0
            s.search_center_y = h / 2.0
            s.search_radius_px = max(w, h)
            s.cell_frames = 0

    def _compute_search_motion(self, dt: float) -> None:
        """Compute bounded pan/tilt rate commands for current search phase."""
        cfg = self._config
        s = self._state

        pan_rate = 0.0
        tilt_rate = 0.0

        if s.phase == SearchPhase.LOCAL_LAST_KNOWN:
            # Slow pan toward projected position
            target_pan = self._pan_from_image_x(s.search_center_x)
            target_tilt = self._tilt_from_image_y(s.search_center_y)
            pan_rate = self._clamp_rate(target_pan - self._current_pan_deg, cfg.max_pan_rate_deg_s)
            tilt_rate = self._clamp_rate(target_tilt - self._current_tilt_deg, cfg.max_tilt_rate_deg_s)

        elif s.phase == SearchPhase.UNCERTAINTY_REGION:
            # Slow sweep within uncertainty circle
            angle = s.total_search_time_s * 0.5  # slow rotation
            sweep_pan = self._pan_from_image_x(
                s.search_center_x + s.search_radius_px * 0.5 * math.cos(angle)
            )
            sweep_tilt = self._tilt_from_image_y(
                s.search_center_y + s.search_radius_px * 0.5 * math.sin(angle)
            )
            pan_rate = self._clamp_rate(sweep_pan - self._current_pan_deg, cfg.max_pan_rate_deg_s)
            tilt_rate = self._clamp_rate(sweep_tilt - self._current_tilt_deg, cfg.max_tilt_rate_deg_s)

        elif s.phase == SearchPhase.PREDICTIVE_SEARCH:
            # Pan along velocity vector
            target_pan = self._pan_from_image_x(s.search_center_x)
            target_tilt = self._tilt_from_image_y(s.search_center_y)
            pan_rate = self._clamp_rate(target_pan - self._current_pan_deg, cfg.max_pan_rate_deg_s)
            tilt_rate = self._clamp_rate(target_tilt - self._current_tilt_deg, cfg.max_tilt_rate_deg_s)

        elif s.phase == SearchPhase.EXPANDING_SPIRAL:
            # Constant-linear-speed spiral position.
            b = cfg.spiral_pitch_px / (2.0 * math.pi)
            r_now = min(cfg.spiral_max_radius_px, b * s.pattern_angle_rad)
            dtheta = cfg.spiral_arc_step_px / max(
                math.sqrt(b * b + r_now * r_now), 1.0)
            s.pattern_angle_rad += dtheta
            s.spiral_radius_px = min(
                cfg.spiral_max_radius_px, b * s.pattern_angle_rad)
            spiral_x = (s.search_center_x
                        + s.spiral_radius_px * math.cos(s.pattern_angle_rad))
            spiral_y = (s.search_center_y
                        + s.spiral_radius_px * math.sin(s.pattern_angle_rad))
            target_pan = self._pan_from_image_x(spiral_x)
            target_tilt = self._tilt_from_image_y(spiral_y)
            pan_rate = self._clamp_rate(target_pan - self._current_pan_deg, cfg.max_pan_rate_deg_s)
            tilt_rate = self._clamp_rate(target_tilt - self._current_tilt_deg, cfg.max_tilt_rate_deg_s)
            s.pattern_step += 1

        elif s.phase == SearchPhase.GLOBAL_SWEEP:
            # Raster scan over real sky: grid cells map across
            # +/- sweep_span_deg around the search origin. Cells advance
            # on mount arrival (dwell), never on a fixed cadence, with a
            # per-cell frame cap as backstop.
            total_cells = cfg.global_sweep_rows * cfg.global_sweep_cols
            cell = s.pattern_step % total_cells
            row = cell // cfg.global_sweep_cols
            col = cell % cfg.global_sweep_cols
            span = cfg.sweep_span_deg
            target_pan = self._origin_pan_deg + (
                (col + 0.5) / cfg.global_sweep_cols * 2.0 - 1.0) * span
            target_tilt = self._origin_tilt_deg + (
                (row + 0.5) / cfg.global_sweep_rows * 2.0 - 1.0) * span * 0.5
            pan_rate = self._clamp_rate(target_pan - self._current_pan_deg, cfg.max_pan_rate_deg_s * 0.8)
            tilt_rate = self._clamp_rate(target_tilt - self._current_tilt_deg, cfg.max_tilt_rate_deg_s * 0.8)
            s.cell_frames += 1
            arrived = (abs(target_pan - self._current_pan_deg) <= cfg.global_arrive_tol_deg
                       and abs(target_tilt - self._current_tilt_deg) <= cfg.global_arrive_tol_deg)
            if arrived or s.cell_frames >= cfg.global_cell_dwell_max_frames:
                s.pattern_step += 1
                s.cell_frames = 0
            if s.pattern_step >= total_cells * 1000:
                # Wrap the ever-growing step to keep numbers small; the
                # modulo above keeps the raster itself cycling forever.
                s.pattern_step %= total_cells

        self._pan_rate_deg_s = pan_rate
        self._tilt_rate_deg_s = tilt_rate

    def _pan_from_image_x(self, x_px: float) -> float:
        """Convert image x pixel to an absolute camera pan target.

        Pixel offsets are measured from frame center (boresight) and
        added to the search-origin orientation, so waypoints track real
        sky positions instead of collapsing onto pan=0.
        """
        w = self._config.image_width
        return (self._origin_pan_deg
                + (x_px - w / 2.0) / w * self._fov_h_deg)

    def _tilt_from_image_y(self, y_px: float) -> float:
        """Convert image y pixel to an absolute camera tilt target."""
        h = self._config.image_height
        return (self._origin_tilt_deg
                - (y_px - h / 2.0) / h * self._fov_v_deg)

    @property
    def _fov_h_deg(self) -> float:
        return 4.0  # PS spec

    @property
    def _fov_v_deg(self) -> float:
        return 3.0  # PS spec

    @staticmethod
    def _clamp_rate(rate: float, max_rate: float) -> float:
        return max(-max_rate, min(max_rate, rate))

    def reset(self) -> None:
        self._state = SearchState()
        self._lost_timestamp_s = 0.0
        self._origin_pan_deg = 0.0
        self._origin_tilt_deg = 0.0
        self._pan_rate_deg_s = 0.0
        self._tilt_rate_deg_s = 0.0
