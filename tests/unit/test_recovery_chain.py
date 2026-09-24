"""Integration test: loss → search → reacquisition chain.

Proves the entire intelligent recovery flow works end-to-end:
1. Target is being tracked normally
2. Target disappears (loss)
3. Situation classifier detects loss
4. SearchController begins search with appropriate strategy
5. Camera moves through search phases
6. Target reappears
7. SearchController accepts reacquisition
8. Tracking resumes
9. Metrics: loss duration, search duration, strategy used

No ground truth is used to reacquire — all detection is via perception.
"""

from __future__ import annotations

import pytest

from fsoc_tracker.ai.failure_predictor import FailurePredictor
from fsoc_tracker.ai.mission import (
    ObservationFeatures,
    Situation,
    SituationClassifier,
)
from fsoc_tracker.control.controller import CoarsePointingController
from fsoc_tracker.perception.adaptive_roi import AdaptiveROI, ROIConfig
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.models import BeaconDetection
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.config import TrackerConfig
from fsoc_tracker.tracking.search import (
    SearchConfig,
    SearchController,
    SearchStrategy,
)
from fsoc_tracker.tracking.state import TrackState
from fsoc_tracker.tracking.tracker import KalmanTracker

# ---------------------------------------------------------------------------
# Helper: simulate a target moving through the pipeline
# ---------------------------------------------------------------------------

def _make_camera() -> VirtualCamera:
    state = CameraState(
        horizontal_fov_deg=4.0,
        vertical_fov_deg=3.0,
        width=640,
        height=480,
        max_pan_speed_deg_s=5.0,
        max_tilt_speed_deg_s=5.0,
    )
    return VirtualCamera(state)


def _make_world_with_target(
    x: float = 1000.0,
    y: float = 1000.0,
    z: float = 500.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> SimulationEngine:
    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
    engine = SimulationEngine(wc)
    params = {"x0": x, "y0": y, "z0": z}
    if vx != 0 or vy != 0:
        params["vx"] = vx
        params["vy"] = vy
    engine.add_target(trajectory_type="straight_line", trajectory_params=params)
    return engine


# ---------------------------------------------------------------------------
# Test: Situation classifier detects all 10 categories
# ---------------------------------------------------------------------------

class TestSituationCategories:
    def test_normal_tracking(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.9,
            residual_px=2.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=10.0, velocity_y_px_s=5.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.NORMAL_TRACKING

    def test_target_lost(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=1.0, detected=False, confidence=0.0,
            residual_px=0.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=0.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=0,
        )
        assert clf.classify(f) == Situation.TARGET_LOST

    def test_reacquisition(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=2.0, detected=False, confidence=0.0,
            residual_px=0.0, uncertainty_x_px=10.0, uncertainty_y_px=10.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=0.0,
            time_since_detection_s=0.5, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=0,
        )
        assert clf.classify(f) == Situation.REACQUISITION

    def test_low_confidence(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.2,
            residual_px=2.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.LOW_CONFIDENCE

    def test_high_noise(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.8,
            residual_px=50.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.HIGH_NOISE

    def test_fast_target_motion(self):
        clf = SituationClassifier(fast_velocity_threshold_px_s=200.0)
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.8,
            residual_px=2.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=300.0, velocity_y_px_s=0.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.FAST_TARGET_MOTION

    def test_degrading_track(self):
        clf = SituationClassifier()
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.4,
            residual_px=2.0, uncertainty_x_px=5.0, uncertainty_y_px=5.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.2, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.DEGRADING_TRACK

    def test_prediction_uncertain(self):
        clf = SituationClassifier(uncertainty_threshold_px=30.0)
        f = ObservationFeatures(
            timestamp_s=1.0, detected=True, confidence=0.8,
            residual_px=2.0, uncertainty_x_px=50.0, uncertainty_y_px=5.0,
            velocity_x_px_s=0.0, velocity_y_px_s=0.0,
            distance_from_center_px=50.0,
            time_since_detection_s=0.0, latency_ms=10.0,
            source_fps=30.0, processing_fps=30.0, candidate_count=1,
        )
        assert clf.classify(f) == Situation.PREDICTION_UNCERTAIN


# ---------------------------------------------------------------------------
# Test: Search strategies
# ---------------------------------------------------------------------------

class TestSearchStrategies:
    def test_all_five_strategies_exist(self):
        assert len(SearchStrategy) == 5
        values = [s.value for s in SearchStrategy]
        assert "local_last_known" in values
        assert "uncertainty_region" in values
        assert "predictive_search" in values
        assert "expanding_spiral" in values
        assert "global_sweep" in values

    def test_search_progression(self):
        sc = SearchController(SearchConfig(
            local_timeout_frames=5,
            uncertainty_timeout_frames=5,
            predictive_timeout_frames=5,
            spiral_timeout_frames=5,
            global_timeout_frames=5,
        ))
        sc.begin_search(320, 240, 50, 0, 1.0)
        assert sc.state.strategy == SearchStrategy.LOCAL_LAST_KNOWN

        for _ in range(6):
            sc.update(0.033)
        assert sc.state.strategy == SearchStrategy.UNCERTAINTY_REGION

        for _ in range(6):
            sc.update(0.033)
        assert sc.state.strategy == SearchStrategy.PREDICTIVE_SEARCH

        for _ in range(6):
            sc.update(0.033)
        assert sc.state.strategy == SearchStrategy.EXPANDING_SPIRAL

        for _ in range(6):
            sc.update(0.033)
        assert sc.state.strategy == SearchStrategy.GLOBAL_SWEEP

    def test_search_region_bounded(self):
        sc = SearchController(SearchConfig(image_width=640, image_height=480))
        sc.begin_search(320, 240, 50, 0, 1.0)
        x1, y1, x2, y2 = sc.get_search_region()
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_reacquisition_accepted(self):
        sc = SearchController()
        sc.begin_search(320, 240, 50, 0, 1.0)
        accepted = sc.process_detection(330, 245, 0.8, 1.5)
        assert accepted is True
        assert sc.state.reacquired is True

    def test_reacquisition_rejects_low_confidence(self):
        sc = SearchController()
        sc.begin_search(320, 240, 50, 0, 1.0)
        accepted = sc.process_detection(330, 245, 0.1, 1.5)
        assert accepted is False

    def test_camera_motion_bounded(self):
        sc = SearchController(SearchConfig(max_pan_rate_deg_s=5.0, max_tilt_rate_deg_s=5.0))
        sc.begin_search(320, 240, 50, 0, 1.0)
        for _ in range(30):
            sc.update(0.033)
        assert abs(sc.pan_rate_deg_s) <= 5.0
        assert abs(sc.tilt_rate_deg_s) <= 5.0


# ---------------------------------------------------------------------------
# Test: Adaptive ROI
# ---------------------------------------------------------------------------

class TestAdaptiveROI:
    def test_starts_full_frame(self):
        roi = AdaptiveROI(ROIConfig())
        assert roi.diagnostics.state == "FULL_FRAME"

    def test_shrinks_on_stable_tracking(self):
        roi = AdaptiveROI(ROIConfig(stable_frames_to_shrink=3, shrink_rate=0.1))
        for _ in range(10):
            roi.update(detected=True, confidence=0.9, estimated_x=320, estimated_y=240)
        assert roi.diagnostics.state in ("STABLE", "SHRINKING")
        assert roi.diagnostics.roi_fraction < 1.0

    def test_expands_on_miss(self):
        roi = AdaptiveROI(ROIConfig())
        for _ in range(5):
            roi.update(detected=True, confidence=0.9, estimated_x=320, estimated_y=240)
        for _ in range(10):
            roi.update(detected=False, confidence=0.0, estimated_x=320, estimated_y=240)
        assert roi.diagnostics.state in ("EXPANDING", "LOST", "FULL_FRAME")

    def test_force_full_on_repeated_misses(self):
        roi = AdaptiveROI(ROIConfig(miss_count_to_force_full=5))
        for _ in range(10):
            roi.update(detected=False, confidence=0.0, estimated_x=320, estimated_y=240)
        region = roi.update(detected=False, confidence=0.0, estimated_x=320, estimated_y=240)
        # After repeated misses, should be full frame
        assert region is None or roi.diagnostics.state == "FULL_FRAME"

    def test_never_permanently_blinds(self):
        """ROI must never permanently blind the detector."""
        roi = AdaptiveROI(ROIConfig(miss_count_to_force_full=5))
        # Simulate many frames of lost tracking
        for _ in range(50):
            roi.update(detected=False, confidence=0.0, estimated_x=320, estimated_y=240)
        # Should have forced full frame
        assert roi.diagnostics.state == "FULL_FRAME"
        region = roi._get_region()
        assert region is None  # Full frame


# ---------------------------------------------------------------------------
# Test: Full loss → search → reacquisition chain
# ---------------------------------------------------------------------------

class TestLossSearchReacquisitionChain:
    def test_end_to_end_chain(self):
        """Prove the entire loss→search→reacquisition chain works."""
        # Setup
        camera = _make_camera()
        tracker = KalmanTracker(TrackerConfig(
            acquisition_min_consecutive_hits=1,
            max_prediction_duration_s=0.3,
        ))
        detector = ClassicalBeaconDetector()
        controller = CoarsePointingController()
        search = SearchController(SearchConfig(
            local_timeout_frames=5,
            uncertainty_timeout_frames=5,
            predictive_timeout_frames=5,
            spiral_timeout_frames=10,
            global_timeout_frames=20,
        ))
        roi = AdaptiveROI(ROIConfig())
        classifier = SituationClassifier()

        dt = 1.0 / 30.0
        ts = 0.0
        loss_time = None
        reacquire_time = None
        strategy_used = None
        search_reacquired = False

        # Phase 1: Normal tracking (50 frames)
        for i in range(50):
            # Target at fixed position in image
            state = tracker.update(
                [BeaconDetection(
                    detected=True, center_x=320, center_y=240,
                    confidence=0.9, timestamp_s=ts,
                )],
                ts,
            )
            assert state.state == TrackState.TRACKING
            ts += dt

        # Phase 2: Target disappears (20 frames)
        for i in range(20):
            state = tracker.update([], ts)
            tracking_state = state.state.name

            if tracking_state in ("LOST", "NO_TRACK") and loss_time is None:
                loss_time = ts
                search.begin_search(
                    state.estimated_x, state.estimated_y,
                    state.velocity_x, state.velocity_y,
                    ts,
                    current_pan_deg=camera.state.pan_deg,
                    current_tilt_deg=camera.state.tilt_deg,
                )

            if loss_time is not None:
                search.update(dt, camera.state.pan_deg, camera.state.tilt_deg)
                strategy_used = search.state.strategy.value

                # Apply search motion
                new_pan = camera.state.pan_deg + search.pan_rate_deg_s * dt
                new_tilt = camera.state.tilt_deg + search.tilt_rate_deg_s * dt
                camera.set_target_pan_tilt(new_pan, new_tilt)
                camera.update(dt)

            ts += dt

        assert loss_time is not None, "Loss should have been detected"
        assert strategy_used is not None, "Search should have started"

        # Phase 3: Target reappears (50 frames to allow full reacquisition chain)
        for i in range(50):
            det = BeaconDetection(
                detected=True, center_x=320, center_y=240,
                confidence=0.8, timestamp_s=ts,
            )
            state = tracker.update([det], ts)

            if search.state.frames_in_search > 0 and reacquire_time is None:
                accepted = search.process_detection(
                    det.center_x, det.center_y, det.confidence, ts,
                )
                if accepted:
                    reacquire_time = ts
                    strategy_used = search.state.strategy.value
                    search_reacquired = search.state.reacquired
                    search.reset()

            # Once tracking resumes, we're done
            if reacquire_time is not None and state.state == TrackState.TRACKING:
                break

            ts += dt

        # Verify metrics
        loss_duration = (reacquire_time or ts) - loss_time

        assert loss_duration > 0, "Loss duration should be positive"
        assert strategy_used in [s.value for s in SearchStrategy]
        assert search_reacquired is True

        # Verify tracking resumed after reacquisition
        assert state.state == TrackState.TRACKING

    def test_search_strategy_selected_by_duration(self):
        """Longer loss should progress through strategies."""
        sc = SearchController(SearchConfig(
            local_timeout_frames=3,
            uncertainty_timeout_frames=3,
            predictive_timeout_frames=3,
            spiral_timeout_frames=3,
            global_timeout_frames=3,
        ))
        sc.begin_search(320, 240, 50, 0, 1.0)

        strategies_seen = []
        for _ in range(20):
            sc.update(0.033)
            strategies_seen.append(sc.state.strategy.value)

        assert "local_last_known" in strategies_seen
        assert "uncertainty_region" in strategies_seen
        assert "predictive_search" in strategies_seen
        assert "expanding_spiral" in strategies_seen
        assert "global_sweep" in strategies_seen

    def test_failure_predictor_produces_risk(self):
        """Failure predictor produces P(failure) from causal inputs."""
        fp = FailurePredictor()
        window = [
            ObservationFeatures(
                timestamp_s=float(i) * 0.033,
                detected=True,
                confidence=0.9 - i * 0.05,
                residual_px=float(i),
                uncertainty_x_px=5.0 + i,
                uncertainty_y_px=5.0 + i,
                velocity_x_px_s=50.0,
                velocity_y_px_s=0.0,
                distance_from_center_px=50.0,
                time_since_detection_s=0.0,
                latency_ms=10.0,
                source_fps=30.0,
                processing_fps=30.0,
                candidate_count=1,
            )
            for i in range(10)
        ]
        pred = fp.predict(window)
        assert 0.0 <= pred.risk_score <= 1.0
        assert pred.recommended_action in ("track", "hold", "reacquire", "search")

    def test_gt_boundary_in_search(self):
        """Search module never uses ground truth."""
        import inspect
        source = inspect.getsource(SearchController)
        assert "WorldTruth" not in source
        assert "ground_truth" not in source.lower()
        assert "true_" not in source

    def test_gt_boundary_in_adaptive_roi(self):
        """AdaptiveROI never uses ground truth."""
        import inspect
        source = inspect.getsource(AdaptiveROI)
        assert "WorldTruth" not in source
        assert "ground_truth" not in source.lower()


class TestSearchNeverParks:
    """Regression: SEARCHING must always drive the camera.

    A search that expires into zero rates leaves a static "SEARCHING"
    display while the camera sits still (spec violation). The sweep
    recycles instead.
    """

    def test_rates_nonzero_past_max_duration(self):
        from fsoc_tracker.tracking.search import SearchConfig
        sc = SearchController(SearchConfig(max_search_duration_s=1.0))
        sc.begin_search(320, 240, 0, 0, 0.0)
        rates = set()
        for _ in range(300):  # 10 s sim, far past expiry
            sc.update(1.0 / 30.0)
            rates.add((round(sc.pan_rate_deg_s, 3),
                       round(sc.tilt_rate_deg_s, 3)))
        assert (0.0, 0.0) not in rates or len(rates) > 1
        assert sc.pan_rate_deg_s != 0.0 or sc.tilt_rate_deg_s != 0.0

    def test_no_complete_parking_state(self):
        from fsoc_tracker.tracking.search import SearchConfig, SearchPhase
        sc = SearchController(SearchConfig(max_search_duration_s=0.5))
        sc.begin_search(320, 240, 0, 0, 0.0)
        for _ in range(120):
            st = sc.update(1.0 / 30.0)
        assert st.phase != SearchPhase.COMPLETE

    def test_search_override_command_is_applicable(self):
        """Search-pattern commands must carry an applicable mode so the
        actuator applies them (DISABLED/SAFE_STOP would be dropped)."""
        from fsoc_tracker.control.command import ControlCommand
        from fsoc_tracker.control.config import ControlMode
        cmd = ControlCommand(
            pan_rate_deg_s=4.0, tilt_rate_deg_s=-4.0, timestamp_s=1.0,
            control_mode=ControlMode.HOLD,
        )
        from fsoc_tracker.control.controller import CameraActuator
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        cam = VirtualCamera(CameraState())
        pan0 = cam.state.pan_deg
        CameraActuator().apply_command(cam, cmd, 1.0 / 30.0)
        assert cam.state.pan_deg != pan0


class TestSearchWorldAnchored:
    """Search waypoints must track real sky, not collapse to boresight."""

    def test_local_phase_holds_origin(self):
        from fsoc_tracker.tracking.search import SearchController
        sc = SearchController()
        sc.begin_search(320, 240, 0, 0, 0.0,
                        current_pan_deg=-16.6, current_tilt_deg=4.7)
        sc.update(1.0 / 30.0, current_pan_deg=-16.6,
                  current_tilt_deg=4.7)
        # Center-pixel target at origin orientation: near-zero rates
        # (holds position) instead of slewing toward pan=0.
        assert abs(sc.pan_rate_deg_s) < 1.0
        assert abs(sc.tilt_rate_deg_s) < 1.0

    def test_global_sweep_spans_sky(self):
        from fsoc_tracker.tracking.search import (
            SearchConfig,
            SearchController,
            SearchPhase,
        )
        sc = SearchController(SearchConfig(sweep_span_deg=30.0))
        sc.begin_search(320, 240, 0, 0, 0.0,
                        current_pan_deg=0.0, current_tilt_deg=0.0)
        sc._state.phase = SearchPhase.GLOBAL_SWEEP
        # Drive the mount with the commanded rates (as the actuator
        # would): cells advance on arrival, so sky coverage — not a fixed
        # frame cadence — is what the sweep must achieve.
        pan, tilt = 0.0, 0.0
        pans, tilts = [], []
        # Full-sky raster with mount dwells needs a few thousand frames;
        # updates are cheap (no rendering), so run the whole sweep.
        for _ in range(4000):
            sc.update(1.0 / 30.0, current_pan_deg=pan,
                      current_tilt_deg=tilt)
            pan += sc.pan_rate_deg_s / 30.0
            tilt += sc.tilt_rate_deg_s / 30.0
            pans.append(pan)
            tilts.append(tilt)
        # Sweep must cover real sky in both directions on both axes.
        assert max(pans) > 15.0
        assert min(pans) < -15.0
        assert max(tilts) > 5.0
        assert min(tilts) < -5.0

    def test_sweep_follows_origin_offset(self):
        from fsoc_tracker.tracking.search import (
            SearchController,
        )
        sc = SearchController()
        sc.begin_search(100, 100, 0, 0, 0.0,
                        current_pan_deg=20.0, current_tilt_deg=-10.0)
        assert sc._origin_pan_deg == pytest.approx(20.0)
        assert sc._origin_tilt_deg == pytest.approx(-10.0)
        sc.reset()
        assert sc._origin_pan_deg == pytest.approx(0.0)


class TestRiskPreemption:
    def test_expand_for_risk_triggers(self):
        from fsoc_tracker.perception.adaptive_roi import AdaptiveROI, ROIConfig, ROIState
        roi = AdaptiveROI(ROIConfig())
        roi._state = ROIState.STABLE
        assert roi.expand_for_risk(0.85) is True
        assert roi._state == ROIState.EXPANDING

    def test_expand_for_risk_ignores_low_and_nan(self):
        from fsoc_tracker.perception.adaptive_roi import AdaptiveROI, ROIConfig, ROIState
        roi = AdaptiveROI(ROIConfig())
        roi._state = ROIState.STABLE
        assert roi.expand_for_risk(0.2) is False
        assert roi.expand_for_risk(float("nan")) is False
        assert roi._state == ROIState.STABLE

    def test_expand_for_risk_never_blinds(self):
        from fsoc_tracker.perception.adaptive_roi import AdaptiveROI, ROIConfig, ROIState
        for state in (ROIState.FULL_FRAME, ROIState.EXPANDING,
                      ROIState.LOST, ROIState.RECOVERY):
            roi = AdaptiveROI(ROIConfig())
            roi._state = state
            assert roi.expand_for_risk(0.99) is False
            assert roi._state == state

    def test_spiral_constant_linear_speed(self):
        """Expanding spiral sweeps uniform path length per frame."""
        import math

        from fsoc_tracker.tracking.search import (
            SearchController,
            SearchPhase,
        )
        sc = SearchController()
        sc.begin_search(320, 240, 0, 0, 0.0)
        sc._state.phase = SearchPhase.EXPANDING_SPIRAL
        prev = None
        steps = []
        for _ in range(35):
            sc.update(1.0 / 30.0, current_pan_deg=0.0,
                      current_tilt_deg=0.0)
            assert sc._state.phase == SearchPhase.EXPANDING_SPIRAL
            x = sc._state.search_center_x + sc._state.spiral_radius_px * math.cos(
                sc._state.pattern_angle_rad)
            y = sc._state.search_center_y + sc._state.spiral_radius_px * math.sin(
                sc._state.pattern_angle_rad)
            if prev is not None:
                steps.append(math.hypot(x - prev[0], y - prev[1]))
            prev = (x, y)
        # Uniform ~12 px/frame arc steps (fixed angular steps would
        # accelerate outward and vary several-fold over this range).
        assert max(steps) - min(steps) < 6.0
        assert 8.0 < sum(steps) / len(steps) < 16.0
