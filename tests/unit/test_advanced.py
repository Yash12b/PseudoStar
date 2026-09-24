"""Comprehensive tests for Stage 13 — Advanced Intelligence.

Tests cover:
- Image quality analysis
- Uncertainty estimation
- Perception policy
- Fusion engine
- Adaptive Kalman
- Maneuver detection
- Coarse-to-fine refinement
- Search/reacquisition
- Lock quality
- Adaptive controller
- PID tuning
- Diagnostic events
- Advanced config
- End-to-end integration
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Image Quality Analysis
# ---------------------------------------------------------------------------

class TestImageQualityAnalyzer:
    def _make_image(self, brightness: float = 128.0, noise: float = 0.0, h: int = 100, w: int = 100) -> np.ndarray:
        img = np.full((h, w), brightness, dtype=np.uint8)
        if noise > 0:
            rng = np.random.default_rng(42)
            img = np.clip(img.astype(np.float64) + rng.normal(0, noise, (h, w)), 0, 255).astype(np.uint8)
        return img

    def test_bright_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityLevel
        analyzer = ImageQualityAnalyzer()
        img = self._make_image(brightness=200.0, noise=5.0)
        state = analyzer.analyze(img)
        assert state.mean_brightness > 150
        assert state.level in (QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.DEGRADED, QualityLevel.POOR)

    def test_dark_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityLevel
        analyzer = ImageQualityAnalyzer()
        img = self._make_image(brightness=5.0)
        state = analyzer.analyze(img)
        assert state.mean_brightness < 20
        assert state.level in (QualityLevel.POOR, QualityLevel.CRITICAL, QualityLevel.DEGRADED)

    def test_noisy_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        img = self._make_image(brightness=128.0, noise=40.0)
        state = analyzer.analyze(img)
        assert state.noise_estimate > 5.0

    def test_uniform_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        img = np.full((100, 100), 128, dtype=np.uint8)
        state = analyzer.analyze(img)
        assert state.contrast == 0.0
        assert state.edge_density == 0.0

    def test_empty_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityLevel
        analyzer = ImageQualityAnalyzer()
        state = analyzer.analyze(np.array([]))
        assert state.level == QualityLevel.CRITICAL

    def test_none_image(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityLevel
        analyzer = ImageQualityAnalyzer()
        state = analyzer.analyze(None)
        assert state.level == QualityLevel.CRITICAL

    def test_sbr_with_region(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        img = np.full((100, 100), 10, dtype=np.uint8)
        img[45:55, 45:55] = 200
        state = analyzer.analyze(img, detection_region=(45, 45, 55, 55))
        assert state.signal_to_background > 5.0

    def test_beacon_quality(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        img = np.full((100, 100), 10, dtype=np.uint8)
        yy, xx = np.mgrid[40:60, 40:60]
        dist = np.sqrt((xx - 50) ** 2 + (yy - 50) ** 2)
        img[40:60, 40:60] = np.clip(255 - dist * 10, 0, 255).astype(np.uint8)
        state = analyzer.analyze(img, detection_region=(40, 40, 60, 60))
        assert state.beacon_quality > 0.3

    def test_quality_to_dict(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        state = analyzer.analyze(np.full((50, 50), 128, dtype=np.uint8))
        d = state.to_dict()
        assert "level" in d
        assert "mean_brightness" in d

    def test_custom_thresholds(self):
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityThresholds
        thresholds = QualityThresholds(min_brightness=100.0)
        analyzer = ImageQualityAnalyzer(thresholds)
        img = np.full((50, 50), 50, dtype=np.uint8)
        state = analyzer.analyze(img)
        assert not state.brightness_ok


# ---------------------------------------------------------------------------
# Uncertainty Estimation
# ---------------------------------------------------------------------------

class TestUncertaintyEstimator:
    def test_low_uncertainty(self):
        from fsoc_tracker.perception.uncertainty import UncertaintyEstimator, UncertaintyLevel
        estimator = UncertaintyEstimator()
        state = estimator.estimate(
            detection_confidence=0.9,
            measurement_residual=2.0,
            quality_factor=0.9,
            missed_detections=0,
            time_since_detection_s=0.0,
            kalman_position_uncertainty=(3.0, 3.0),
        )
        assert state.level in (UncertaintyLevel.VERY_LOW, UncertaintyLevel.LOW)

    def test_high_uncertainty(self):
        from fsoc_tracker.perception.uncertainty import UncertaintyEstimator, UncertaintyLevel
        estimator = UncertaintyEstimator()
        state = estimator.estimate(
            detection_confidence=0.1,
            measurement_residual=40.0,
            quality_factor=0.2,
            missed_detections=5,
            time_since_detection_s=2.0,
            kalman_position_uncertainty=(60.0, 60.0),
        )
        assert state.level in (UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH)

    def test_with_covariance_matrix(self):
        from fsoc_tracker.perception.uncertainty import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        cov = np.diag([25.0, 25.0, 100.0, 100.0])
        state = estimator.estimate(kalman_covariance=cov)
        assert state.position_uncertainty_x == pytest.approx(5.0, abs=0.1)
        assert state.position_uncertainty_y == pytest.approx(5.0, abs=0.1)

    def test_uncertainty_dict(self):
        from fsoc_tracker.perception.uncertainty import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        state = estimator.estimate()
        d = state.to_dict()
        assert "level" in d
        assert "position_uncertainty_total" in d


# ---------------------------------------------------------------------------
# Perception Policy
# ---------------------------------------------------------------------------

class TestPerceptionPolicy:
    def test_searching_gives_full_frame(self):
        from fsoc_tracker.perception.policy import PerceptionMode, PerceptionPolicy
        from fsoc_tracker.perception.quality import QualityState
        from fsoc_tracker.perception.uncertainty import UncertaintyState
        policy = PerceptionPolicy()
        decision = policy.decide(QualityState(), UncertaintyState(), "SEARCHING")
        assert decision.mode == PerceptionMode.FULL_FRAME

    def test_tracking_gives_roi(self):
        from fsoc_tracker.perception.policy import PerceptionMode, PerceptionPolicy
        from fsoc_tracker.perception.quality import QualityLevel, QualityState
        from fsoc_tracker.perception.uncertainty import UncertaintyState
        policy = PerceptionPolicy()
        q = QualityState(level=QualityLevel.GOOD)
        decision = policy.decide(q, UncertaintyState(), "TRACKING", has_detection=True)
        assert decision.mode == PerceptionMode.ROI

    def test_high_uncertainty_widens_roi(self):
        from fsoc_tracker.perception.policy import PerceptionPolicy
        from fsoc_tracker.perception.quality import QualityLevel, QualityState
        from fsoc_tracker.perception.uncertainty import UncertaintyLevel, UncertaintyState
        policy = PerceptionPolicy()
        q = QualityState(level=QualityLevel.GOOD)
        u = UncertaintyState(level=UncertaintyLevel.HIGH)
        decision = policy.decide(q, u, "TRACKING", has_detection=True)
        assert decision.roi_expand_factor >= 2.0

    def test_critical_quality_fallback(self):
        from fsoc_tracker.perception.policy import PerceptionMode, PerceptionPolicy
        from fsoc_tracker.perception.quality import QualityLevel, QualityState
        from fsoc_tracker.perception.uncertainty import UncertaintyState
        policy = PerceptionPolicy()
        q = QualityState(level=QualityLevel.CRITICAL)
        decision = policy.decide(q, UncertaintyState(), "TRACKING", has_detection=True)
        assert decision.mode == PerceptionMode.FULL_FRAME
        assert decision.use_enhanced_denoising


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

class TestFusionEngine:
    def _make_result(self, detected: bool, cx: float, cy: float, conf: float):
        from fsoc_tracker.perception.models import (
            BeaconDetection,
            PerceptionResult,
            PerceptionStatus,
        )
        det = BeaconDetection(
            detected=detected, center_x=cx, center_y=cy, confidence=conf,
            visibility_state=PerceptionStatus.DETECTED if detected else PerceptionStatus.NO_TARGET,
        )
        return PerceptionResult(
            detections=[det] if det else [],
            primary_detection=det if det else None,
            status=PerceptionStatus.DETECTED if detected else PerceptionStatus.NO_TARGET,
        )

    def test_classical_only(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        classical = self._make_result(True, 320.0, 240.0, 0.8)
        fused = engine.fuse(classical_result=classical)
        assert fused.detected
        assert fused.source == "classical"

    def test_ai_only(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        ai = self._make_result(True, 321.0, 239.0, 0.9)
        fused = engine.fuse(ai_result=ai)
        assert fused.detected
        assert fused.source == "ai"

    def test_both_sources_fuse(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        classical = self._make_result(True, 320.0, 240.0, 0.8)
        ai = self._make_result(True, 322.0, 238.0, 0.9)
        fused = engine.fuse(classical_result=classical, ai_result=ai)
        assert fused.detected
        assert fused.source == "hybrid"
        assert 319 < fused.center_x < 323

    def test_no_sources_no_detection(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        fused = engine.fuse()
        assert not fused.detected

    def test_tracker_prediction_used_when_no_detection(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        fused = engine.fuse(predicted_position=(100.0, 200.0))
        assert fused.detected
        assert fused.confidence < 0.3

    def test_confidence_rejection(self):
        from fsoc_tracker.perception.fusion import FusionConfig, FusionEngine
        config = FusionConfig(min_fusion_confidence=0.5)
        engine = FusionEngine(config)
        classical = self._make_result(True, 320.0, 240.0, 0.1)
        fused = engine.fuse(classical_result=classical)
        assert fused.rejected

    def test_quality_factor_influences_score(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        classical = self._make_result(True, 320.0, 240.0, 0.5)
        fused_good = engine.fuse(classical_result=classical, quality_factor=1.0)
        engine2 = FusionEngine()
        classical2 = self._make_result(True, 320.0, 240.0, 0.5)
        fused_bad = engine2.fuse(classical_result=classical2, quality_factor=0.1)
        assert fused_good.fused_score >= fused_bad.fused_score

    def test_max_confidence_method(self):
        from fsoc_tracker.perception.fusion import FusionConfig, FusionEngine, FusionMethod
        config = FusionConfig(method=FusionMethod.MAX_CONFIDENCE)
        engine = FusionEngine(config)
        classical = self._make_result(True, 320.0, 240.0, 0.5)
        ai = self._make_result(True, 330.0, 250.0, 0.9)
        fused = engine.fuse(classical_result=classical, ai_result=ai)
        assert fused.confidence == 0.9
        assert fused.source == "ai"

    def test_reset(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        engine = FusionEngine()
        classical = self._make_result(True, 320.0, 240.0, 0.8)
        engine.fuse(classical_result=classical)
        engine.reset()
        assert engine._consistent_count == 0


# ---------------------------------------------------------------------------
# Maneuver Detection
# ---------------------------------------------------------------------------

class TestManeuverDetector:
    def test_smooth_motion(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector, MotionClass
        detector = ManeuverDetector()
        for _ in range(10):
            state = detector.update(1.0, 1.0, 100.0, 0.0, 1.0 / 30.0)
        assert state.motion_class == MotionClass.SMOOTH

    def test_maneuvering_detected(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector, MotionClass
        detector = ManeuverDetector()
        for _ in range(5):
            detector.update(1.0, 1.0, 100.0, 0.0, 1.0 / 30.0)
        for _ in range(5):
            state = detector.update(20.0, 20.0, 100.0, 200.0, 1.0 / 30.0)
        assert state.motion_class in (MotionClass.MANEUVERING, MotionClass.UNPREDICTABLE)

    def test_unpredictable_extreme_innovation(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector, MotionClass
        detector = ManeuverDetector()
        state = detector.update(50.0, 50.0, 100.0, 0.0, 1.0 / 30.0)
        assert state.motion_class == MotionClass.UNPREDICTABLE

    def test_reset(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector
        detector = ManeuverDetector()
        detector.update(20.0, 20.0, 100.0, 200.0, 1.0 / 30.0)
        detector.reset()
        assert len(detector._innovation_history) == 0

    def test_maneuver_dict(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector
        detector = ManeuverDetector()
        state = detector.update(5.0, 5.0, 100.0, 0.0, 1.0 / 30.0)
        d = state.to_dict()
        assert "motion_class" in d


# ---------------------------------------------------------------------------
# Coarse-to-Fine Refinement
# ---------------------------------------------------------------------------

class TestCoarseToFineRefiner:
    def _make_image_with_beacon(self, cx: float = 50.5, cy: float = 50.5) -> np.ndarray:
        img = np.full((100, 100), 10, dtype=np.uint8)
        yy, xx = np.mgrid[40:61, 40:61]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img[40:61, 40:61] = np.clip(255 - dist * 8, 0, 255).astype(np.uint8)
        return img

    def test_intensity_weighted_refinement(self):
        from fsoc_tracker.perception.refinement import (
            CoarseToFineRefiner,
            RefinementConfig,
            RefinementMethod,
        )
        refiner = CoarseToFineRefiner(RefinementConfig(method=RefinementMethod.INTENSITY_WEIGHTED))
        img = self._make_image_with_beacon(50.5, 50.5)
        result = refiner.refine(img, 50.0, 50.0)
        assert result.succeeded
        assert abs(result.x - 50.5) < 1.0
        assert abs(result.y - 50.5) < 1.0

    def test_gaussian_fit_refinement(self):
        from fsoc_tracker.perception.refinement import CoarseToFineRefiner, RefinementConfig
        refiner = CoarseToFineRefiner(RefinementConfig(method="gaussian_fit"))
        img = self._make_image_with_beacon(52.0, 48.0)
        result = refiner.refine(img, 50.0, 50.0)
        assert result.succeeded
        assert abs(result.x - 52.0) < 2.0
        assert abs(result.y - 48.0) < 2.0

    def test_refinement_delta(self):
        from fsoc_tracker.perception.refinement import CoarseToFineRefiner, RefinementConfig
        refiner = CoarseToFineRefiner(RefinementConfig())
        img = self._make_image_with_beacon(55.0, 45.0)
        result = refiner.refine(img, 50.0, 50.0)
        assert result.refinement_delta_x != 0.0 or result.refinement_delta_y != 0.0

    def test_empty_image(self):
        from fsoc_tracker.perception.refinement import CoarseToFineRefiner, RefinementConfig
        refiner = CoarseToFineRefiner(RefinementConfig())
        result = refiner.refine(np.array([]), 50.0, 50.0)
        assert not result.succeeded or result.x == 50.0


# ---------------------------------------------------------------------------
# Search Controller
# ---------------------------------------------------------------------------

class TestSearchController:
    def test_begin_search(self):
        from fsoc_tracker.tracking.search import SearchController, SearchPhase, SearchStrategy
        ctrl = SearchController()
        state = ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        assert state.phase == SearchPhase.LOCAL_LAST_KNOWN
        assert state.strategy == SearchStrategy.LOCAL_LAST_KNOWN
        assert state.search_radius_px > 0

    def test_search_advances_phases(self):
        from fsoc_tracker.tracking.search import SearchConfig, SearchController, SearchPhase
        config = SearchConfig(
            local_timeout_frames=3,
            uncertainty_timeout_frames=3,
            predictive_timeout_frames=3,
            spiral_timeout_frames=3,
            global_timeout_frames=3,
        )
        ctrl = SearchController(config)
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        for _ in range(10):
            ctrl.update(1.0 / 30.0)
        assert ctrl.state.phase != SearchPhase.LOCAL_LAST_KNOWN

    def test_search_region_bounded(self):
        from fsoc_tracker.tracking.search import SearchConfig, SearchController
        config = SearchConfig(image_width=640, image_height=480)
        ctrl = SearchController(config)
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        region = ctrl.get_search_region()
        assert region[0] >= 0
        assert region[1] >= 0
        assert region[2] <= 640
        assert region[3] <= 480

    def test_detection_acceptance(self):
        from fsoc_tracker.tracking.search import SearchController
        ctrl = SearchController()
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        accepted = ctrl.process_detection(325.0, 245.0, 0.8, 0.1)
        assert accepted

    def test_low_confidence_rejected(self):
        from fsoc_tracker.tracking.search import SearchController
        ctrl = SearchController()
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        accepted = ctrl.process_detection(325.0, 245.0, 0.05, 0.1)
        assert not accepted

    def test_search_dict(self):
        from fsoc_tracker.tracking.search import SearchController
        ctrl = SearchController()
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        d = ctrl.state.to_dict()
        assert "phase" in d


# ---------------------------------------------------------------------------
# Lock Quality
# ---------------------------------------------------------------------------

class TestLockQuality:
    def test_perfect_lock(self):
        from fsoc_tracker.tracking.lock_quality import LockQualityEstimator
        est = LockQualityEstimator()
        score = est.estimate(
            position_error_px=0.0,
            detection_confidence=1.0,
            track_age_s=5.0,
            consecutive_hits=20,
            position_uncertainty=1.0,
            measurement_residual=0.0,
            image_quality_factor=1.0,
            has_detection=True,
        )
        assert score > 0.8

    def test_no_detection(self):
        from fsoc_tracker.tracking.lock_quality import LockQualityEstimator
        est = LockQualityEstimator()
        score = est.estimate(has_detection=False)
        assert score == 0.0

    def test_poor_conditions(self):
        from fsoc_tracker.tracking.lock_quality import LockQualityEstimator
        est = LockQualityEstimator()
        score = est.estimate(
            position_error_px=40.0,
            detection_confidence=0.1,
            track_age_s=0.1,
            consecutive_hits=1,
            position_uncertainty=70.0,
            measurement_residual=30.0,
            image_quality_factor=0.2,
            has_detection=True,
        )
        assert score < 0.4


# ---------------------------------------------------------------------------
# Adaptive Controller
# ---------------------------------------------------------------------------

class TestAdaptiveController:
    def test_low_error_reduces_gains(self):
        from fsoc_tracker.control.adaptive import AdaptiveController, GainSchedule
        ctrl = AdaptiveController()
        ctrl.set_baseline_gains(0.5, 0.1, 0.5, 0.1)
        t = ctrl.compute_adaptation(2.0, 2.0, 10.0, 0.0)
        assert t.gain_schedule == GainSchedule.LOW_ERROR
        assert t.kp_scale_pan < 1.0

    def test_high_error_increases_gains(self):
        from fsoc_tracker.control.adaptive import AdaptiveController, GainSchedule
        ctrl = AdaptiveController()
        ctrl.set_baseline_gains(0.5, 0.1, 0.5, 0.1)
        t = ctrl.compute_adaptation(30.0, 30.0, 100.0, 0.0)
        assert t.gain_schedule == GainSchedule.HIGH_ERROR
        assert t.kp_scale_pan >= 1.0

    def test_feed_forward(self):
        from fsoc_tracker.control.adaptive import AdaptiveController
        ctrl = AdaptiveController()
        t = ctrl.compute_adaptation(10.0, 0.0, 200.0, 0.0, image_width=640)
        assert t.feed_forward_pan != 0.0

    def test_oscillation_detection(self):
        from fsoc_tracker.control.adaptive import AdaptiveController
        ctrl = AdaptiveController()
        errors = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
        osc_detected = False
        for e in errors:
            t = ctrl.compute_adaptation(e, 0.0, 0.0, 0.0)
            if t.oscillation_detected:
                osc_detected = True
        assert osc_detected

    def test_disabled_returns_defaults(self):
        from fsoc_tracker.control.adaptive import AdaptiveController, AdaptiveControllerConfig
        config = AdaptiveControllerConfig(enabled=False)
        ctrl = AdaptiveController(config)
        t = ctrl.compute_adaptation(10.0, 10.0, 100.0, 0.0)
        assert not t.adaptive_active

    def test_apply_to_pid(self):
        from fsoc_tracker.control.adaptive import AdaptiveController, AdaptiveTelemetry
        from fsoc_tracker.control.pid import PIDController
        ctrl = AdaptiveController()
        pid = PIDController(kp=0.5, kd=0.1)
        t = AdaptiveTelemetry(kp_scale_pan=1.5)
        t.kd_scale_pan = 1.2
        ctrl.apply_to_pid(pid, t, "pan")
        assert pid.kp == pytest.approx(0.75, abs=0.01)
        assert pid.kd == pytest.approx(0.12, abs=0.01)

    def test_adaptive_dict(self):
        from fsoc_tracker.control.adaptive import AdaptiveTelemetry
        t = AdaptiveTelemetry()
        d = t.to_dict()
        assert "gain_schedule" in d
        assert "reasons" in d


# ---------------------------------------------------------------------------
# Adaptive Kalman
# ---------------------------------------------------------------------------

class TestAdaptiveKalman:
    def test_quality_affects_r(self):
        from fsoc_tracker.perception.quality import QualityLevel, QualityState
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        mgr = AdaptiveKalmanManager()
        q_good = QualityState(level=QualityLevel.GOOD)
        q_bad = QualityState(level=QualityLevel.POOR)
        _, r_good = mgr.update(quality=q_good)
        mgr.reset()
        _, r_bad = mgr.update(quality=q_bad)
        assert r_bad > r_good

    def test_maneuver_increases_q(self):
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        from fsoc_tracker.tracking.maneuver import ManeuverState, MotionClass
        mgr = AdaptiveKalmanManager()
        m_smooth = ManeuverState(motion_class=MotionClass.SMOOTH)
        m_maneuver = ManeuverState(motion_class=MotionClass.MANEUVERING)
        q_smooth, _ = mgr.update(maneuver=m_smooth)
        mgr.reset()
        q_maneuver, _ = mgr.update(maneuver=m_maneuver)
        assert q_maneuver > q_smooth

    def test_bounded_scales(self):
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        mgr = AdaptiveKalmanManager()
        for _ in range(50):
            mgr.update(detection_confidence=0.0, has_detection=False)
        assert mgr.q_scale <= 5.0
        assert mgr.r_scale <= 10.0

    def test_smoothing(self):
        from fsoc_tracker.perception.quality import QualityLevel, QualityState
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        mgr = AdaptiveKalmanManager()
        q_bad = QualityState(level=QualityLevel.CRITICAL)
        for _ in range(3):
            mgr.update(quality=q_bad)
        assert mgr.r_scale < 10.0

    def test_apply_to_matrices(self):
        import numpy as np

        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        mgr = AdaptiveKalmanManager()
        mgr._current_q_scale = 2.0
        mgr._current_r_scale = 3.0
        Q = np.eye(4)
        R = np.eye(2)
        Q_a, R_a = mgr.apply_to_matrices(Q, R)
        assert Q_a[0, 0] == pytest.approx(2.0)
        assert R_a[0, 0] == pytest.approx(3.0)

    def test_disabled_returns_ones(self):
        from fsoc_tracker.tracking.adaptive_kalman import (
            AdaptiveKalmanConfig,
            AdaptiveKalmanManager,
        )
        config = AdaptiveKalmanConfig(enabled=False)
        mgr = AdaptiveKalmanManager(config)
        q, r = mgr.update()
        assert q == 1.0
        assert r == 1.0


# ---------------------------------------------------------------------------
# PID Tuning
# ---------------------------------------------------------------------------

class TestPIDTuning:
    def test_evaluate_candidate(self):
        from fsoc_tracker.control.tuning import PIDTuningEvaluator
        evaluator = PIDTuningEvaluator()
        errors = [30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0]
        cmds = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]
        ts = [i * 0.033 for i in range(7)]
        c = evaluator.evaluate_candidate(errors, cmds, ts)
        assert c.rmse_px > 0
        assert c.objective_score > 0

    def test_empty_data(self):
        from fsoc_tracker.control.tuning import PIDTuningEvaluator
        evaluator = PIDTuningEvaluator()
        c = evaluator.evaluate_candidate([], [], [])
        assert c.rmse_px == 0.0

    def test_grid_search(self):
        from fsoc_tracker.control.tuning import PIDSweepRunner
        runner = PIDSweepRunner()
        grid = {"kp": [0.3, 0.5, 0.7], "kd": [0.05, 0.1]}
        from fsoc_tracker.control.tuning import TuningCandidate
        def eval_fn(kp, kd):
            return TuningCandidate(pan_kp=kp, pan_kd=kd, objective_score=kp + kd)
        results = runner.grid_search(grid, eval_fn)
        assert len(results) == 6
        assert results[0].objective_score >= results[-1].objective_score

    def test_random_search(self):
        from fsoc_tracker.control.tuning import PIDSweepRunner
        runner = PIDSweepRunner(seed=42)
        ranges = {"kp": (0.1, 1.0), "kd": (0.01, 0.3)}
        from fsoc_tracker.control.tuning import TuningCandidate
        def eval_fn(kp, kd):
            return TuningCandidate(pan_kp=kp, pan_kd=kd, objective_score=kp)
        results = runner.random_search(ranges, eval_fn, n_trials=20)
        assert len(results) == 20

    def test_select_best(self):
        from fsoc_tracker.control.tuning import PIDSweepRunner, TuningCandidate
        runner = PIDSweepRunner()
        candidates = [
            TuningCandidate(objective_score=0.9, lock_retention_pct=95.0),
            TuningCandidate(objective_score=0.95, lock_retention_pct=60.0),
        ]
        best = runner.select_best(candidates, min_lock_retention=80.0)
        assert best.lock_retention_pct >= 80.0


# ---------------------------------------------------------------------------
# Diagnostic Events
# ---------------------------------------------------------------------------

class TestDiagnosticLog:
    def test_log_event(self):
        from fsoc_tracker.advanced.diagnostics import DiagnosticEvent, DiagnosticLog
        log = DiagnosticLog()
        log.log(DiagnosticEvent.SEARCH_START, 0.0, "target_lost")
        assert len(log.entries) == 1

    def test_bounded_log(self):
        from fsoc_tracker.advanced.diagnostics import DiagnosticEvent, DiagnosticLog
        log = DiagnosticLog(max_entries=10)
        for i in range(20):
            log.log(DiagnosticEvent.PERCEPTION_SWITCH, float(i), "reason")
        assert len(log.entries) <= 10

    def test_query_by_type(self):
        from fsoc_tracker.advanced.diagnostics import DiagnosticEvent, DiagnosticLog
        log = DiagnosticLog()
        log.log(DiagnosticEvent.SEARCH_START, 0.0, "r1")
        log.log(DiagnosticEvent.LOCK_GAINED, 0.1, "r2")
        log.log(DiagnosticEvent.SEARCH_END, 0.2, "r3")
        search_events = log.get_events(DiagnosticEvent.SEARCH_START)
        assert len(search_events) == 1

    def test_count_recent(self):
        from fsoc_tracker.advanced.diagnostics import DiagnosticEvent, DiagnosticLog
        log = DiagnosticLog()
        log.log(DiagnosticEvent.OSCILLATION_DETECTED, 1.0, "r")
        log.log(DiagnosticEvent.OSCILLATION_DETECTED, 1.5, "r")
        log.log(DiagnosticEvent.OSCILLATION_DETECTED, 5.0, "r")
        count = log.count_recent(DiagnosticEvent.OSCILLATION_DETECTED, window_s=1.0, current_time=2.0)
        assert count == 2

    def test_to_list(self):
        from fsoc_tracker.advanced.diagnostics import DiagnosticEvent, DiagnosticLog
        log = DiagnosticLog()
        log.log(DiagnosticEvent.SEARCH_START, 0.0, "r")
        lst = log.to_list()
        assert len(lst) == 1
        assert "event" in lst[0]


# ---------------------------------------------------------------------------
# Advanced Config
# ---------------------------------------------------------------------------

class TestAdvancedConfig:
    def test_default_config(self):
        from fsoc_tracker.advanced import AdvancedConfig
        cfg = AdvancedConfig()
        flags = cfg.feature_flags()
        assert "quality_analysis" in flags
        assert flags["quality_analysis"] is True

    def test_disable_feature(self):
        from fsoc_tracker.advanced import AdvancedConfig
        cfg = AdvancedConfig()
        cfg.adaptive_kalman.enabled = False
        flags = cfg.feature_flags()
        assert flags["adaptive_kalman"] is False

    def test_all_features_flaggable(self):
        from fsoc_tracker.advanced import AdvancedConfig
        cfg = AdvancedConfig()
        flags = cfg.feature_flags()
        assert len(flags) >= 15


# ---------------------------------------------------------------------------
# End-to-End Advanced Integration
# ---------------------------------------------------------------------------

class TestEndToEndAdvanced:
    def test_quality_to_policy_to_fusion_pipeline(self):
        from fsoc_tracker.perception.fusion import FusionEngine
        from fsoc_tracker.perception.models import (
            BeaconDetection,
            PerceptionResult,
            PerceptionStatus,
        )
        from fsoc_tracker.perception.policy import PerceptionMode, PerceptionPolicy
        from fsoc_tracker.perception.quality import ImageQualityAnalyzer, QualityLevel
        from fsoc_tracker.perception.uncertainty import UncertaintyEstimator, UncertaintyLevel

        rng = np.random.default_rng(42)
        img = rng.integers(80, 180, size=(480, 640), dtype=np.uint8)
        img[235:245, 315:325] = 250

        quality_analyzer = ImageQualityAnalyzer()
        quality = quality_analyzer.analyze(img, (315, 235, 325, 245))
        assert quality.level != QualityLevel.CRITICAL

        uncertainty_est = UncertaintyEstimator()
        uncertainty = uncertainty_est.estimate(
            kalman_position_uncertainty=(5.0, 5.0),
            detection_confidence=0.8,
            quality_factor=1.0 - quality.blur_score,
        )
        assert uncertainty.level in (UncertaintyLevel.VERY_LOW, UncertaintyLevel.LOW, UncertaintyLevel.MODERATE)

        policy = PerceptionPolicy()
        decision = policy.decide(quality, uncertainty, "TRACKING", has_detection=True)
        assert decision.mode in (PerceptionMode.ROI, PerceptionMode.FULL_FRAME)

        fusion = FusionEngine()
        det = BeaconDetection(
            detected=True, center_x=320.0, center_y=240.0, confidence=0.8,
            visibility_state=PerceptionStatus.DETECTED,
        )
        result = PerceptionResult(
            detections=[det], primary_detection=det,
            status=PerceptionStatus.DETECTED,
        )
        fused = fusion.fuse(classical_result=result, quality_factor=1.0 - quality.blur_score)
        assert fused.detected

    def test_maneuver_to_adaptive_kalman(self):
        from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
        from fsoc_tracker.tracking.maneuver import ManeuverDetector

        detector = ManeuverDetector()
        manager = AdaptiveKalmanManager()

        for _ in range(3):
            detector.update(2.0, 2.0, 100.0, 0.0, 1.0 / 30.0)
        for _ in range(5):
            m = detector.update(25.0, 25.0, 100.0, 300.0, 1.0 / 30.0)
            q, r = manager.update(maneuver=m)

        assert manager.q_scale > 1.0

    def test_search_to_reacquisition_flow(self):
        from fsoc_tracker.tracking.search import SearchController
        ctrl = SearchController()
        ctrl.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        ctrl.update(1.0 / 30.0)
        ctrl.update(1.0 / 30.0)
        region = ctrl.get_search_region()
        assert region[2] > region[0]
        assert region[3] > region[1]
        accepted = ctrl.process_detection(322.0, 242.0, 0.8, 0.1)
        assert accepted

    def test_adaptive_controller_full_cycle(self):
        from fsoc_tracker.control.adaptive import AdaptiveController
        ctrl = AdaptiveController()
        ctrl.set_baseline_gains(0.5, 0.1, 0.5, 0.1)
        errors = [30.0, 20.0, 10.0, 5.0, 2.0, 1.0]
        for e in errors:
            t = ctrl.compute_adaptation(e, e * 0.5, 50.0, 25.0)
        assert t.adaptive_active
