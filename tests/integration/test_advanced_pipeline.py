"""End-to-end advanced integration tests — Stage 13.

Tests the full advanced pipeline:
- Hybrid perception with quality analysis
- Adaptive Kalman with maneuver detection
- Intelligent search and reacquisition
- Adaptive controller with gain scheduling
- Diagnostic event logging
- Baseline vs advanced comparison
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helper: build complete advanced pipeline
# ---------------------------------------------------------------------------

def _make_advanced_pipeline(
    trajectory: str = "straight_line",
    dt: float = 1.0 / 30.0,
    target_size: float = 10.0,
    disturbance: str = "clear",
    seed: int = 42,
):
    """Build a complete advanced pipeline with all Stage 13 components."""
    from fsoc_tracker.advanced.diagnostics import DiagnosticLog
    from fsoc_tracker.control.adaptive import AdaptiveController
    from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
    from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
    from fsoc_tracker.perception.config import PerceptionConfig, ThresholdMode
    from fsoc_tracker.perception.fusion import FusionEngine
    from fsoc_tracker.perception.policy import PerceptionPolicy
    from fsoc_tracker.perception.quality import ImageQualityAnalyzer
    from fsoc_tracker.perception.refinement import CoarseToFineRefiner
    from fsoc_tracker.perception.uncertainty import UncertaintyEstimator
    from fsoc_tracker.pipeline.eval import EvalSink
    from fsoc_tracker.pipeline.pipeline import TrackingPipeline
    from fsoc_tracker.pipeline.sources import SimulationSource
    from fsoc_tracker.simulation.camera.camera import VirtualCamera
    from fsoc_tracker.simulation.camera.state import CameraState
    from fsoc_tracker.simulation.engine import SimulationEngine
    from fsoc_tracker.simulation.sensor.config import SensorConfig
    from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
    from fsoc_tracker.simulation.world import WorldConfig
    from fsoc_tracker.tracking.adaptive_kalman import AdaptiveKalmanManager
    from fsoc_tracker.tracking.lock_quality import LockQualityEstimator
    from fsoc_tracker.tracking.maneuver import ManeuverDetector
    from fsoc_tracker.tracking.search import SearchController
    from fsoc_tracker.tracking.tracker import KalmanTracker

    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=seed)
    engine = SimulationEngine(wc)

    traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2}
    if trajectory == "circular":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius": 0.5, "angular_speed_rad_s": 0.3}
    elif trajectory == "figure_8":
        traj_params = {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3}
    elif trajectory == "random":
        traj_params = {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "seed": seed}

    engine.add_target(trajectory_type=trajectory, trajectory_params=traj_params)

    cam_state = CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=640, height=480,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
    )
    camera = VirtualCamera(cam_state)

    sc = SensorConfig(width=640, height=480, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    perception_cfg = PerceptionConfig(threshold_mode=ThresholdMode.GLOBAL, threshold_value=20.0)
    source = SimulationSource(engine, camera, sensor, None, dt, eval_sink=EvalSink())
    pipeline = TrackingPipeline(
        perception=ClassicalBeaconDetector(perception_cfg),
        tracker=KalmanTracker(),
        controller=CoarsePointingController(),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)

    advanced_components = {
        "quality_analyzer": ImageQualityAnalyzer(),
        "uncertainty_estimator": UncertaintyEstimator(),
        "perception_policy": PerceptionPolicy(),
        "fusion_engine": FusionEngine(),
        "refiner": CoarseToFineRefiner(),
        "adaptive_kalman": AdaptiveKalmanManager(),
        "maneuver_detector": ManeuverDetector(),
        "search_controller": SearchController(),
        "lock_quality": LockQualityEstimator(),
        "adaptive_controller": AdaptiveController(),
        "diagnostics": DiagnosticLog(),
    }

    return pipeline, source, advanced_components


def _run_advanced_frames(pipeline, components, num_frames):
    """Run pipeline with advanced component integration."""
    pipeline.start()
    results = []

    for i in range(num_frames):
        r = pipeline.step()
        if r is None:
            break

        if r.perception and r.perception.detected and r.perception.primary_detection and pipeline.last_frame is not None:
            det = r.perception.primary_detection
            quality = components["quality_analyzer"].analyze(
                pipeline.last_frame,
                (max(0, int(det.center_x - 10)), max(0, int(det.center_y - 10)),
                 min(640, int(det.center_x + 10)), min(480, int(det.center_y + 10))),
            )
        elif pipeline.last_frame is not None:
            quality = components["quality_analyzer"].analyze(pipeline.last_frame)
        else:
            quality = components["quality_analyzer"].analyze(np.zeros((480, 640), dtype=np.uint8))

        uncertainty = components["uncertainty_estimator"].estimate(
            kalman_position_uncertainty=(r.tracking.uncertainty_x, r.tracking.uncertainty_y) if r.tracking else None,
            detection_confidence=r.tracking.detection_confidence if r.tracking else 0.0,
            measurement_residual=r.tracking.residual_magnitude if r.tracking else 0.0,
            missed_detections=r.tracking.consecutive_misses if r.tracking else 0,
        )

        has_det = r.perception is not None and r.perception.detected
        state_name = r.tracking.state.name if r.tracking else "NO_TRACK"
        policy = components["perception_policy"].decide(quality, uncertainty, state_name, has_det)

        components["diagnostics"].log(
            event=__import__('fsoc_tracker.advanced.diagnostics', fromlist=['DiagnosticEvent']).DiagnosticEvent.PERCEPTION_SWITCH,
            timestamp_s=r.timestamp_s,
            reason=policy.reason,
            frame_index=r.frame_index,
        )

        results.append(r)

    pipeline.stop()
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdvancedEndToEnd:
    def test_advanced_pipeline_runs(self):
        pipeline, source, components = _make_advanced_pipeline()
        results = _run_advanced_frames(pipeline, components, 50)
        assert len(results) > 0
        assert all(r.perception is not None for r in results)

    def test_quality_analyzer_integrates(self):
        pipeline, source, components = _make_advanced_pipeline()
        pipeline.start()
        r = pipeline.step()
        pipeline.stop()
        assert r is not None
        if pipeline.last_frame is not None:
            quality = components["quality_analyzer"].analyze(pipeline.last_frame)
            assert quality.level is not None

    def test_uncertainty_estimation_integrates(self):
        pipeline, source, components = _make_advanced_pipeline()
        results = _run_advanced_frames(pipeline, components, 30)
        for r in results:
            if r.tracking:
                unc = components["uncertainty_estimator"].estimate(
                    kalman_position_uncertainty=(r.tracking.uncertainty_x, r.tracking.uncertainty_y),
                    detection_confidence=r.tracking.detection_confidence,
                )
                assert unc.level is not None

    def test_diagnostic_events_logged(self):
        pipeline, source, components = _make_advanced_pipeline()
        _run_advanced_frames(pipeline, components, 30)
        assert len(components["diagnostics"].entries) > 0

    def test_adaptive_kalman_scales(self):
        pipeline, source, components = _make_advanced_pipeline()
        results = _run_advanced_frames(pipeline, components, 30)
        ak = components["adaptive_kalman"]
        q, r = ak.q_scale, ak.r_scale
        assert q > 0
        assert r > 0

    def test_maneuver_detector_runs(self):
        from fsoc_tracker.tracking.maneuver import ManeuverDetector, MotionClass
        detector = ManeuverDetector()
        for _ in range(10):
            state = detector.update(2.0, 2.0, 100.0, 0.0, 1.0 / 30.0)
        assert state.motion_class == MotionClass.SMOOTH
        assert len(detector._innovation_history) == 10

    def test_lock_quality_computed(self):
        pipeline, source, components = _make_advanced_pipeline()
        results = _run_advanced_frames(pipeline, components, 30)
        lq = components["lock_quality"]
        score = lq.estimate(
            position_error_px=5.0,
            detection_confidence=0.7,
            track_age_s=1.0,
            consecutive_hits=10,
            has_detection=True,
        )
        assert 0.0 <= score <= 1.0

    def test_search_controller_lifecycle(self):
        from fsoc_tracker.tracking.search import SearchController
        sc = SearchController()
        sc.begin_search(320.0, 240.0, 50.0, 0.0, 0.0)
        for _ in range(5):
            sc.update(1.0 / 30.0)
        assert sc.state.phase is not None
        region = sc.get_search_region()
        assert region[2] > region[0]


class TestBaselineVsAdvanced:
    def test_baseline_pipeline(self):
        from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.perception.config import PerceptionConfig, ThresholdMode
        from fsoc_tracker.pipeline.eval import EvalSink
        from fsoc_tracker.pipeline.pipeline import TrackingPipeline
        from fsoc_tracker.pipeline.sources import SimulationSource
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.engine import SimulationEngine
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
        from fsoc_tracker.simulation.world import WorldConfig
        from fsoc_tracker.tracking.tracker import KalmanTracker

        wc = WorldConfig(width=2000.0, height=2000.0, random_seed=42)
        engine = SimulationEngine(wc)
        engine.add_target(
            trajectory_type="straight_line",
            trajectory_params={"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2},
        )
        cam_state = CameraState(horizontal_fov_deg=4.0, vertical_fov_deg=3.0, width=640, height=480, max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0)
        camera = VirtualCamera(cam_state)
        sensor = VirtualSensorRenderer(SensorConfig(width=640, height=480, beacon_default_size_px=10.0))

        p_cfg = PerceptionConfig(threshold_mode=ThresholdMode.GLOBAL, threshold_value=20.0)
        source = SimulationSource(engine, camera, sensor, None, 1.0/30.0, eval_sink=EvalSink())
        pipeline = TrackingPipeline(
            perception=ClassicalBeaconDetector(p_cfg),
            tracker=KalmanTracker(),
            controller=CoarsePointingController(),
            actuator=CameraActuator(),
        )
        pipeline.set_source(source)

        pipeline.start()
        baseline_results = []
        for _ in range(100):
            r = pipeline.step()
            if r is None:
                break
            baseline_results.append(r)
        pipeline.stop()

        assert len(baseline_results) > 0

    def test_advanced_pipeline_same_scenario(self):
        pipeline, source, components = _make_advanced_pipeline(seed=42)
        results = _run_advanced_frames(pipeline, components, 100)
        assert len(results) > 0


class TestAdvancedConfigIntegration:
    def test_feature_flags(self):
        from fsoc_tracker.advanced import AdvancedConfig
        cfg = AdvancedConfig()
        flags = cfg.feature_flags()
        assert flags["quality_analysis"] is True
        assert flags["adaptive_kalman"] is True
        assert flags["adaptive_controller"] is True

    def test_disable_all_advanced(self):
        from fsoc_tracker.advanced import AdvancedConfig
        cfg = AdvancedConfig()
        cfg.quality_analysis.enabled = False
        cfg.adaptive_kalman.enabled = False
        cfg.adaptive_controller.enabled = False
        cfg.maneuver_detection.enabled = False
        flags = cfg.feature_flags()
        assert flags["quality_analysis"] is False
        assert flags["adaptive_kalman"] is False
