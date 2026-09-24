"""Tests for the perception → tracking → control pipeline chain.

Verifies:
1. Centroid derived from image data (not GT)
2. Centroid under noisy detections
3. Missed detection handling
4. State prediction continuity
5. Target loss transition
6. Target reacquisition
7. detection_age_s populated correctly
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.perception.centroid import compute_centroid
from fsoc_tracker.perception.classical import detect_beacon
from fsoc_tracker.perception.config import PerceptionConfig, ThresholdMode
from fsoc_tracker.perception.models import BeaconDetection, PerceptionStatus
from fsoc_tracker.tracking.config import TrackerConfig
from fsoc_tracker.tracking.state import TrackState
from fsoc_tracker.tracking.tracker import KalmanTracker


def _make_config(**kw) -> PerceptionConfig:
    defaults = dict(
        threshold_mode=ThresholdMode.PERCENTILE,
        percentile_value=95.0,
        min_confidence=0.3,
        min_area=2,
        max_area=5000,
    )
    defaults.update(kw)
    return PerceptionConfig(**defaults)


def _beacon_image(
    h: int = 120, w: int = 160,
    cx: float = 80.0, cy: float = 60.0,
    radius: int = 5,
    intensity: float = 255.0,
    noise: float = 0.0,
) -> np.ndarray:
    """Create a synthetic image with a Gaussian beacon at (cx, cy)."""
    img = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(radius, 1.0)
    img = intensity * np.exp(-r2 / (2.0 * sigma ** 2))
    if noise > 0:
        rng = np.random.default_rng(42)
        img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ── 1. Centroid is derived from image data ──────────────────────────────

class TestCentroidFromImage:
    def test_bright_spot_centroid(self) -> None:
        """Centroid of a bright spot matches its known position."""
        img = _beacon_image(cx=80.0, cy=60.0, radius=4)
        mask = img > 100
        cx, cy = compute_centroid(img, mask)
        assert abs(cx - 80.0) < 1.0
        assert abs(cy - 60.0) < 1.0

    def test_centroid_not_from_gt(self) -> None:
        """Centroid is purely image-based; no world truth needed."""
        img = _beacon_image(cx=40.0, cy=30.0, radius=3)
        mask = img > 50
        cx, cy = compute_centroid(img, mask)
        assert 30.0 < cx < 50.0
        assert 20.0 < cy < 40.0

    def test_detector_uses_image_centroid(self) -> None:
        """ClassicalBeaconDetector returns detection with image-derived centroid."""
        img = _beacon_image(cx=100.0, cy=50.0, radius=6)
        cfg = _make_config()
        result = detect_beacon(img, cfg, timestamp_s=0.0)
        assert result.detected
        det = result.primary_detection
        assert abs(det.center_x - 100.0) < 2.0
        assert abs(det.center_y - 50.0) < 2.0


# ── 2. Centroid under noisy detections ──────────────────────────────────

class TestNoisyCentroid:
    def test_noise_shifts_centroid(self) -> None:
        """Noisy image shifts centroid but detection still works."""
        img_clean = _beacon_image(cx=80.0, cy=60.0, radius=4, noise=0)
        img_noisy = _beacon_image(cx=80.0, cy=60.0, radius=4, noise=20)

        mask_clean = img_clean > 100
        mask_noisy = img_noisy > 100

        cx_clean, cy_clean = compute_centroid(img_clean, mask_clean)
        cx_noisy, cy_noisy = compute_centroid(img_noisy, mask_noisy)

        # Noisy centroid is within a few pixels of clean
        assert abs(cx_noisy - cx_clean) < 3.0
        assert abs(cy_noisy - cy_clean) < 3.0

    def test_tracker_filters_noise(self) -> None:
        """Kalman filter smooths noisy detections."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=1,
            minimum_detection_confidence=0.0,
        )
        tracker = KalmanTracker(cfg)

        # Feed sequence of noisy detections around (100, 60)
        rng = np.random.default_rng(99)
        base_x, base_y = 100.0, 60.0
        positions = []
        for i in range(20):
            t = i * 0.033
            noise_x = rng.normal(0, 3.0)
            noise_y = rng.normal(0, 3.0)
            det = BeaconDetection(
                detected=True,
                center_x=base_x + noise_x,
                center_y=base_y + noise_y,
                confidence=0.9,
                timestamp_s=t,
                visibility_state=PerceptionStatus.DETECTED,
            )
            state = tracker.update([det], t)
            positions.append((state.estimated_x, state.estimated_y))

        # Final estimate should be close to base
        final_x, final_y = positions[-1]
        assert abs(final_x - base_x) < 5.0
        assert abs(final_y - base_y) < 5.0


# ── 3. Missed detection handling ────────────────────────────────────────

class TestMissedDetection:
    def test_miss_increases_miss_count(self) -> None:
        """Missed detection increments consecutive_misses."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        # Acquire track
        det = _det(100, 60, ts=0.0)
        tracker.update([det], 0.0)
        assert tracker.state == TrackState.TRACKING

        # Miss detection
        state = tracker.update([], 0.033)
        assert state.consecutive_misses == 1

    def test_miss_continues_prediction(self) -> None:
        """During miss, prediction continues (prediction_only=True)."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        # Acquire and move target
        for i in range(10):
            t = i * 0.033
            det = _det(100.0 + i * 2, 60.0, ts=t)
            tracker.update([det], t)

        # Miss 3 frames
        for i in range(3):
            t = (10 + i) * 0.033
            state = tracker.update([], t)
            assert state.prediction_only is True
            assert state.state == TrackState.TRACKING


def _det(x, y, confidence=0.9, ts=0.0):
    return BeaconDetection(
        detected=True, center_x=x, center_y=y,
        confidence=confidence, timestamp_s=ts,
        visibility_state=PerceptionStatus.DETECTED,
    )


# ── 4. State prediction continuity ─────────────────────────────────────

class TestStatePrediction:
    def test_prediction_steps_forward(self) -> None:
        """Predicted position steps forward in the direction of motion."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        # Move target linearly rightward
        for i in range(10):
            t = i * 0.033
            det = _det(100.0 + i * 5, 60.0, ts=t)
            state = tracker.update([det], t)

        # predicted_x should be > estimated_x (moving right)
        assert state.predicted_x > state.estimated_x

    def test_prediction_dt_matches(self) -> None:
        """Prediction uses the actual dt, not a hardcoded FPS."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        # Slow frame: dt=0.1s
        det1 = _det(100, 60, ts=0.0)
        tracker.update([det1], 0.0)

        det2 = _det(105, 60, ts=0.1)
        state = tracker.update([det2], 0.1)

        # Velocity should reflect ~50 px/s (5 px / 0.1s), allow for Kalman lag
        vx, vy = state.velocity_x, state.velocity_y
        assert abs(vx - 50.0) < 50.0


# ── 5. Target loss transition ──────────────────────────────────────────

class TestTargetLoss:
    def test_long_miss_transitions_to_lost(self) -> None:
        """Sustained miss transitions TRACKING → LOST."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=1,
            max_prediction_duration_s=0.5,
        )
        tracker = KalmanTracker(cfg)

        # Acquire track
        det = _det(100, 60, ts=0.0)
        tracker.update([det], 0.0)
        assert tracker.state == TrackState.TRACKING

        # Miss for > max_prediction_duration
        t = 0.0
        for i in range(30):
            t = 0.033 * (i + 1)
            state = tracker.update([], t)

        # Should have transitioned to LOST
        assert state.state == TrackState.LOST

    def test_lost_has_last_known_position(self) -> None:
        """LOST state retains last known position."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=1,
            max_prediction_duration_s=0.3,
        )
        tracker = KalmanTracker(cfg)

        # Acquire at known position
        det = _det(150, 80, ts=0.0)
        state = tracker.update([det], 0.0)
        acquired_x = state.estimated_x
        acquired_y = state.estimated_y

        # Miss
        for i in range(20):
            state = tracker.update([], 0.033 * (i + 1))

        assert state.state == TrackState.LOST
        # Position should be near last known
        assert abs(state.estimated_x - acquired_x) < 50.0


# ── 6. Target reacquisition ────────────────────────────────────────────

class TestReacquisition:
    def test_reappear_transitions_to_reacquiring(self) -> None:
        """Detection during LOST transitions to REACQUIRING."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=1,
            max_prediction_duration_s=0.2,
        )
        tracker = KalmanTracker(cfg)

        # Acquire
        det = _det(100, 60, ts=0.0)
        tracker.update([det], 0.0)

        # Lose
        for i in range(15):
            state = tracker.update([], 0.033 * (i + 1))
        assert state.state == TrackState.LOST

        # Reappear
        re_det = _det(110, 60, ts=0.5)
        state = tracker.update([re_det], 0.5)
        assert state.state == TrackState.REACQUIRING

    def test_reacquiring_becomes_tracking(self) -> None:
        """REACQUIRING with consecutive hits transitions to TRACKING."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=2,
            max_prediction_duration_s=0.2,
            reacquisition_timeout_s=2.0,
        )
        tracker = KalmanTracker(cfg)

        # Acquire: need 3 total detections (1 in NO_TRACK + 2 in ACQUIRING)
        tracker.update([_det(100, 60, ts=0.0)], 0.0)
        tracker.update([_det(100, 60, ts=0.033)], 0.033)
        tracker.update([_det(100, 60, ts=0.066)], 0.066)
        assert tracker.state == TrackState.TRACKING

        # Lose
        for i in range(15):
            state = tracker.update([], 0.066 + 0.033 * (i + 1))
        assert state.state == TrackState.LOST

        # Reappear: first hit → REACQUIRING
        state = tracker.update([_det(110, 60, ts=0.6)], 0.6)
        assert state.state == TrackState.REACQUIRING

        # Second hit → TRACKING
        state = tracker.update([_det(112, 60, ts=0.633)], 0.633)
        assert state.state == TrackState.TRACKING


# ── 7. detection_age_s populated correctly ──────────────────────────────

class TestDetectionAge:
    def test_detection_age_zero_after_hit(self) -> None:
        """detection_age_s is ~0 when detection is present."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        det = _det(100, 60, ts=0.1)
        state = tracker.update([det], 0.1)
        assert state.detection_age_s < 0.01

    def test_detection_age_grows_during_miss(self) -> None:
        """detection_age_s grows during sustained miss."""
        cfg = TrackerConfig(acquisition_min_consecutive_hits=1)
        tracker = KalmanTracker(cfg)

        # Acquire
        det = _det(100, 60, ts=0.0)
        tracker.update([det], 0.0)

        # Miss
        state = tracker.update([], 0.1)
        assert state.detection_age_s > 0.08

    def test_detection_age_resets_on_reacq(self) -> None:
        """detection_age_s resets to ~0 after reacquisition."""
        cfg = TrackerConfig(
            acquisition_min_consecutive_hits=1,
            max_prediction_duration_s=0.2,
        )
        tracker = KalmanTracker(cfg)

        # Acquire, lose, reacquire
        tracker.update([_det(100, 60, ts=0.0)], 0.0)
        for i in range(15):
            tracker.update([], 0.033 * (i + 1))
        state = tracker.update([_det(110, 60, ts=0.5)], 0.5)
        assert state.detection_age_s < 0.01


class TestStaticClutterSuppression:
    """Pipeline drops overlay-static candidates on video sources only."""

    def _pipe(self, source=None):
        from fsoc_tracker.pipeline.pipeline import TrackingPipeline
        from fsoc_tracker.pipeline.sources import VideoSource
        pipe = TrackingPipeline(
            perception=__import__(
                "fsoc_tracker.perception.classical_engine",
                fromlist=["ClassicalBeaconDetector"]).ClassicalBeaconDetector(
                    config=PerceptionConfig.for_video()),
            tracker=KalmanTracker(),
        )
        pipe._source = VideoSource("/nonexistent.mp4") if source == "video" else source
        return pipe

    def _frame_with_dot(self, x, y, size=5, peak=230):
        img = np.full((240, 320), 5, dtype=np.uint8)
        yy, xx = np.ogrid[:240, :320]
        img[(xx - x) ** 2 + (yy - y) ** 2 <= (size // 2) ** 2] = peak
        return img

    def _run_frames(self, pipe, positions):
        from fsoc_tracker.perception.classical import detect_beacon
        out = []
        for i, (x, y) in enumerate(positions):
            res = detect_beacon(self._frame_with_dot(x, y),
                                PerceptionConfig.for_video(), 0.0, i)
            out.append(pipe._suppress_static_clutter(res))
        return out

    def test_static_dot_dropped_after_history(self):
        pipe = self._pipe(source="video")
        results = self._run_frames(pipe, [(160, 120)] * 14)
        assert results[0].status == PerceptionStatus.DETECTED
        assert results[-1].status == PerceptionStatus.NO_TARGET
        assert results[-1].diagnostics.get("static_suppressed", 0) >= 1

    def test_moving_dot_survives(self):
        pipe = self._pipe(source="video")
        results = self._run_frames(
            pipe, [(100 + 5 * i, 120) for i in range(14)])
        assert all(r.status == PerceptionStatus.DETECTED for r in results[6:])
        assert all(len(r.detections) >= 1 for r in results[6:])

    def test_non_video_source_bypasses(self):
        pipe = self._pipe(source=None)
        results = self._run_frames(pipe, [(160, 120)] * 14)
        assert all(r.status == PerceptionStatus.DETECTED for r in results)

    def test_history_reset_on_new_source(self):
        pipe = self._pipe(source="video")
        self._run_frames(pipe, [(160, 120)] * 14)
        assert len(pipe._static_hist) == 11
        from fsoc_tracker.pipeline.sources import VideoSource
        pipe.set_source(VideoSource("/other.mp4"))
        assert pipe._static_hist == []


class TestTrackerAblation:
    """Runtime Kalman bypass (raw-detection tracking)."""

    def _pipe(self):
        import sys
        sys.path.insert(0, "src")
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
        from fsoc_tracker.perception.config import PerceptionConfig
        from fsoc_tracker.pipeline.pipeline import TrackingPipeline
        from fsoc_tracker.tracking.tracker import KalmanTracker
        return TrackingPipeline(
            perception=ClassicalBeaconDetector(config=PerceptionConfig()),
            tracker=KalmanTracker(), controller=CoarsePointingController())

    def _frame(self, x, y):
        import numpy as np

        from fsoc_tracker.core.models import ColorModel, Frame
        img = np.full((240, 320), 5, dtype=np.uint8)
        yy, xx = np.ogrid[:240, :320]
        img[(xx - x) ** 2 + (yy - y) ** 2 <= 25] = 230
        return Frame(image=img, width=320, height=240, channels=1,
                     color_model=ColorModel.GRAY, source_id="t",
                     source_type="sim", frame_index=0, timestamp_s=0.0)

    def test_raw_mode_mirrors_detection(self):
        pipe = self._pipe()
        pipe.set_tracker_enabled(False)
        res = pipe.process_frame(self._frame(160, 120))
        assert res.perception.detected
        assert res.tracking.state.name == "TRACKING"
        assert res.tracking.locked is True
        assert abs(res.tracking.estimated_x - 160) < 3.0
        assert abs(res.tracking.estimated_y - 120) < 3.0

    def test_raw_mode_miss_is_lost(self):
        import numpy as np

        from fsoc_tracker.core.models import ColorModel, Frame
        pipe = self._pipe()
        pipe.set_tracker_enabled(False)
        img = np.full((240, 320), 5, dtype=np.uint8)
        res = pipe.process_frame(Frame(
            image=img, width=320, height=240, channels=1,
            color_model=ColorModel.GRAY, source_id="t",
            source_type="sim", frame_index=0, timestamp_s=0.0))
        assert res.tracking.state.name == "LOST"
        assert res.tracking.locked is False

    def test_reenable_resets_kalman(self):
        pipe = self._pipe()
        pipe.set_tracker_enabled(False)
        pipe.process_frame(self._frame(160, 120))
        pipe.set_tracker_enabled(True)
        assert pipe._use_kalman_tracker is True
