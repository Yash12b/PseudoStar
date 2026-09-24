"""Tests for temporal observable dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from fsoc_tracker.ai.temporal_dataset import (
    FEATURE_DIM,
    HORIZONS_S,
    TemporalDatasetSplit,
    TemporalSample,
    TemporalSequence,
    _compute_failure_labels,
    _compute_future_displacements,
    _run_trajectory,
    _tracking_to_observation,
)
from fsoc_tracker.tracking.state import TrackingState, TrackState


class TestTemporalSample:
    def test_shapes(self):
        feat = np.zeros(FEATURE_DIM, dtype=np.float64)
        disp = np.zeros((len(HORIZONS_S), 2), dtype=np.float64)
        unc = np.zeros((len(HORIZONS_S), 2), dtype=np.float64)
        sample = TemporalSample(
            features=feat,
            future_displacements=disp,
            future_uncertainties=unc,
            situation_label="normal_tracking",
            failure_within_0_5s=False,
            timestamp_s=1.0,
            dt=0.033,
        )
        assert sample.features.shape == (FEATURE_DIM,)
        assert sample.future_displacements.shape == (len(HORIZONS_S), 2)
        assert sample.situation_label == "normal_tracking"
        assert sample.failure_within_0_5s is False


class TestTrackingToObservation:
    def test_no_track(self):
        ts = TrackingState(state=TrackState.NO_TRACK)
        obs = _tracking_to_observation(ts, 1.0, 0.033, 30.0, 30.0, 0, 100.0, 640, 480)
        assert obs.detected is False
        assert obs.confidence == 0.0
        assert obs.velocity_x_px_s == 0.0

    def test_tracking_with_detection(self):
        ts = TrackingState(
            state=TrackState.TRACKING,
            has_detection=True,
            detection_confidence=0.85,
            uncertainty_x=3.0,
            uncertainty_y=4.0,
            velocity_x=10.0,
            velocity_y=-5.0,
            estimated_x=320.0,
            estimated_y=240.0,
        )
        obs = _tracking_to_observation(ts, 1.0, 0.033, 30.0, 25.0, 3, 80.0, 640, 480)
        assert obs.detected is True
        assert obs.confidence == 0.85
        assert obs.uncertainty_x_px == 3.0
        assert obs.velocity_x_px_s == 10.0
        assert obs.source_fps == 30.0
        assert obs.processing_fps == 25.0
        assert obs.candidate_count == 3
        assert obs.roi_radius_px == 80.0


class TestFutureDisplacements:
    def test_computation(self):
        samples = []
        for i in range(60):
            feat = np.zeros(FEATURE_DIM, dtype=np.float64)
            feat[0] = i * 0.033  # timestamp
            feat[6] = 10.0  # velocity_x
            feat[7] = -5.0  # velocity_y
            feat[3] = 3.0  # uncertainty_x
            feat[4] = 4.0  # uncertainty_y
            samples.append(TemporalSample(
                features=feat,
                future_displacements=np.zeros((len(HORIZONS_S), 2)),
                future_uncertainties=np.zeros((len(HORIZONS_S), 2)),
                situation_label="normal_tracking",
                failure_within_0_5s=False,
                timestamp_s=i * 0.033,
                dt=0.033,
                estimated_x=float(i) * 10.0 * 0.033,
                estimated_y=float(i) * -5.0 * 0.033,
            ))
        disp, unc = _compute_future_displacements(samples, 0)
        assert disp.shape == (len(HORIZONS_S), 2)
        assert unc.shape == (len(HORIZONS_S), 2)
        for h_idx, horizon in enumerate(HORIZONS_S):
            target_ts = 0.0 + horizon
            found_idx = None
            for j in range(1, len(samples)):
                if samples[j].timestamp_s >= target_ts:
                    found_idx = j
                    break
            if found_idx is not None:
                expected_dx = samples[found_idx].estimated_x - samples[0].estimated_x
                expected_dy = samples[found_idx].estimated_y - samples[0].estimated_y
                assert disp[h_idx, 0] == pytest.approx(expected_dx, rel=1e-6)
                assert disp[h_idx, 1] == pytest.approx(expected_dy, rel=1e-6)


class TestFailureLabels:
    def test_no_failure(self):
        samples = []
        for i in range(50):
            feat = np.zeros(FEATURE_DIM, dtype=np.float64)
            feat[0] = i * 0.033
            samples.append(TemporalSample(
                features=feat,
                future_displacements=np.zeros((len(HORIZONS_S), 2)),
                future_uncertainties=np.zeros((len(HORIZONS_S), 2)),
                situation_label="normal_tracking",
                failure_within_0_5s=False,
                timestamp_s=i * 0.033,
                dt=0.033,
            ))
        assert _compute_failure_labels(samples, 0) is False

    def test_failure_detected(self):
        samples = []
        for i in range(50):
            feat = np.zeros(FEATURE_DIM, dtype=np.float64)
            feat[0] = i * 0.033
            samples.append(TemporalSample(
                features=feat,
                future_displacements=np.zeros((len(HORIZONS_S), 2)),
                future_uncertainties=np.zeros((len(HORIZONS_S), 2)),
                situation_label="target_lost",
                failure_within_0_5s=(10 <= i <= 15),
                timestamp_s=i * 0.033,
                dt=0.033,
            ))
        assert _compute_failure_labels(samples, 5) is True
        assert _compute_failure_labels(samples, 20) is False


class TestTemporalDatasetSplit:
    def test_to_arrays(self):
        samples1 = [TemporalSample(
            features=np.ones(FEATURE_DIM, dtype=np.float64) * i,
            future_displacements=np.ones((len(HORIZONS_S), 2), dtype=np.float64) * i,
            future_uncertainties=np.ones((len(HORIZONS_S), 2), dtype=np.float64) * 0.1,
            situation_label="normal_tracking",
            failure_within_0_5s=False,
            timestamp_s=i * 0.033,
            dt=0.033,
        ) for i in range(5)]
        samples2 = [TemporalSample(
            features=np.ones(FEATURE_DIM, dtype=np.float64) * (i + 10),
            future_displacements=np.ones((len(HORIZONS_S), 2), dtype=np.float64) * (i + 10),
            future_uncertainties=np.ones((len(HORIZONS_S), 2), dtype=np.float64) * 0.2,
            situation_label="target_lost",
            failure_within_0_5s=True,
            timestamp_s=i * 0.033,
            dt=0.033,
        ) for i in range(3)]
        split = TemporalDatasetSplit(
            sequences=[
                TemporalSequence(samples=samples1, trajectory_type="straight_line", disturbance="clear", seed=42),
                TemporalSequence(samples=samples2, trajectory_type="circular", disturbance="light", seed=43),
            ],
            split_name="test",
        )
        arrs = split.to_arrays()
        assert arrs["features"].shape == (2, 5, FEATURE_DIM)
        assert arrs["mask"][0, :5].sum() == 5.0
        assert arrs["mask"][1, :3].sum() == 3.0
        assert arrs["failure_labels"][1, 0] == 1.0
        assert arrs["failure_labels"][0, 0] == 0.0

    def test_empty_split(self):
        split = TemporalDatasetSplit(sequences=[], split_name="empty")
        assert split.to_arrays() == {}
        assert split.num_sequences == 0
        assert split.num_samples == 0


class TestRunTrajectory:
    def test_straight_line_produces_samples(self):
        seq = _run_trajectory(
            trajectory_type="straight_line",
            traj_params={"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2},
            disturbance="clear",
            seed=42,
            duration_s=2.0,
            dt=1.0 / 30.0,
        )
        assert len(seq.samples) > 0
        assert seq.samples[0].features.shape == (FEATURE_DIM,)
        assert seq.samples[0].future_displacements.shape == (len(HORIZONS_S), 2)

    def test_disturbance_applied(self):
        seq = _run_trajectory(
            trajectory_type="straight_line",
            traj_params={"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2},
            disturbance="light",
            seed=42,
            duration_s=2.0,
            dt=1.0 / 30.0,
        )
        assert len(seq.samples) > 0
