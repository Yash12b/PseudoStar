"""Temporal observable dataset generator for the intelligence layer.

Runs simulation trajectories through the full pipeline and captures
runtime-observable state at each frame. Produces training data for:
- Temporal predictor (future displacement at multiple horizons)
- Situation classifier (from observable features only)
- Failure predictor (will tracking fail within N frames?)

Ground truth is used ONLY offline to generate future-position labels.
The features themselves are strictly runtime-observable quantities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_tracker.ai.mission import (
    ObservationFeatures,
    Situation,
    SituationClassifier,
)
from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.core.time import compute_dt
from fsoc_tracker.disturbances.config import get_preset_config
from fsoc_tracker.disturbances.pipeline import DisturbancePipeline
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.state import CameraState
from fsoc_tracker.simulation.engine import SimulationEngine
from fsoc_tracker.simulation.sensor.config import SensorConfig
from fsoc_tracker.simulation.sensor.renderer import VirtualSensorRenderer
from fsoc_tracker.simulation.world import WorldConfig
from fsoc_tracker.tracking.state import TrackingState, TrackState
from fsoc_tracker.tracking.tracker import KalmanTracker

HORIZONS_S: list[float] = [0.025, 0.05, 0.1, 0.25, 0.5]
FEATURE_DIM = 15
SEQUENCE_LENGTH = 20


@dataclass
class TemporalSample:
    """One frame's features + future labels."""
    features: np.ndarray  # (FEATURE_DIM,) float64
    future_displacements: np.ndarray  # (len(HORIZONS_S), 2) float64
    future_uncertainties: np.ndarray  # (len(HORIZONS_S), 2) float64
    situation_label: str
    failure_within_0_5s: bool
    timestamp_s: float
    dt: float
    estimated_x: float = 0.0  # tracker estimated position (for label computation)
    estimated_y: float = 0.0


@dataclass
class TemporalSequence:
    """A sequence of consecutive temporal samples for GRU training."""
    samples: list[TemporalSample]
    trajectory_type: str
    disturbance: str
    seed: int


@dataclass
class TemporalDatasetSplit:
    """A collection of sequences for one split."""
    sequences: list[TemporalSequence] = field(default_factory=list)
    split_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def num_samples(self) -> int:
        return sum(len(s.samples) for s in self.sequences)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert to padded numpy arrays for training.

        Returns dict with:
            features: (N_seq, T_max, FEATURE_DIM)
            future_displacements: (N_seq, T_max, H, 2)
            future_uncertainties: (N_seq, T_max, H, 2)
            situation_labels: (N_seq, T_max)
            failure_labels: (N_seq, T_max)
            mask: (N_seq, T_max) — 1.0 for valid timesteps
        """
        if not self.sequences:
            return {}

        t_max = max(len(s.samples) for s in self.sequences)
        n_seq = len(self.sequences)
        n_h = len(HORIZONS_S)
        mask = np.zeros((n_seq, t_max), dtype=np.float64)
        feat = np.zeros((n_seq, t_max, FEATURE_DIM), dtype=np.float64)
        disp = np.zeros((n_seq, t_max, n_h, 2), dtype=np.float64)
        unc = np.zeros((n_seq, t_max, n_h, 2), dtype=np.float64)
        sit = np.zeros((n_seq, t_max), dtype=np.float64)
        fail = np.zeros((n_seq, t_max), dtype=np.float64)

        for i, seq in enumerate(self.sequences):
            for j, sample in enumerate(seq.samples):
                mask[i, j] = 1.0
                feat[i, j] = sample.features
                disp[i, j] = sample.future_displacements
                unc[i, j] = sample.future_uncertainties
                try:
                    sit[i, j] = list(Situation).index(Situation(sample.situation_label))
                except (ValueError, KeyError):
                    sit[i, j] = 0.0
                fail[i, j] = 1.0 if sample.failure_within_0_5s else 0.0

        return {
            "features": feat,
            "future_displacements": disp,
            "future_uncertainties": unc,
            "situation_labels": sit,
            "failure_labels": fail,
            "mask": mask,
        }


def _tracking_to_observation(
    ts: TrackingState,
    timestamp_s: float,
    dt: float,
    source_fps: float,
    processing_fps: float,
    candidate_count: int,
    roi_radius_px: float,
    image_width: int,
    image_height: int,
) -> ObservationFeatures:
    """Extract runtime-observable features from tracking state.

    No ground truth is used — everything comes from the tracker and clock.
    """
    detected = ts.has_detection and ts.state in (TrackState.TRACKING, TrackState.ACQUIRING, TrackState.REACQUIRING)
    confidence = ts.detection_confidence if detected else 0.0
    residual = float(np.sqrt(ts.residual_x**2 + ts.residual_y**2)) if ts.has_detection else 0.0
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    dist_from_center = float(np.sqrt(
        (ts.estimated_x - center_x)**2 + (ts.estimated_y - center_y)**2
    ))

    return ObservationFeatures(
        timestamp_s=timestamp_s,
        detected=detected,
        confidence=confidence,
        residual_px=residual,
        uncertainty_x_px=max(0.0, ts.uncertainty_x),
        uncertainty_y_px=max(0.0, ts.uncertainty_y),
        velocity_x_px_s=ts.velocity_x,
        velocity_y_px_s=ts.velocity_y,
        distance_from_center_px=dist_from_center,
        time_since_detection_s=ts.time_since_last_detection_s,
        latency_ms=0.0,
        source_fps=source_fps,
        processing_fps=processing_fps,
        candidate_count=candidate_count,
        roi_radius_px=roi_radius_px,
    )


def _compute_future_displacements(
    samples: list[TemporalSample],
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground-truth future displacements and uncertainties.

    Displacement is the actual tracker-estimated position difference:
    future_estimated_position - current_estimated_position.
    This is NOT velocity * horizon (that would be constant-velocity).
    """
    n_h = len(HORIZONS_S)
    displacements = np.zeros((n_h, 2), dtype=np.float64)
    uncertainties = np.zeros((n_h, 2), dtype=np.float64)

    current_ts = samples[index].timestamp_s
    current_x = samples[index].estimated_x
    current_y = samples[index].estimated_y

    for h_idx, horizon in enumerate(HORIZONS_S):
        target_ts = current_ts + horizon
        found = False
        for future_idx in range(index + 1, len(samples)):
            if samples[future_idx].timestamp_s >= target_ts:
                future_x = samples[future_idx].estimated_x
                future_y = samples[future_idx].estimated_y
                displacements[h_idx, 0] = future_x - current_x
                displacements[h_idx, 1] = future_y - current_y
                uncertainties[h_idx, 0] = samples[future_idx].features[4]  # uncertainty_x
                uncertainties[h_idx, 1] = samples[future_idx].features[5]  # uncertainty_y
                found = True
                break
        if not found:
            last = samples[-1]
            displacements[h_idx, 0] = last.estimated_x - current_x
            displacements[h_idx, 1] = last.estimated_y - current_y
            uncertainties[h_idx, 0] = samples[index].features[4] * (1.0 + horizon)
            uncertainties[h_idx, 1] = samples[index].features[5] * (1.0 + horizon)

    return displacements, uncertainties


def _compute_failure_labels(
    samples: list[TemporalSample],
    index: int,
    horizon_s: float = 0.5,
) -> bool:
    """Check if tracking fails within the next horizon_s seconds.

    Failure = tracker enters LOST state within the horizon window.
    """
    current_ts = samples[index].timestamp_s
    target_ts = current_ts + horizon_s
    for future_idx in range(index + 1, min(index + 100, len(samples))):
        if samples[future_idx].timestamp_s > target_ts:
            break
        fail_val = samples[future_idx].failure_within_0_5s
        if fail_val:
            return True
    return False


def _run_trajectory(
    trajectory_type: str,
    traj_params: dict,
    disturbance: str,
    seed: int,
    duration_s: float = 10.0,
    dt: float = 1.0 / 30.0,
    nominal_fps: float = 30.0,
    target_size: float = 10.0,
) -> TemporalSequence:
    """Run one trajectory and capture all per-frame observable features."""
    from fsoc_tracker.disturbances.context import DisturbanceContext

    wc = WorldConfig(width=2000.0, height=2000.0, random_seed=seed)
    engine = SimulationEngine(wc)
    engine.add_target(trajectory_type=trajectory_type, trajectory_params=traj_params)

    cam_state = CameraState(
        horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        width=640, height=480,
        max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0,
    )
    camera = VirtualCamera(cam_state)
    sc = SensorConfig(width=640, height=480, beacon_default_size_px=target_size)
    sensor = VirtualSensorRenderer(sc)

    dist_cfg = get_preset_config(disturbance)
    disturbance_pipe = DisturbancePipeline(dist_cfg) if disturbance != "clear" else None

    perception_cfg = PerceptionConfig(
        threshold_mode="percentile",
        percentile_value=95.0,
        normalize_contrast_enabled=True,
    )
    perception = ClassicalBeaconDetector(perception_cfg)
    tracker = KalmanTracker()
    controller = CoarsePointingController()
    actuator = CameraActuator()

    samples: list[TemporalSample] = []
    prev_ts: float | None = None
    sim_time = 0.0
    frame_index = 0
    classifier = SituationClassifier()
    max_frames = int(duration_s / dt)

    for _ in range(max_frames):
        world_state = engine.step(dt)
        targets = world_state.get_active_targets()

        rendered = sensor.render(camera, targets, sim_time, frame_index)
        image = rendered.image.copy()

        if disturbance_pipe is not None and disturbance_pipe.config.enabled:
            dist_ctx = DisturbanceContext(
                timestamp_s=sim_time,
                dt=dt,
                frame_index=frame_index,
                image_width=640,
                image_height=480,
            )
            image = disturbance_pipe.apply_to_image(image, dist_ctx)

        if prev_ts is None:
            frame_dt = dt
        else:
            frame_dt = compute_dt(sim_time, prev_ts, nominal_fps=nominal_fps, fallback_dt=dt)
        prev_ts = sim_time

        perception_result = perception.detect(image, sim_time, frame_index)
        detections = perception_result.detections if perception_result.detected else []
        tracking_state = tracker.update(detections, sim_time)

        candidate_count = len(perception_result.detections) if perception_result else 0
        processing_fps = 1.0 / max(frame_dt, 1e-6)

        obs = _tracking_to_observation(
            tracking_state, sim_time, frame_dt,
            source_fps=nominal_fps,
            processing_fps=processing_fps,
            candidate_count=candidate_count,
            roi_radius_px=100.0,
            image_width=640, image_height=480,
        )

        is_lost = tracking_state.state == TrackState.LOST

        feat_vec = np.array([
            obs.timestamp_s,
            float(obs.detected),
            obs.confidence,
            obs.residual_px,
            obs.uncertainty_x_px,
            obs.uncertainty_y_px,
            obs.velocity_x_px_s,
            obs.velocity_y_px_s,
            obs.distance_from_center_px,
            obs.time_since_detection_s,
            obs.latency_ms,
            obs.source_fps,
            obs.processing_fps,
            float(obs.candidate_count),
            obs.roi_radius_px,
        ], dtype=np.float64)

        samples.append(TemporalSample(
            features=feat_vec,
            future_displacements=np.zeros((len(HORIZONS_S), 2), dtype=np.float64),
            future_uncertainties=np.zeros((len(HORIZONS_S), 2), dtype=np.float64),
            situation_label=classifier.classify(obs).value,
            failure_within_0_5s=is_lost,
            timestamp_s=sim_time,
            dt=frame_dt,
            estimated_x=tracking_state.estimated_x,
            estimated_y=tracking_state.estimated_y,
        ))

        command, _ = controller.compute(tracking_state, camera.intrinsics, frame_dt, sim_time)
        if command is not None:
            actuator.apply_command(camera, command, frame_dt)

        sim_time += dt
        frame_index += 1

    for i in range(len(samples)):
        disp, unc = _compute_future_displacements(samples, i)
        samples[i].future_displacements = disp
        samples[i].future_uncertainties = unc
        samples[i].failure_within_0_5s = _compute_failure_labels(samples, i)

    return TemporalSequence(
        samples=samples,
        trajectory_type=trajectory_type,
        disturbance=disturbance,
        seed=seed,
    )


TRAJECTORY_CONFIGS: dict[str, dict] = {
    "straight_line": {
        "default": {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.3, "vy": 0.2},
        "fast": {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.8, "vy": 0.5},
        "slow": {"x0": 1000.0, "y0": 1000.0, "z0": 100.0, "vx": 0.1, "vy": 0.05},
        "offset": {"x0": 800.0, "y0": 1200.0, "z0": 100.0, "vx": 0.4, "vy": -0.3},
    },
    "circular": {
        "default": {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius": 0.5, "angular_speed_rad_s": 0.3},
        "fast": {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "radius": 0.8, "angular_speed_rad_s": 0.6},
    },
    "figure_8": {
        "default": {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3},
    },
    "sinusoidal": {
        "default": {"cx": 1000.0, "cy": 1000.0, "cz": 100.0, "amplitude_x": 0.5, "amplitude_y": 0.3, "freq_x": 0.3, "freq_y": 0.2},
    },
}

DISTURBANCE_PROFILES = ["clear", "light"]


def generate_temporal_dataset(
    output_dir: str = "artifacts/datasets/temporal-v1",
    sequences_per_config: int = 3,
    duration_s: float = 10.0,
    dt: float = 1.0 / 30.0,
    seed: int = 42,
    save: bool = True,
) -> dict[str, TemporalDatasetSplit]:
    """Generate full temporal dataset with train/val/test/hard_test splits.

    Uses multiple trajectory types, disturbance profiles, and parameter
    variations to create diverse training data.
    """
    rng = np.random.default_rng(seed)

    configs_to_run: list[tuple[str, str, str, int]] = []
    for traj_type, variants in TRAJECTORY_CONFIGS.items():
        for variant_name, params in variants.items():
            for dist in DISTURBANCE_PROFILES:
                for _ in range(sequences_per_config):
                    seq_seed = int(rng.integers(0, 1_000_000))
                    params_copy = dict(params)
                    if traj_type == "random" and "seed" in params_copy:
                        params_copy["seed"] = seq_seed
                    configs_to_run.append((traj_type, variant_name, dist, seq_seed))

    rng.shuffle(configs_to_run)

    n_total = len(configs_to_run)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_total - n_train - n_val

    train_configs = configs_to_run[:n_train]
    val_configs = configs_to_run[n_train:n_train + n_val]
    test_configs = configs_to_run[n_train + n_val:]

    hard_test_configs: list[tuple[str, str, str, int]] = []
    for traj_type in ["straight_line", "circular", "sinusoidal"]:
        for dist in ["light"]:
            for _ in range(sequences_per_config):
                seq_seed = int(rng.integers(0, 1_000_000))
                hard_test_configs.append((traj_type, "fast", dist, seq_seed))

    def _build_split(name: str, configs: list[tuple[str, str, str, int]]) -> TemporalDatasetSplit:
        sequences = []
        for traj_type, _variant, dist, seq_seed in configs:
            params = dict(TRAJECTORY_CONFIGS[traj_type]["default"])
            if traj_type == "random" and "seed" in params:
                params["seed"] = seq_seed
            try:
                seq = _run_trajectory(
                    trajectory_type=traj_type,
                    traj_params=params,
                    disturbance=dist,
                    seed=seq_seed,
                    duration_s=duration_s,
                    dt=dt,
                )
                if seq.samples:
                    sequences.append(seq)
            except Exception:
                continue
        return TemporalDatasetSplit(
            sequences=sequences,
            split_name=name,
            config={
                "num_configs": len(configs),
                "duration_s": duration_s,
                "dt": dt,
                "seed": seed,
            },
        )

    splits = {
        "train": _build_split("train", train_configs),
        "validation": _build_split("validation", val_configs),
        "test": _build_split("test", test_configs),
        "hard_test": _build_split("hard_test", hard_test_configs),
    }

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset_version": "temporal-observable-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons_s": HORIZONS_S,
            "feature_dim": FEATURE_DIM,
            "sequence_length": SEQUENCE_LENGTH,
            "splits": {},
        }
        for name, split in splits.items():
            arrs = split.to_arrays()
            split_dir = out / name
            split_dir.mkdir(exist_ok=True)
            if arrs:
                for key, arr in arrs.items():
                    np.save(split_dir / f"{key}.npy", arr)
            seq_meta = []
            for seq in split.sequences:
                seq_meta.append({
                    "trajectory_type": seq.trajectory_type,
                    "disturbance": seq.disturbance,
                    "seed": seq.seed,
                    "num_samples": len(seq.samples),
                })
            (split_dir / "sequences.json").write_text(
                json.dumps(seq_meta, indent=2), encoding="utf-8"
            )
            manifest["splits"][name] = {
                "num_sequences": split.num_sequences,
                "num_samples": split.num_samples,
            }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return splits
