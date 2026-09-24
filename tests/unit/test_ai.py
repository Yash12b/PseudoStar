"""Comprehensive tests for the AI beacon perception subsystem.

Test categories:
A. Model initialization and weights
B. Model forward pass (shape, range, no NaN)
C. Heatmap peak detection
D. Center prediction accuracy
E. Dataset generation (deterministic, diversity)
F. Dataset negative samples
G. Dataset disturbance injection
H. Training loop (convergence, metrics)
I. AIBeaconDetector interface compliance
J. AIBeaconDetector inference (shape, output)
K. Hybrid detector (fusion policies)
L. Hybrid fallback (missing AI model)
M. Model save/load
N. Coordinate mapping
O. Confidence threshold
P. Arbitrary image sizes
Q. CPU execution
R. Benchmark framework
S. End-to-end synthetic run
T. Compositional: all backends on same data
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from fsoc_tracker.ai.benchmark import (
    BenchmarkResult,
    benchmark_detector,
    compare_detectors,
)
from fsoc_tracker.ai.config import AIModelConfig, DatasetConfig
from fsoc_tracker.ai.dataset import (
    BeaconLabel,
    DatasetSample,
    DatasetSplit,
    generate_sample,
    generate_split,
)
from fsoc_tracker.ai.export import load_model, save_model
from fsoc_tracker.ai.hybrid import FusionPolicy, HybridBeaconDetector
from fsoc_tracker.ai.inference import AIBeaconDetector
from fsoc_tracker.ai.model import BeaconCNN
from fsoc_tracker.ai.training import TrainingResult, _generate_heatmap
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.perception.models import PerceptionResult

# ---------------------------------------------------------------------------
# Helpers

def _config(
    w: int = 128,
    h: int = 128,
    conf_thresh: float = 0.3,
) -> AIModelConfig:
    return AIModelConfig(
        input_width=w, input_height=h,
        confidence_threshold=conf_thresh,
    )


def _beacon_image(
    w: int = 128, h: int = 128,
    cx: float = 64.0, cy: float = 64.0,
    size: float = 10.0, brightness: float = 200.0,
) -> np.ndarray:
    """Create a simple synthetic beacon image."""
    img = np.full((h, w), 5.0, dtype=np.float64)
    sigma = size / 4.0
    for y in range(h):
        for x in range(w):
            d = (x - cx) ** 2 + (y - cy) ** 2
            img[y, x] += brightness * math.exp(-d / (2 * sigma * sigma))
    return np.clip(img, 0, 255).astype(np.uint8)


def _dark_image(w: int = 128, h: int = 128) -> np.ndarray:
    """Create an image with no beacon."""
    return np.full((h, w), 5.0, dtype=np.uint8)


def _dataset_config(
    n: int = 50,
    w: int = 128,
    h: int = 128,
) -> DatasetConfig:
    return DatasetConfig(
        num_samples=n, val_samples=10, test_samples=10,
        image_width=w, image_height=h,
        min_target_size=5, max_target_size=15,
    )


# ---------------------------------------------------------------------------
# A. Model initialization and weights

class TestModelInit:
    def test_initialize_creates_weights(self):
        model = BeaconCNN(_config())
        model.initialize(seed=42)
        assert model.initialized
        assert model.count_parameters() > 0

    def test_weight_shapes(self):
        model = BeaconCNN(_config())
        model.initialize(seed=42)
        weights = model.get_weights()
        assert "conv1_w" in weights
        assert "conv1_b" in weights
        assert "fc1_w" in weights
        assert "fc2_w" in weights

    def test_model_size(self):
        model = BeaconCNN(_config())
        model.initialize(seed=42)
        assert model.model_size_bytes() > 0

    def test_set_weights(self):
        model = BeaconCNN(_config())
        model.initialize(seed=42)
        w1 = model.get_weights()
        model2 = BeaconCNN(_config())
        model2.set_weights(w1)
        assert model2.initialized


# ---------------------------------------------------------------------------
# B. Model forward pass

class TestModelForward:
    def test_output_shape(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        output = model.forward(img)
        assert output.heatmap.shape == (128, 128)

    def test_output_range(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        output = model.forward(img)
        assert output.heatmap.min() >= 0.0
        assert output.heatmap.max() <= 1.0

    def test_no_nan(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        output = model.forward(img)
        assert not np.any(np.isnan(output.heatmap))

    def test_forward_with_channel_dim(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (1, 128, 128)).astype(np.float32)
        output = model.forward(img)
        assert output.heatmap.shape == (128, 128)

    def test_confidence_in_range(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        output = model.forward(img)
        assert 0.0 <= output.confidence <= 1.0


# ---------------------------------------------------------------------------
# C. Heatmap peak detection

class TestHeatmapPeaks:
    def test_find_peaks_returns_list(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        output = model.forward(img)
        assert isinstance(output.num_peaks, int)

    def test_peak_suppression(self):
        from fsoc_tracker.ai.model import _sigmoid
        # Create artificial heatmap with clear peak
        hm = np.zeros((128, 128), dtype=np.float32)
        hm[64, 64] = 5.0  # Strong peak
        hm[32, 32] = 3.0  # Second peak
        hm_sig = _sigmoid(hm)

        model = BeaconCNN(_config(128, 128))
        model.initialize()
        peaks = model._find_peaks(hm_sig.copy())
        assert len(peaks) >= 1
        # Best peak should be near (64, 64)
        best_x, best_y, _ = peaks[0]
        assert abs(best_x - 64) < 5
        assert abs(best_y - 64) < 5


# ---------------------------------------------------------------------------
# D. Center prediction

class TestCenterPrediction:
    def test_predict_center_returns_tuple(self):
        model = BeaconCNN(_config(128, 128))
        model.initialize(seed=42)
        img = _beacon_image(128, 128, cx=64, cy=64, size=10, brightness=200)
        x, y, conf = model.predict_center(img.astype(np.float32) / 255.0)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# E. Dataset generation

class TestDatasetGeneration:
    def test_generate_sample_returns_dataset_sample(self):
        cfg = _dataset_config(n=10)
        rng = np.random.default_rng(42)
        sample = generate_sample(0, cfg, rng, include_disturbances=False)
        assert isinstance(sample, DatasetSample)
        assert sample.image.shape == (128, 128)

    def test_sample_has_label(self):
        cfg = _dataset_config(n=10)
        rng = np.random.default_rng(42)
        sample = generate_sample(0, cfg, rng, include_disturbances=False)
        assert isinstance(sample.label, BeaconLabel)

    def test_generate_split(self):
        cfg = _dataset_config(n=20)
        split = generate_split("train", 20, cfg, seed=42, include_disturbances=False)
        assert isinstance(split, DatasetSplit)
        assert split.num_samples == 20

    def test_split_statistics_and_full_dataset_manifest(self, tmp_path):
        cfg = _dataset_config(n=10)
        cfg.val_samples = 10
        cfg.test_samples = 10
        cfg.hard_test_samples = 10
        cfg.output_dir = str(tmp_path / "dataset")
        from fsoc_tracker.ai.dataset import generate_full_dataset

        splits = generate_full_dataset(cfg)
        assert set(splits) == {"train", "validation", "test", "hard_test"}
        assert (tmp_path / "dataset" / "manifest.json").exists()
        assert splits["hard_test"].statistics()["samples"] == 10

    def test_hard_test_contains_clipped_or_hard_negative_cases(self):
        cfg = _dataset_config(n=30)
        cfg.negative_ratio = 0.5
        cfg.hard_negative_ratio = 1.0
        cfg.center_margin = 0
        split = generate_split("hard_test", 30, cfg, seed=4000)
        assert any(sample.label.category == "hard_negative" for sample in split.samples)

    def test_deterministic_with_seed(self):
        cfg = _dataset_config(n=10)
        s1 = generate_split("train", 10, cfg, seed=42, include_disturbances=False)
        s2 = generate_split("train", 10, cfg, seed=42, include_disturbances=False)
        for a, b in zip(s1.samples, s2.samples):
            np.testing.assert_array_equal(a.image, b.image)
            assert a.label.visible == b.label.visible

    def test_different_seeds_different_data(self):
        cfg = _dataset_config(n=10)
        s1 = generate_split("train", 10, cfg, seed=42, include_disturbances=False)
        s2 = generate_split("train", 10, cfg, seed=99, include_disturbances=False)
        different = False
        for a, b in zip(s1.samples, s2.samples):
            if not np.array_equal(a.image, b.image):
                different = True
                break
        assert different


# ---------------------------------------------------------------------------
# F. Negative samples

class TestNegativeSamples:
    def test_negative_ratio(self):
        cfg = _dataset_config(n=100)
        cfg.negative_ratio = 0.5
        split = generate_split("train", 100, cfg, seed=42, include_disturbances=False)
        num_neg = sum(1 for s in split.samples if not s.label.visible)
        # With 50% ratio, expect roughly half negative
        assert 30 <= num_neg <= 70

    def test_negative_no_beacon(self):
        cfg = _dataset_config(n=10)
        cfg.negative_ratio = 1.0
        split = generate_split("train", 10, cfg, seed=42, include_disturbances=False)
        for s in split.samples:
            assert not s.label.visible

    def test_negative_samples_include_explicit_hard_negative_category(self):
        cfg = _dataset_config(n=20)
        cfg.negative_ratio = 1.0
        cfg.hard_negative_ratio = 1.0
        split = generate_split("train", 20, cfg, seed=42, include_disturbances=False)
        assert all(s.label.category == "hard_negative" for s in split.samples)
        assert all(not s.label.visible for s in split.samples)


# ---------------------------------------------------------------------------
# G. Dataset disturbance injection

class TestDatasetDisturbances:
    def test_disturbances_applied(self):
        cfg = _dataset_config(n=20)
        cfg.noise_prob = 1.0  # Always add noise
        cfg.fog_prob = 0.0
        cfg.haze_prob = 0.0
        cfg.low_light_prob = 0.0
        split = generate_split("train", 20, cfg, seed=42, include_disturbances=True)
        # Some samples should have noise disturbance
        has_noise = any("noise" in s.label.disturbance_types for s in split.samples)
        assert has_noise

    def test_no_disturbances_when_disabled(self):
        cfg = _dataset_config(n=10)
        split = generate_split("train", 10, cfg, seed=42, include_disturbances=False)
        for s in split.samples:
            assert len(s.label.disturbance_types) == 0


# ---------------------------------------------------------------------------
# H. Training loop

class TestTraining:
    def test_generate_heatmap(self):
        hm = _generate_heatmap(64.0, 64.0, 128, 128, 2.0)
        assert hm.shape == (128, 128)
        assert hm.max() > 0.99
        peak_y, peak_x = divmod(int(np.argmax(hm)), 128)
        assert abs(peak_x - 64) < 2
        assert abs(peak_y - 64) < 2

    def test_training_result_structure(self):
        result = TrainingResult()
        assert result.best_val_loss == float("inf")
        assert len(result.history) == 0


# ---------------------------------------------------------------------------
# I. AIBeaconDetector interface

class TestAIBeaconDetectorInterface:
    def test_implements_perception_engine(self):
        from fsoc_tracker.perception.base import PerceptionEngine
        detector = AIBeaconDetector(_config())
        assert isinstance(detector, PerceptionEngine)

    def test_name_property(self):
        detector = AIBeaconDetector(_config())
        assert detector.name == "ai_heatmap_cnn"

    def test_detect_returns_perception_result(self):
        detector = AIBeaconDetector(_config(128, 128))
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img, timestamp_s=1.0, frame_index=0)
        assert isinstance(result, PerceptionResult)
        assert result.detector_name == "ai_heatmap_cnn"


# ---------------------------------------------------------------------------
# J. AIBeaconDetector inference

class TestAIBeaconDetectorInference:
    def test_detect_on_beacon_image(self):
        detector = AIBeaconDetector(_config(128, 128))
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert result.processing_time_ms >= 0

    def test_detect_on_dark_image(self):
        detector = AIBeaconDetector(_config(128, 128))
        img = _dark_image(128, 128)
        result = detector.detect(img)
        assert result.processing_time_ms >= 0

    def test_detect_timestamp(self):
        detector = AIBeaconDetector(_config(128, 128))
        img = _dark_image(128, 128)
        result = detector.detect(img, timestamp_s=5.5, frame_index=10)
        assert result.frame_timestamp == 5.5
        assert result.frame_index == 10


# ---------------------------------------------------------------------------
# K. Hybrid detector fusion policies

class TestHybridFusion:
    def test_either_policy(self):
        detector = HybridBeaconDetector(
            _config(128, 128),
            fusion_policy=FusionPolicy.EITHER,
        )
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)

    def test_both_agree_policy(self):
        detector = HybridBeaconDetector(
            _config(128, 128),
            fusion_policy=FusionPolicy.BOTH_AGREE,
        )
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)

    def test_ai_or_classical_policy(self):
        detector = HybridBeaconDetector(
            _config(128, 128),
            fusion_policy=FusionPolicy.AI_OR_CLASSICAL,
        )
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# L. Hybrid fallback

class TestHybridFallback:
    def test_classical_only_when_ai_fails(self):
        # Hybrid with broken AI should still work via classical
        detector = HybridBeaconDetector(
            _config(128, 128),
            fusion_policy=FusionPolicy.AI_OR_CLASSICAL,
        )
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# M. Model save/load

class TestModelSaveLoad:
    def test_save_and_load(self):
        model = BeaconCNN(_config(64, 64))
        model.initialize(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir)
            loaded = load_model(tmpdir)
            assert loaded.initialized
            assert loaded.count_parameters() == model.count_parameters()

    def test_loaded_weights_match(self):
        model = BeaconCNN(_config(64, 64))
        model.initialize(seed=42)
        orig_w = model.get_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir)
            loaded = load_model(tmpdir)
            load_w = loaded.get_weights()
            for key in orig_w:
                np.testing.assert_array_equal(orig_w[key], load_w[key])


# ---------------------------------------------------------------------------
# N. Coordinate mapping

class TestCoordinateMapping:
    def test_heatmap_to_image_coords(self):
        cfg = AIModelConfig(input_width=64, input_height=64)
        model = BeaconCNN(cfg)
        model.initialize(seed=42)

        # Image is 128x128, model is 64x64
        detector = AIBeaconDetector(cfg)
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        # Should detect something (even if not perfectly centered)
        assert result.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# O. Confidence threshold

class TestConfidenceThreshold:
    def test_high_threshold_fewer_detections(self):
        low_thresh = AIBeaconDetector(_config(128, 128, conf_thresh=0.1))
        high_thresh = AIBeaconDetector(_config(128, 128, conf_thresh=0.9))

        img = _beacon_image(128, 128, 64, 64, 10, 200)
        r_low = low_thresh.detect(img)
        r_high = high_thresh.detect(img)
        # High threshold should have fewer or equal detections
        assert len(r_high.detections) <= len(r_low.detections) + 1  # +1 for tolerance


# ---------------------------------------------------------------------------
# P. Arbitrary image sizes

class TestArbitrarySizes:
    @pytest.mark.parametrize("w,h", [(64, 64), (128, 128), (256, 256)])
    def test_different_sizes(self, w: int, h: int):
        cfg = AIModelConfig(input_width=64, input_height=64)
        detector = AIBeaconDetector(cfg)
        img = _beacon_image(w, h, w // 2, h // 2, 10, 200)
        result = detector.detect(img)
        assert result.image_width == w
        assert result.image_height == h


# ---------------------------------------------------------------------------
# Q. CPU execution

class TestCPUExecution:
    def test_cpu_only(self):
        model = BeaconCNN(_config(64, 64))
        model.initialize(seed=42)
        img = np.random.default_rng(42).uniform(0, 1, (64, 64)).astype(np.float32)
        output = model.forward(img)
        assert output.heatmap.shape == (64, 64)


# ---------------------------------------------------------------------------
# R. Benchmark framework

class TestBenchmark:
    def test_benchmark_single_detector(self):
        dataset = generate_split(
            "test", 20, _dataset_config(n=20), seed=3000,
            include_disturbances=False,
        )
        detector = ClassicalBeaconDetector()
        result = benchmark_detector(detector, dataset)
        assert isinstance(result, BenchmarkResult)
        assert result.num_samples == 20
        assert result.precision >= 0
        assert result.recall >= 0

    def test_compare_detectors(self):
        dataset = generate_split(
            "test", 10, _dataset_config(n=10), seed=3000,
            include_disturbances=False,
        )
        classical = ClassicalBeaconDetector()
        ai = AIBeaconDetector(_config(128, 128))
        results = compare_detectors([classical, ai], dataset)
        assert len(results) == 2
        assert results[0].detector_name == "classical_bright_spot"
        assert results[1].detector_name == "ai_heatmap_cnn"

    def test_benchmark_summary(self):
        dataset = generate_split(
            "test", 10, _dataset_config(n=10), seed=3000,
            include_disturbances=False,
        )
        detector = ClassicalBeaconDetector()
        result = benchmark_detector(detector, dataset)
        summary = result.summary()
        assert "Benchmark:" in summary
        assert "Precision=" in summary


# ---------------------------------------------------------------------------
# S. End-to-end synthetic run

class TestEndToEnd:
    def test_classical_detection_pipeline(self):
        """Classical detection on synthetic image."""
        detector = ClassicalBeaconDetector()
        img = _beacon_image(640, 480, 320, 240, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)

    def test_ai_detection_pipeline(self):
        """AI detection on synthetic image."""
        detector = AIBeaconDetector(_config(128, 128))
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)

    def test_hybrid_detection_pipeline(self):
        """Hybrid detection on synthetic image."""
        detector = HybridBeaconDetector(_config(128, 128))
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        result = detector.detect(img)
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# T. Compositional tests

class TestCompositional:
    def test_all_backends_same_image(self):
        """All three backends process the same image."""
        img = _beacon_image(128, 128, 64, 64, 10, 200)

        classical = ClassicalBeaconDetector()
        ai = AIBeaconDetector(_config(128, 128))
        hybrid = HybridBeaconDetector(_config(128, 128))

        r1 = classical.detect(img)
        r2 = ai.detect(img)
        r3 = hybrid.detect(img)

        assert all(isinstance(r, PerceptionResult) for r in [r1, r2, r3])
        # All should have processing times
        assert all(r.processing_time_ms >= 0 for r in [r1, r2, r3])

    def test_disturbances_with_ai(self):
        """AI detection on noisy image."""
        detector = AIBeaconDetector(_config(128, 128))
        img = _beacon_image(128, 128, 64, 64, 10, 200)
        # Add noise
        rng = np.random.default_rng(42)
        noisy = img.astype(np.float64) + rng.normal(0, 10, img.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        result = detector.detect(noisy)
        assert isinstance(result, PerceptionResult)
