"""Candidate scoring system.

Evaluates each candidate region against expected beacon properties
using configurable weighted scoring.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.perception.models import CandidateFeatures


def score_candidate(
    features: CandidateFeatures,
    config: PerceptionConfig,
) -> float:
    """Score a candidate region for beacon likelihood.

    Returns a normalized score in [0, 1].
    """
    if features.area <= 0:
        return 0.0

    scores = []
    weights = []

    intensity_score = _score_intensity(features, config)
    scores.append(intensity_score)
    weights.append(config.w_intensity)

    size_score = _score_size(features, config)
    scores.append(size_score)
    weights.append(config.w_size)

    shape_score = _score_shape(features, config)
    scores.append(shape_score)
    weights.append(config.w_shape)

    contrast_score = _score_contrast(features, config)
    scores.append(contrast_score)
    weights.append(config.w_contrast)

    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
    return float(weighted_sum / total_weight)


def _score_intensity(features: CandidateFeatures, config: PerceptionConfig) -> float:
    """Score based on intensity relative to expected beacon brightness."""
    if features.max_intensity <= 0:
        return 0.0

    brightness_ratio = features.mean_intensity / 255.0
    brightness_score = min(1.0, brightness_ratio * 2.0)

    return brightness_score


def _score_size(features: CandidateFeatures, config: PerceptionConfig) -> float:
    """Score based on candidate size relative to expected beacon size."""
    if features.area <= 0:
        return 0.0

    expected_area = config.expected_size_px ** 2
    area_diff = abs(features.area - expected_area)
    tolerance_area = config.size_tolerance_px ** 2

    if area_diff <= tolerance_area:
        return 1.0

    score = max(0.0, 1.0 - (area_diff - tolerance_area) / expected_area)
    return score


def _score_shape(features: CandidateFeatures, config: PerceptionConfig) -> float:
    """Score based on shape compactness and aspect ratio."""
    scores = []

    if features.circularity > 0:
        scores.append(min(1.0, features.circularity))

    if features.aspect_ratio > 0:
        ar = features.aspect_ratio
        if ar > 1.0:
            ar = 1.0 / ar
        scores.append(ar)

    return float(np.mean(scores)) if scores else 0.0


def _score_contrast(features: CandidateFeatures, config: PerceptionConfig) -> float:
    """Score based on local contrast."""
    if features.local_contrast <= 0:
        return 0.0

    contrast_score = min(1.0, features.local_contrast / 100.0)
    return contrast_score
