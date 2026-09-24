"""Classical bright-spot beacon detector.

Baseline implementation using thresholding, connected components,
and weighted centroid estimation.
"""

from __future__ import annotations

import time

import numpy as np

from fsoc_tracker.perception.candidates import extract_features, generate_candidates
from fsoc_tracker.perception.centroid import compute_centroid
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.perception.models import (
    BeaconDetection,
    PerceptionResult,
    PerceptionStatus,
    TargetClass,
)
from fsoc_tracker.perception.scoring import score_candidate


def detect_beacon(
    image: np.ndarray,
    config: PerceptionConfig,
    timestamp_s: float = 0.0,
    frame_index: int = 0,
) -> PerceptionResult:
    """Run classical bright-spot detection on a single frame.

    Args:
        image: Input grayscale image (uint8).
        config: Detection configuration.
        timestamp_s: Frame timestamp for metadata.
        frame_index: Frame index for metadata.

    Returns:
        PerceptionResult with detections and metadata.
    """
    t_start = time.perf_counter()

    result = PerceptionResult(
        detector_name="classical_bright_spot",
        frame_timestamp=timestamp_s,
        frame_index=frame_index,
    )

    if image is None or image.size == 0:
        result.status = PerceptionStatus.NO_TARGET
        result.processing_time_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    h, w = image.shape[:2]
    result.image_width = w
    result.image_height = h

    from fsoc_tracker.perception.preprocessing import preprocess_frame
    preprocessed = preprocess_frame(image, config)

    candidates = generate_candidates(
        preprocessed.image,
        preprocessed.background_level,
        config,
    )
    # A very small PSF can have only one or two pixels above a high
    # percentile threshold. Retry with a conservative lower threshold only
    # when the strict pass produced no candidates; this preserves noise
    # rejection for ordinary frames while keeping 5 px beacons observable.
    if not candidates and config.threshold_mode.value == "percentile":
        fallback_config = config.model_copy(update={
            "percentile_value": min(float(config.percentile_value), 80.0),
        })
        candidates = generate_candidates(
            preprocessed.image,
            preprocessed.background_level,
            fallback_config,
        )
    # Second fallback: even 80th percentile can miss video beacons
    # whose intensity is near the scene median (e.g. lit indoor scenes).
    if not candidates and config.threshold_mode.value == "percentile":
        fallback2_config = config.model_copy(update={
            "percentile_value": 50.0,
        })
        candidates = generate_candidates(
            preprocessed.image,
            preprocessed.background_level,
            fallback2_config,
        )

    result.diagnostics["num_candidates_raw"] = len(candidates)
    result.diagnostics["candidate_cap"] = config.max_candidates
    result.diagnostics["background_level"] = preprocessed.background_level
    result.diagnostics["noise_estimate"] = preprocessed.noise_estimate

    if not candidates:
        result.status = PerceptionStatus.NO_TARGET
        result.processing_time_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    result.status = PerceptionStatus.CANDIDATE

    scored_candidates: list[tuple[float, BeaconDetection]] = []
    # Masks arrive largest-first: fully process only the top-K so
    # noise-flood frames stay within budget; clean frames (< K) are
    # unaffected.
    for mask in candidates[:config.max_scored_candidates]:
        features = extract_features(preprocessed.image, mask)
        detection = _features_to_detection(
            features, config, timestamp_s, frame_index,
            image=preprocessed.image, mask=mask,
        )

        if detection is not None:
            sc = score_candidate(features, config)
            detection.confidence = sc
            scored_candidates.append((sc, detection))

    if not scored_candidates:
        result.status = PerceptionStatus.NO_TARGET
        result.processing_time_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    all_detections = [d for _, d in scored_candidates]
    result.detections = all_detections

    best_score, best_detection = scored_candidates[0]

    # Every candidate meeting the perception threshold is a confirmed
    # detection — not just the primary. The tracker's association stage
    # needs the full above-threshold set to choose among candidates;
    # flagging only the winner collapsed association to a single
    # candidate everywhere (benchmark and production alike).
    for sc, det in scored_candidates:
        if sc >= config.min_confidence:
            det.detected = True
            det.visibility_state = PerceptionStatus.DETECTED

    if best_score >= config.min_confidence:
        result.primary_detection = best_detection
        result.status = PerceptionStatus.DETECTED
    else:
        result.status = PerceptionStatus.UNCERTAIN

    result.processing_time_ms = (time.perf_counter() - t_start) * 1000.0
    return result


def _features_to_detection(
    features,
    config: PerceptionConfig,
    timestamp_s: float,
    frame_index: int,
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> BeaconDetection | None:
    """Convert candidate features to a BeaconDetection."""
    if features.area <= 0:
        return None

    if image is not None and mask is not None:
        centroid_x, centroid_y = compute_centroid(
            image, mask, method=config.centroid_method,
        )
    else:
        centroid_x, centroid_y = features.centroid_x, features.centroid_y

    return BeaconDetection(
        target_class=TargetClass.BEACON,
        center_x=centroid_x,
        center_y=centroid_y,
        bbox=features.bbox,
        width=features.width,
        height=features.height,
        area=features.area,
        mean_intensity=features.mean_intensity,
        max_intensity=features.max_intensity,
        integrated_intensity=features.integrated_intensity,
        local_contrast=features.local_contrast,
        timestamp_s=timestamp_s,
        frame_index=frame_index,
        algorithm="classical_bright_spot",
        visibility_state=PerceptionStatus.CANDIDATE,
    )
