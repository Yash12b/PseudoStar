"""Detection-to-track association and gating.

Provides nearest-neighbor association with Euclidean and Mahalanobis gating.
The tracker uses this to match incoming detections to existing tracks.
"""

from __future__ import annotations

import math

from fsoc_tracker.perception.models import BeaconDetection
from fsoc_tracker.tracking.config import AssociationMethod, TrackerConfig


def euclidean_distance(
    det_x: float, det_y: float,
    pred_x: float, pred_y: float,
) -> float:
    """Euclidean distance between detection and predicted position."""
    dx = det_x - pred_x
    dy = det_y - pred_y
    return math.sqrt(dx * dx + dy * dy)


def associate_nearest(
    detections: list[BeaconDetection],
    predicted_position: tuple[float, float],
    config: TrackerConfig,
    kalman_mahal_fn: object | None = None,
    min_confidence: float | None = None,
    ignore_detected_flag: bool = False,
) -> BeaconDetection | None:
    """Associate the best detection to the predicted position.

    Uses gating to reject implausible associations.

    Args:
        detections: List of candidate detections from perception.
        predicted_position: Kalman filter predicted position (px, py).
        config: Tracker configuration.
        kalman_mahal_fn: Optional callable(measurement) -> Mahalanobis distance.
        min_confidence: Confidence floor override (defaults to the
            config minimum; the rescue pass supplies a lower floor).
        ignore_detected_flag: Accept candidates regardless of their
            detected flag (rescue pass works on sub-threshold lists).

    Returns:
        The best associated detection, or None if no detection passes the gate.
    """
    if not detections:
        return None

    conf_floor = (config.minimum_detection_confidence
                  if min_confidence is None else min_confidence)

    pred_x, pred_y = predicted_position
    gate_px = config.association_gate_px
    use_mahal = config.association_method == AssociationMethod.MAHALANOBIS
    gate_mahal = config.association_gate_mahal

    best_det: BeaconDetection | None = None
    best_score = float("inf")

    for det in detections:
        if not ignore_detected_flag and not det.detected:
            continue
        if det.confidence < conf_floor:
            continue

        det_x, det_y = det.center_x, det.center_y
        euc_dist = euclidean_distance(det_x, det_y, pred_x, pred_y)

        # Euclidean gate
        if euc_dist > gate_px:
            continue

        # Mahalanobis gate with a Euclidean floor: early in a track the
        # covariance is tiny, which would reject valid measurements, so
        # anything very close is always accepted.
        if use_mahal and kalman_mahal_fn is not None:
            mahal = kalman_mahal_fn((det_x, det_y))
            floor = getattr(config, "association_gate_floor_px", 0.0) or 0.0
            if mahal > gate_mahal and euc_dist > floor:
                continue
            score = mahal
        else:
            score = euc_dist

        if score < best_score:
            best_score = score
            best_det = det

    return best_det


def associate_with_appearance(
    detections: list[BeaconDetection],
    predicted_position: tuple[float, float],
    config: TrackerConfig,
    appearance: tuple[float, float] | None,
    kalman_mahal_fn: object | None = None,
    min_confidence: float | None = None,
    ignore_detected_flag: bool = False,
    identity_fn: object | None = None,
) -> BeaconDetection | None:
    """Associate with geometric gating plus appearance/identity preference.

    Identical to associate_nearest when appearance is None or the
    configured appearance_weight is 0. Otherwise, among geometrically
    gated candidates, the score adds an appearance term (size/brightness
    distance to the running signature), so a nearby glint loses to the
    farther but familiar beacon. Guards the distractor-hijack case.

    identity_fn(det) -> (match_fraction, confident): when the
    configured identity_weight is > 0, confident matches below the
    veto threshold skip the candidate outright (it cannot win no
    matter how close it is), and the survivors score minus weight ×
    match, so a code-matching candidate wins over a geometrically
    closer decoy. During the true beacon's OFF gaps every visible
    decoy is skipped and association returns None, so the tracker
    coasts on prediction instead of hijacking. Unconfident (0.5,
    False) scores leave the ranking untouched until code evidence
    exists.
    """
    if not detections:
        return None

    conf_floor = (config.minimum_detection_confidence
                  if min_confidence is None else min_confidence)
    weight = float(getattr(config, "appearance_weight", 0.0) or 0.0)
    id_weight = float(getattr(config, "identity_weight", 0.0) or 0.0)
    veto_thr = float(getattr(config, "identity_veto_threshold", 0.5))
    if id_weight > 0.0 and not callable(identity_fn):
        id_weight = 0.0

    pred_x, pred_y = predicted_position
    gate_px = config.association_gate_px
    use_mahal = config.association_method == AssociationMethod.MAHALANOBIS
    gate_mahal = config.association_gate_mahal

    best_det: BeaconDetection | None = None
    best_score = float("inf")

    for det in detections:
        if not ignore_detected_flag and not det.detected:
            continue
        if det.confidence < conf_floor:
            continue

        det_x, det_y = det.center_x, det.center_y
        euc_dist = euclidean_distance(det_x, det_y, pred_x, pred_y)

        # Euclidean gate
        if euc_dist > gate_px:
            continue

        # Mahalanobis gate with a Euclidean floor: early in a track the
        # covariance is tiny, which would reject valid measurements, so
        # anything very close is always accepted.
        if use_mahal and kalman_mahal_fn is not None:
            mahal = kalman_mahal_fn((det_x, det_y))
            floor = getattr(config, "association_gate_floor_px", 0.0) or 0.0
            if mahal > gate_mahal and euc_dist > floor:
                continue
            score = mahal
        else:
            score = euc_dist

        if weight > 0.0 and appearance is not None:
            ref_area, ref_int = appearance
            area_scale = float(getattr(config, "appearance_area_scale", 100.0)) or 100.0
            int_scale = float(getattr(config, "appearance_intensity_scale", 100.0)) or 100.0
            appear_dist = (abs(float(det.area) - ref_area) / area_scale
                           + abs(float(det.mean_intensity) - ref_int) / int_scale)
            score = score + weight * appear_dist

        if id_weight > 0.0:
            try:
                match, confident = identity_fn(det)
            except Exception:
                match, confident = 0.5, False
            if confident and float(match) < veto_thr:
                continue  # known-bad code: cannot win, whatever geometry says
            score = score - id_weight * float(match)

        if score < best_score:
            best_score = score
            best_det = det

    return best_det
