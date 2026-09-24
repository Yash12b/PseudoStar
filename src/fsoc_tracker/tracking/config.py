"""Tracker configuration.

Strongly typed Pydantic model for all tracking parameters.
Defaults align with SIH26169 reference conditions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FilterType(str, Enum):
    KALMAN_2D_CV = "kalman_2d_cv"


class AssociationMethod(str, Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    MAHALANOBIS = "mahalanobis"


class TimestampGapPolicy(str, Enum):
    REJECT = "reject"
    CLAMP = "clamp"
    RESET = "reset"


class TrackerConfig(BaseModel):
    """Configuration for the temporal tracking subsystem.

    Reference from SIH26169 PS:
        - acquisition <= 2 sec
        - re-acquisition <= 1 sec
        - tracking error <= 10 px
        - target loss < 5%
        - processing >= 20 FPS
    """

    filter_type: FilterType = FilterType.KALMAN_2D_CV

    process_noise_pos: float = Field(default=20.0, gt=0,
        description="Process noise for position (px^2/s^2). Higher = filter trusts measurements more.")
    process_noise_vel: float = Field(default=40.0, gt=0,
        description="Process noise for velocity (px^2/s^4). Higher = filter adapts faster to velocity changes.")

    measurement_noise_x: float = Field(default=5.0, gt=0,
        description="Measurement noise variance for x (px^2). Reflects detector precision.")
    measurement_noise_y: float = Field(default=5.0, gt=0,
        description="Measurement noise variance for y (px^2). Reflects detector precision.")

    initial_position_uncertainty: float = Field(default=100.0, gt=0,
        description="Initial position covariance (px^2). Large = uncertain initial position.")
    initial_velocity_uncertainty: float = Field(default=500.0, gt=0,
        description="Initial velocity covariance (px/s)^2. Large = uncertain initial velocity.")

    association_method: AssociationMethod = AssociationMethod.NEAREST_NEIGHBOR
    association_gate_px: float = Field(default=80.0, gt=0,
        description="Maximum Euclidean distance (px) for valid association.")
    association_gate_mahal: float = Field(default=9.21, gt=0,
        description="Mahalanobis distance threshold (chi-squared, df=2, p=0.01). "
        "Used when association_method=mahalanobis (recommended for "
        "glint-heavy fields; see benchmark --assoc-method).")
    association_gate_floor_px: float = Field(default=15.0, ge=0.0,
        description="Euclidean floor under Mahalanobis gating: very close "
        "candidates are always accepted (protects early-track lock).")

    minimum_detection_confidence: float = Field(default=0.2, ge=0, le=1,
        description="Minimum confidence to consider a detection valid.")

    # Appearance-gated association (anti-hijack): when several candidates
    # pass the geometric gate, prefer the one resembling the tracked
    # target's running size/brightness signature. Weight 0 (default)
    # reproduces pure-geometric association exactly.
    appearance_weight: float = Field(default=0.0, ge=0.0, le=20.0,
        description="Appearance term weight in association score.")
    appearance_area_scale: float = Field(default=100.0, gt=0,
        description="Normalizer for area differences (px^2).")
    appearance_intensity_scale: float = Field(default=100.0, gt=0,
        description="Normalizer for mean-intensity differences.")
    appearance_ema_alpha: float = Field(default=0.2, gt=0.0, le=1.0,
        description="EMA rate for the running appearance signature.")

    # Coded beacon identity (temporal on/off keying): the true beacon
    # blinks an operator-configured binary code; decoys blink a
    # different code or burn steady. Among geometrically gated
    # candidates, association prefers the code-matching one. Empty
    # expected code (default) disables identity entirely — association
    # is then bit-identical to the geometry/appearance path.
    identity_expected_code: str = Field(default="",
        description="Operator-configured binary beacon code, e.g. '10110010'.")
    identity_weight: float = Field(default=0.0, ge=0.0, le=160.0,
        description="Identity term weight in association score (px-equivalent).")
    identity_frames_per_bit: int = Field(default=3, ge=1, le=30,
        description="Frames per code bit (renderer modulation + decoder agree).")
    identity_chain_gate_px: float = Field(default=15.0, gt=0,
        description="Proximity gate chaining per-candidate intensity "
        "samples. Kept well under typical decoy spacing so nearby "
        "different-code sources do not merge into one chain.")
    identity_veto_threshold: float = Field(default=0.75, ge=0.0, le=1.0,
        description="Confident code match below this skips the candidate "
        "before ranking (coast through the true beacon's OFF gaps "
        "instead of hijacking). A steady burner matches half the code "
        "by construction, so the default also rejects unkeyed "
        "lookalikes.")

    # Second-chance (BYTE-style) rescue pass: when the primary
    # association fails on a live track, retry sub-threshold candidates
    # before declaring a miss. Bridges brief fades/occlusions without
    # re-initializing the track. Capped so persistent clutter cannot
    # hold a dead track alive.
    rescue_enabled: bool = Field(default=True,
        description="Enable the low-confidence rescue pass.")
    rescue_confidence_floor: float = Field(default=0.1, ge=0, le=1,
        description="Lower confidence bound for rescue candidates.")
    rescue_max_streak: int = Field(default=5, ge=1,
        description="Max consecutive rescue updates before forcing a miss.")

    acquisition_min_consecutive_hits: int = Field(default=3, ge=1,
        description="Minimum consecutive valid detections before transitioning to TRACKING.")
    acquisition_timeout_s: float = Field(default=2.0, gt=0,
        description="Maximum duration (s) in ACQUIRING state before falling back to SEARCHING.")

    max_prediction_duration_s: float = Field(default=1.0, gt=0,
        description="Maximum duration (s) to continue predicting without a measurement before declaring LOST.")
    reacquisition_timeout_s: float = Field(default=1.0, gt=0,
        description="Maximum duration (s) in REACQUIRING before falling back to SEARCHING.")

    lock_quality_threshold: float = Field(default=0.5, ge=0, le=1,
        description="Minimum quality score to declare LOCKED.")

    timestamp_gap_policy: TimestampGapPolicy = TimestampGapPolicy.CLAMP
    max_timestamp_gap_s: float = Field(default=5.0, gt=0,
        description="Maximum allowed gap (s) before applying gap policy.")

    def build_process_noise_matrix(self, dt: float) -> list[list[float]]:
        """Build 4x4 process noise covariance matrix Q for dt seconds.

        Constant-velocity model:
            Q = G * G^T * sigma^2
        where G is the discrete-time noise input matrix.
        """
        q_pos = self.process_noise_pos
        q_vel = self.process_noise_vel
        dt2 = dt * dt / 2.0
        return [
            [dt2*dt2 * q_pos, 0,                 dt2 * q_pos,  0],
            [0,                 dt2*dt2 * q_pos,  0,            dt2 * q_pos],
            [dt2 * q_pos,       0,                dt * q_vel,   0],
            [0,                 dt2 * q_pos,       0,           dt * q_vel],
        ]

    def build_measurement_noise_matrix(self) -> list[list[float]]:
        """Build 2x2 measurement noise covariance matrix R."""
        return [
            [self.measurement_noise_x, 0],
            [0, self.measurement_noise_y],
        ]
