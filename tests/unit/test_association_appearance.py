"""Appearance-gated association: anti-hijack behavior (measured).

When several candidates pass the geometric gate, the appearance term
(size/brightness distance to the track's running signature) should
prefer the familiar beacon over a nearer unfamiliar glint.

Every number asserted here is measured by running these tests — the
technical report quotes exactly these trials.
"""

from __future__ import annotations

import math
import random

import pytest

from fsoc_tracker.perception.models import BeaconDetection, PerceptionStatus
from fsoc_tracker.tracking.config import TrackerConfig
from fsoc_tracker.tracking.tracker import KalmanTracker
from fsoc_tracker.tracking.association import associate_with_appearance


def _det(
    x: float,
    y: float,
    *,
    conf: float = 0.9,
    area: float = 78.0,
    intensity: float = 240.0,
    ts: float = 0.0,
    flagged: bool = True,
) -> BeaconDetection:
    return BeaconDetection(
        detected=flagged,
        center_x=float(x),
        center_y=float(y),
        confidence=float(conf),
        area=float(area),
        mean_intensity=float(intensity),
        timestamp_s=float(ts),
        visibility_state=PerceptionStatus.DETECTED if flagged else PerceptionStatus.NO_TARGET,
    )


def _cfg(**overrides) -> TrackerConfig:
    defaults = dict(
        association_gate_px=80.0,
        acquisition_min_consecutive_hits=2,
        appearance_weight=0.0,
    )
    defaults.update(overrides)
    return TrackerConfig(**defaults)


class TestAssociateWithAppearance:
    """Single-frame choice under close geometric competition."""

    def test_zero_weight_picks_nearer_glint(self):
        # Prediction at (320, 240): glint 5 px away, familiar beacon 12 px.
        cfg = _cfg(appearance_weight=0.0)
        glint = _det(325.0, 240.0, area=15.0, intensity=80.0)
        beacon = _det(332.0, 240.0, area=78.0, intensity=240.0)
        out = associate_with_appearance(
            [glint, beacon], (320.0, 240.0), cfg, (78.0, 240.0),
        )
        assert out is glint

    def test_weighted_picks_familiar_over_nearer_glint(self):
        # Identical geometry; appearance term must flip the choice.
        cfg = _cfg(appearance_weight=8.0)
        glint = _det(325.0, 240.0, area=15.0, intensity=80.0)
        beacon = _det(332.0, 240.0, area=78.0, intensity=240.0)
        out = associate_with_appearance(
            [glint, beacon], (320.0, 240.0), cfg, (78.0, 240.0),
        )
        assert out is beacon

    def test_none_signature_matches_geometric(self):
        cfg = _cfg(appearance_weight=8.0)
        glint = _det(325.0, 240.0, area=15.0, intensity=80.0)
        beacon = _det(332.0, 240.0, area=78.0, intensity=240.0)
        out = associate_with_appearance(
            [glint, beacon], (320.0, 240.0), cfg, None,
        )
        assert out is glint

    def test_gate_still_rejects_far(self):
        cfg = _cfg(appearance_weight=8.0, association_gate_px=10.0)
        far = _det(500.0, 240.0, area=78.0, intensity=240.0)
        out = associate_with_appearance(
            [far], (320.0, 240.0), cfg, (78.0, 240.0),
        )
        assert out is None


class TestPlantedGlintHijack:
    """End-to-end hijack trials: maneuver displacement + planted glint.

    Protocol per seeded trial (30 seeds):
      1. Acquire on the beacon alone (15 frames) — signature seeds from
         the beacon.
      2. The beacon teleports within the association gate (a sharp
         maneuver Kalman lag cannot absorb in one step), displacing the
         prediction from the true beacon.
      3. For the remaining frames a glint is planted at a fixed offset
         from the beacon *toward the prediction* — always geometrically
         nearer to the track than the beacon is.

    Weight 0 (geometry only) is dragged onto the glint.  Weight 8 pays
    an appearance penalty large enough (measured in
    TestAssociateWithAppearance) to keep choosing the familiar beacon
    while the offset stays below that margin.  A trial counts as a
    hijack when the final estimate is closer to the glint than to the
    beacon.  Counts below are measured by running this test — the
    technical report quotes exactly these trials.
    """

    FRAMES_BEFORE = 15
    FRAMES_AFTER = 45

    def _run_trial(self, seed: int, weight: float) -> bool:
        rng = random.Random(seed)
        cfg = _cfg(appearance_weight=weight)
        trk = KalmanTracker(cfg)
        dt = 1.0 / 30.0
        ts = 0.0

        bx = 300.0 + rng.uniform(-20.0, 20.0)
        by = 240.0 + rng.uniform(-15.0, 15.0)
        vx = rng.uniform(0.5, 2.0)
        vy = rng.uniform(-0.5, 0.5)
        jump = rng.uniform(25.0, 40.0)
        jump_ang = rng.uniform(0.0, 2.0 * math.pi)
        offset = rng.uniform(3.0, 12.0)

        last_est = (bx, by)
        glint_pos = (bx, by)
        beacon_pos = (bx, by)
        jumped = False

        for i in range(self.FRAMES_BEFORE + self.FRAMES_AFTER):
            beacon_pos = (bx + vx * ts, by + vy * ts)
            if i == self.FRAMES_BEFORE:
                # Sharp maneuver: prediction still at the old location.
                beacon_pos = (
                    beacon_pos[0] + jump * math.cos(jump_ang),
                    beacon_pos[1] + jump * math.sin(jump_ang),
                )
                jumped = True

            if not jumped:
                dets = [
                    _det(beacon_pos[0], beacon_pos[1], conf=0.95,
                         area=78.0, intensity=240.0, ts=ts),
                ]
                glint_pos = beacon_pos
            else:
                # Glint: fixed offset from beacon toward the prediction.
                ex, ey = last_est
                dx, dy = ex - beacon_pos[0], ey - beacon_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 1e-6:
                    ux, uy = dx / dist, dy / dist
                else:
                    ux, uy = 1.0, 0.0
                glint_pos = (
                    beacon_pos[0] + ux * offset,
                    beacon_pos[1] + uy * offset,
                )
                dets = [
                    _det(beacon_pos[0], beacon_pos[1], conf=0.95,
                         area=78.0, intensity=240.0, ts=ts),
                    _det(glint_pos[0], glint_pos[1], conf=0.93,
                         area=15.0, intensity=80.0, ts=ts),
                ]

            st = trk.update(dets, ts)
            last_est = (st.estimated_x, st.estimated_y)
            ts += dt

        d_beacon = ((last_est[0] - beacon_pos[0]) ** 2
                    + (last_est[1] - beacon_pos[1]) ** 2) ** 0.5
        d_glint = ((last_est[0] - glint_pos[0]) ** 2
                   + (last_est[1] - glint_pos[1]) ** 2) ** 0.5
        return d_glint < d_beacon

    def test_measured_hijack_counts(self):
        w0 = sum(self._run_trial(s, 0.0) for s in range(30))
        w8 = sum(self._run_trial(s, 8.0) for s in range(30))
        print(f"\nhijacks: weight=0 -> {w0}/30, weight=8 -> {w8}/30")
        # Geometry-only must be vulnerable — otherwise the trial does
        # not exercise the competition at all.
        assert w0 > 0, "weight=0 should hijack on some seeds"
        # Appearance gating must not be worse; target is strict gain.
        assert w8 <= w0
        assert w8 < w0, "appearance term must measurably reduce hijacks"


class TestSignatureLifecycle:
    """EMA signature learns from lock-quality frames only."""

    def test_signature_seeded_at_birth(self):
        trk = KalmanTracker(_cfg())
        trk.update([_det(320.0, 240.0, area=78.0, intensity=240.0)], 0.0)
        assert trk._appear_area == pytest.approx(78.0)
        assert trk._appear_intensity == pytest.approx(240.0)

    def test_signature_reset_on_track_reset(self):
        trk = KalmanTracker(_cfg())
        trk.update([_det(320.0, 240.0)], 0.0)
        trk._reset_track(1.0)
        assert trk._appear_area is None
        assert trk._appear_intensity is None
