"""Coded beacon identity: temporal on/off keying vs decoy hijacks.

Covers code_bit_at timing, chain decoding (CODE_A vs inverted CODE_B
vs steady), association preference under the identity term, renderer
modulation (OFF frames deposit nothing), and tracker integration
(identity_match surfaced, disabled-by-default inert).
"""

from __future__ import annotations

import pytest

from fsoc_tracker.perception.models import BeaconDetection
from fsoc_tracker.tracking.config import TrackerConfig
from fsoc_tracker.tracking.identity import (
    CODE_A,
    CODE_B,
    CodeIdentityTracker,
    code_bit_at,
)


def _det(x: float, y: float, intensity: float,
         conf: float = 0.9) -> BeaconDetection:
    return BeaconDetection(detected=True, confidence=conf,
                           center_x=x, center_y=y,
                           mean_intensity=intensity, area=100.0)


def _feed(code_tracker: CodeIdentityTracker, code: str, x: float,
          frames: int, fpb: int = 3, base: float = 200.0) -> None:
    """Feed synthetic per-frame detections blinking a code.

    Production semantics: an OFF bit deposits no photons, so no
    candidate exists that frame — absence itself is the '0' signal.
    """
    for fi in range(frames):
        if code_bit_at(code, fi, fpb):
            code_tracker.update([_det(x, 240.0, base)], frame_index=fi)
        else:
            code_tracker.update([], frame_index=fi)


class TestCodeBitTiming:
    def test_empty_code_is_steady_on(self):
        assert [code_bit_at("", i, 3) for i in range(10)] == [1] * 10

    def test_bit_windows(self):
        assert code_bit_at("10", 0, 3) == 1
        assert code_bit_at("10", 2, 3) == 1
        assert code_bit_at("10", 3, 3) == 0
        assert code_bit_at("10", 5, 3) == 0
        assert code_bit_at("10", 6, 3) == 1  # wraps

    def test_codes_are_balanced_inverses(self):
        assert len(CODE_A) == len(CODE_B) == 8
        assert all((a != b) for a, b in zip(CODE_A, CODE_B, strict=True))
        assert CODE_A.count("1") == 4


class TestDecoding:
    def test_matching_code_scores_one_confident(self):
        t = CodeIdentityTracker(expected_code=CODE_A)
        _feed(t, CODE_A, 100.0, frames=48)
        det = _det(100.0, 240.0, 200.0)
        t.update([det], frame_index=48)
        match, confident = t.score(det)
        assert confident is True
        assert match == pytest.approx(1.0)

    def test_inverted_code_scores_zero_confident(self):
        t = CodeIdentityTracker(expected_code=CODE_A)
        _feed(t, CODE_B, 100.0, frames=48)
        det = _det(100.0, 240.0, 200.0)
        t.update([det], frame_index=48)
        match, confident = t.score(det)
        assert confident is True
        assert match == pytest.approx(0.0)

    def test_steady_source_matches_half_confident(self):
        # A steady burner is present at every bit position: it matches
        # the four '1's and misses the four '0's by construction. The
        # veto threshold (default 0.75), not confidence, rejects it.
        t = CodeIdentityTracker(expected_code=CODE_A)
        for fi in range(48):
            t.update([_det(100.0, 240.0, 200.0)], frame_index=fi)
        det = _det(100.0, 240.0, 200.0)
        t.update([det], frame_index=48)
        match, confident = t.score(det)
        assert confident is True
        assert match == pytest.approx(0.5)

    def test_short_history_is_not_confident(self):
        t = CodeIdentityTracker(expected_code=CODE_A)
        _feed(t, CODE_A, 100.0, frames=5)
        det = _det(100.0, 240.0, 200.0)
        t.update([det], frame_index=5)
        _, confident = t.score(det)
        assert confident is False

    def test_two_chains_tracked_independently(self):
        t = CodeIdentityTracker(expected_code=CODE_A)
        for fi in range(48):
            dets = []
            if code_bit_at(CODE_A, fi, 3):
                dets.append(_det(100.0, 240.0, 200.0))
            if code_bit_at(CODE_B, fi, 3):
                dets.append(_det(400.0, 240.0, 200.0))
            t.update(dets, frame_index=fi)
        t.update([_det(100.0, 240.0, 200.0),
                  _det(400.0, 240.0, 200.0)], frame_index=48)
        m_a, c_a = t.score(_det(100.0, 240.0, 200.0))
        m_b, c_b = t.score(_det(400.0, 240.0, 200.0))
        assert (c_a, c_b) == (True, True)
        assert m_a == pytest.approx(1.0)
        assert m_b == pytest.approx(0.0)

    def test_disabled_tracker_is_neutral(self):
        t = CodeIdentityTracker(expected_code="")
        assert t.enabled is False
        t.update([_det(100.0, 240.0, 200.0)], frame_index=0)
        assert t.score(_det(100.0, 240.0, 200.0)) == (0.5, False)

    def test_chains_bounded_and_pruned(self):
        t = CodeIdentityTracker(expected_code=CODE_A)
        for fi in range(300):
            # jumps beyond the chain gate: a fresh chain every frame;
            # pruning keeps only the recent age window alive.
            t.update([_det(float(fi) * 30.0, 240.0, 200.0)], frame_index=fi)
        assert len(t._chains) <= 100  # 4-period age window, not 300


class TestAssociationPreference:
    def _fed_tracker(self) -> CodeIdentityTracker:
        t = CodeIdentityTracker(expected_code=CODE_A)
        for fi in range(48):
            dets = []
            if code_bit_at(CODE_A, fi, 3):
                dets.append(_det(100.0, 240.0, 200.0))
            if code_bit_at(CODE_B, fi, 3):
                dets.append(_det(150.0, 240.0, 200.0))
            t.update(dets, frame_index=fi)
        return t

    def test_geometry_only_picks_nearer_decoy(self):
        from fsoc_tracker.tracking.association import (
            associate_with_appearance,
        )
        t = self._fed_tracker()
        cfg = TrackerConfig(identity_weight=0.0)
        beacon = _det(100.0, 240.0, 200.0)
        decoy = _det(142.0, 240.0, 200.0)
        t.update([beacon, decoy], frame_index=48)
        # Prediction sits on the decoy side: geometry prefers the decoy.
        out = associate_with_appearance(
            [beacon, decoy], (142.0, 240.0), cfg, None,
            identity_fn=t.score)
        assert out is decoy

    def test_identity_term_flips_to_coded_beacon(self):
        from fsoc_tracker.tracking.association import (
            associate_with_appearance,
        )
        t = self._fed_tracker()
        cfg = TrackerConfig(identity_weight=40.0)
        beacon = _det(100.0, 240.0, 200.0)
        decoy = _det(142.0, 240.0, 200.0)
        t.update([beacon, decoy], frame_index=48)
        out = associate_with_appearance(
            [beacon, decoy], (142.0, 240.0), cfg, None,
            identity_fn=t.score)
        assert out is beacon

    def test_unconfident_identity_leaves_geometry(self):
        from fsoc_tracker.tracking.association import (
            associate_with_appearance,
        )
        t = CodeIdentityTracker(expected_code=CODE_A)  # no history
        cfg = TrackerConfig(identity_weight=40.0)
        beacon = _det(100.0, 240.0, 200.0)
        decoy = _det(142.0, 240.0, 200.0)
        t.update([beacon, decoy], frame_index=0)
        out = associate_with_appearance(
            [beacon, decoy], (142.0, 240.0), cfg, None,
            identity_fn=t.score)
        assert out is decoy

    def test_confident_mismatch_vetoes_decoy_only_field(self):
        from fsoc_tracker.tracking.association import (
            associate_with_appearance,
        )
        t = self._fed_tracker()
        cfg = TrackerConfig(identity_weight=40.0)
        decoy = _det(142.0, 240.0, 200.0)
        t.update([decoy], frame_index=48)
        # True beacon dark this frame: only a known-bad decoy is gated.
        # Veto returns None so the tracker coasts instead of hijacking.
        out = associate_with_appearance(
            [decoy], (142.0, 240.0), cfg, None,
            identity_fn=t.score)
        assert out is None

    def test_confident_match_survives_veto(self):
        from fsoc_tracker.tracking.association import (
            associate_with_appearance,
        )
        t = self._fed_tracker()
        cfg = TrackerConfig(identity_weight=40.0)
        beacon = _det(100.0, 240.0, 200.0)
        t.update([beacon], frame_index=48)
        out = associate_with_appearance(
            [beacon], (100.0, 240.0), cfg, None,
            identity_fn=t.score)
        assert out is beacon


class TestRendererModulation:
    def _render_on_off(self, code: str, frame_index: int):
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState
        from fsoc_tracker.simulation.sensor.config import SensorConfig
        from fsoc_tracker.simulation.sensor.renderer import (
            VirtualSensorRenderer,
        )
        from fsoc_tracker.simulation.target import WorldTargetState
        cam = VirtualCamera(CameraState(
            position_x=1000.0, position_y=1000.0, position_z=50.0,
            horizontal_fov_deg=30.0, vertical_fov_deg=22.5,
            width=640, height=480))
        tgt = WorldTargetState(target_id=0, x=1000.0, y=1000.0, z=500.0,
                               brightness=1.0, size_px=10.0, code=code)
        cfg = SensorConfig(width=640, height=480)
        out = VirtualSensorRenderer(cfg).render(cam, [tgt], 0.0, frame_index)
        return out.image

    def test_on_bit_deposits_off_bit_does_not(self):
        # CODE_A[0] = '1': frames 0-2 ON; CODE_A[1] = '0': frames 3-5 OFF.
        img_on = self._render_on_off(CODE_A, 1)
        img_off = self._render_on_off(CODE_A, 4)
        assert int(img_on.max()) > 200
        assert int(img_off.max()) <= 10  # background only

    def test_empty_code_always_on(self):
        img = self._render_on_off("", 4)
        assert int(img.max()) > 200


class TestCodedWorld:
    def test_coded_profile_assigns_inverse_codes(self):
        from fsoc_tracker.simulation.world_builder import (
            build_benchmark_world,
        )
        w = build_benchmark_world("coded", seed=42)
        codes = {b.beacon_id: b.code for b in w.beacons}
        assert codes[0] == CODE_A
        assert w.primary_beacon_id == 0
        for bid in range(1, 6):
            assert codes[bid] == CODE_B
        # Equal brightness/size: code is the ONLY separator.
        assert {b.brightness for b in w.beacons} == {1.0}
        assert {b.size_px for b in w.beacons} == {10.0}


class TestTrackerIntegration:
    def test_identity_match_surfaced_and_default_neutral(self):
        from fsoc_tracker.tracking.tracker import KalmanTracker
        trk = KalmanTracker(TrackerConfig())
        assert trk._identity is None
        st = trk.update([_det(320.0, 240.0, 200.0)], 0.0)
        assert st.identity_match == pytest.approx(0.5)

    def test_enabled_identity_tracks_match(self):
        from fsoc_tracker.tracking.tracker import KalmanTracker
        cfg = TrackerConfig(identity_expected_code=CODE_A,
                            identity_weight=40.0)
        trk = KalmanTracker(cfg)
        assert trk._identity is not None and trk._identity.enabled
        for i in range(60):
            if code_bit_at(CODE_A, i, 3):
                trk.update([_det(320.0, 240.0, 200.0)], i / 30.0)
            else:
                trk.update([], i / 30.0)
        st = trk.get_state()
        assert st.identity_match == pytest.approx(1.0)
