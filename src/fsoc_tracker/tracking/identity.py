"""Coded optical beacon identity (temporal on/off keying).

AstraLock-X-style idea, measured here with an on/off ablation: the true
beacon blinks a binary code (brightness × code bit per frame); decoys
blink a different code or burn steady. The tracker chains per-candidate
intensity samples by proximity, decodes bits with a chain-adaptive
threshold, and scores correlation against the operator-configured
expected code. Association then prefers the code-matching candidate,
defeating lookalike hijacks that geometry and size/brightness cannot.

Honesty notes:
- The expected code is OPERATOR CONFIGURATION (like tuning a radio to
  a callsign), never simulator truth: it comes from TrackerConfig,
  set explicitly via CLI/GUI. Per-frame GT is never consulted.
- Decoding is purely observational (rendered pixel intensities).
- With no expected code configured, every function here is inert and
  association is bit-identical to the geometric path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Published benchmark codes: primary vs decoy. Balanced 4/4 duty cycle;
# decoy is the exact inverse — same brightness, same size, opposite
# phase — so only the temporal code separates them.
CODE_A = "10110010"
CODE_B = "01001101"

MIN_AMPLITUDE = 5.0  # legacy guard (kept for API stability)
# A '1' bit must be BRIGHT when present: ON keying deposits photons,
# so a present-but-dim sample is evidence AGAINST the code. Beacons
# render at ~90+ intensity units in the reference configuration.
IDENTITY_BRIGHT_MIN = 40.0
# Decode window: two full code periods. Confidence needs one full
# period actually observed (chain birth onward).
IDENTITY_WINDOW_PERIODS = 2


def code_bit_at(code: str, frame_index: int, frames_per_bit: int) -> int:
    """Bit transmitted at a frame index (ON/OFF keying).

    Empty code = steady ON (today's behavior everywhere, unchanged).
    """
    if not code:
        return 1
    fpb = max(1, int(frames_per_bit))
    return 1 if code[(int(frame_index) // fpb) % len(code)] == "1" else 0


def _valid_code(code: str) -> bool:
    return bool(code) and all(c in "01" for c in code) and "0" in code and "1" in code


@dataclass
class _Chain:
    """Proximity-chained intensity history of one candidate blob."""

    x: float = 0.0
    y: float = 0.0
    last_seen: int = -1
    birth_step: int = 0
    # frame_index -> mean intensity sample, present ONLY when the blob
    # was detected that frame. Absence is signal: OFF keying means the
    # beacon deposited no photons, so missing samples mark '0' bits.
    samples: dict[int, float] = field(default_factory=dict)


class CodeIdentityTracker:
    """Decodes temporal beacon codes from per-candidate intensities.

    One update() per frame with the full detection list; score(det)
    returns (match_fraction, confident) for association. The internal
    step counter aligns with the renderer's frame index because every
    production path performs exactly one tracker update per frame; an
    explicit frame_index in update metadata resyncs after seeks.
    """

    def __init__(
        self,
        expected_code: str = "",
        frames_per_bit: int = 3,
        chain_gate_px: float = 15.0,
        keep_periods: int = 4,
    ) -> None:
        self._code = str(expected_code or "")
        self._fpb = max(1, int(frames_per_bit))
        self._gate = max(1.0, float(chain_gate_px))
        self._keep_periods = max(2, int(keep_periods))
        self._chains: dict[int, _Chain] = {}
        self._next_id = 1
        self._step = -1  # increments to 0 on first update()

    @property
    def enabled(self) -> bool:
        return _valid_code(self._code)

    @property
    def code_length(self) -> int:
        return len(self._code)

    @property
    def period_frames(self) -> int:
        return len(self._code) * self._fpb if self._code else 0

    def reset(self) -> None:
        self._chains.clear()
        self._step = -1

    def update(self, detections: list, frame_index: int | None = None) -> None:
        """Chain this frame's candidates by proximity; record intensities."""
        if not self.enabled:
            return
        self._step = int(frame_index) if frame_index is not None else self._step + 1
        step = self._step
        max_age = self.period_frames * self._keep_periods

        # Greedy nearest-first matching: closest pair claims the chain.
        unmatched = list(detections)
        pairs: list[tuple[float, _Chain, object]] = []
        for det in unmatched:
            dx = float(getattr(det, "center_x", 0.0))
            dy = float(getattr(det, "center_y", 0.0))
            for ch in self._chains.values():
                dist = ((dx - ch.x) ** 2 + (dy - ch.y) ** 2) ** 0.5
                if dist <= self._gate:
                    pairs.append((dist, ch, det))
        pairs.sort(key=lambda p: p[0])
        claimed_chains: set[int] = set()
        claimed_dets: set[int] = set()
        for _, ch, det in pairs:
            if id(ch) in claimed_chains or id(det) in claimed_dets:
                continue
            claimed_chains.add(id(ch))
            claimed_dets.add(id(det))
            ch.x = float(getattr(det, "center_x", ch.x))
            ch.y = float(getattr(det, "center_y", ch.y))
            ch.last_seen = step
            ch.samples[step] = float(getattr(det, "mean_intensity", 0.0) or 0.0)
        for det in unmatched:
            if id(det) in claimed_dets:
                continue
            ch = _Chain(
                x=float(getattr(det, "center_x", 0.0)),
                y=float(getattr(det, "center_y", 0.0)),
                last_seen=step,
                birth_step=step,
                samples={step: float(getattr(det, "mean_intensity", 0.0) or 0.0)},
            )
            self._chains[self._next_id] = ch
            self._next_id += 1

        # Prune stale chains and old samples (bounded memory on long runs).
        cutoff = step - max_age
        dead = [cid for cid, ch in self._chains.items()
                if ch.last_seen < cutoff or ch.last_seen > step]
        for cid in dead:
            del self._chains[cid]
        for ch in self._chains.values():
            if len(ch.samples) > max_age + 1:
                for k in [k for k in ch.samples if k < cutoff]:
                    del ch.samples[k]

    def score(self, det: object) -> tuple[float, bool]:
        """(match_fraction, confident) of this detection vs expected code.

        No history or no observable blink -> (0.5, False): neutral, so
        association falls back to pure geometry until evidence exists.
        """
        if not self.enabled:
            return (0.5, False)
        ch = self._chain_for(det)
        if ch is None:
            return (0.5, False)
        return self._decode(ch)

    def _chain_for(self, det: object) -> _Chain | None:
        # Nearest recently-seen chain by POSITION, never by object id:
        # detection objects are short-lived per frame, so CPython id()
        # values get reused after collection and id-matching misfires.
        # score() runs in the same frame as update(), so the owning
        # chain was seen at the current step.
        dx = float(getattr(det, "center_x", 0.0))
        dy = float(getattr(det, "center_y", 0.0))
        best: _Chain | None = None
        best_key: tuple[int, float] | None = None
        for ch in self._chains.values():
            age = self._step - ch.last_seen
            if age < 0 or age > 1:
                continue
            d = ((dx - ch.x) ** 2 + (dy - ch.y) ** 2) ** 0.5
            if d > self._gate:
                continue
            key = (age, d)
            if best_key is None or key < best_key:
                best_key = key
                best = ch
        return best

    def _decode(self, ch: _Chain) -> tuple[float, bool]:
        """Correlate presence/brightness pattern against expected code.

        For each bit position over the decode window: an expected '1'
        needs the blob present most of the time AND bright; an expected
        '0' needs it mostly absent (OFF keying deposits nothing, so a
        dark frame yields no candidate at all). A steady burner matches
        exactly half the code by construction; an inverted code matches
        none. Confidence needs one full period observed since birth.
        """
        L = len(self._code)
        period = L * self._fpb
        window = IDENTITY_WINDOW_PERIODS * period
        start = max(ch.birth_step, self._step - window + 1)
        if self._step - start + 1 < period:
            return (0.5, False)
        hits = 0
        for p in range(L):
            frames = [f for f in range(start, self._step + 1)
                      if (f // self._fpb) % L == p]
            if not frames:
                continue
            det = [ch.samples[f] for f in frames if f in ch.samples]
            presence = len(det) / len(frames)
            bright = sum(det) / len(det) if det else 0.0
            if self._code[p] == "1":
                ok = presence >= 0.5 and bright >= IDENTITY_BRIGHT_MIN
            else:
                ok = presence <= 0.5
            hits += 1 if ok else 0
        return (hits / L, True)


__all__ = ["CODE_A", "CODE_B", "CodeIdentityTracker", "code_bit_at"]
