"""Optical link simulation for FSOC coarse alignment.

This is a SIMPLIFIED geometric abstraction — not a physics-based
communication simulator. It models the consequence of pointing
alignment on link availability.

ASSUMPTIONS (documented):
    1. Beam direction = Terminal A's optical axis (camera pan/tilt).
       The beam does NOT point at the true beacon position.
       Misalignment is the entire point — it shows tracking error.
    2. Angular error = angle between optical axis and direction to beacon.
    3. Beam alignment decays smoothly with angular error (Gaussian-like).
    4. Link quality = beam_alignment × atmospheric_factor × distance_factor.
       No beam divergence, no received power model, no SNR.
    5. Distance factor: quality degrades with range (inverse-square heuristic).
    6. Atmospheric factor: fog/rain/haze reduce quality multiplicatively.
    7. Hysteresis: status requires sustained alignment/error to transition.
       This prevents oscillation between LOCKED and DEGRADED.
    8. Terminal A is the local tracking system (camera).
       Terminal B is the remote beacon (target position from simulation).
    9. Terminal B's orientation is not used by this model.

CONCEPTUAL FLOW:
    tracking → coarse alignment → beam alignment → optical link availability

The link is subordinate to the camera-tracking mission. It visualizes
the consequence of tracking quality on the communication channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fsoc_tracker.simulation.terminal import TerminalState


class LinkStatus(str, Enum):
    """Link status state machine.

    Acquisition:
        NO_LINK → SEARCHING (poor) / ALIGNING (moderate) / LOCKED (good)
        SEARCHING → LOST (sustained poor) / ALIGNING / LOCKED
        ALIGNING → LOCKED (good) / LOST (sustained poor)
    Tracking:
        LOCKED ↔ DEGRADED → LOST → REACQUIRING → LOCKED
        REACQUIRING → LOST on regression.
    """
    NO_LINK = "NO_LINK"
    SEARCHING = "SEARCHING"
    ALIGNING = "ALIGNING"
    LOCKED = "LOCKED"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


@dataclass
class OpticalLinkState:
    status: LinkStatus = LinkStatus.NO_LINK
    angular_error_deg: float = 0.0
    beam_alignment_percent: float = 0.0
    range_m: float = 0.0
    link_quality_percent: float = 0.0

    # Tuning parameters
    alignment_threshold_deg: float = 0.5    # Soft threshold for LOCKED
    degraded_threshold_deg: float = 2.0     # Threshold for DEGRADED
    hysteresis_frames: int = 5              # Frames required to transition

    # Internal hysteresis counters
    _locked_count: int = field(default=0, repr=False)
    _degraded_count: int = field(default=0, repr=False)
    _lost_count: int = field(default=0, repr=False)


def atmospheric_attenuation_from_disturbance(config: Any) -> float:
    """Derive link attenuation (0–1) from a disturbance config.

    Heuristic only, not physics: fog/rain/haze reduce quality
    multiplicatively. Returns 1.0 when disturbances are disabled
    or the config shape is unrecognized (fail-open for the link;
    the image pipeline still shows the degradation).
    """
    try:
        if config is None or not getattr(config, "enabled", False):
            return 1.0
        atm = getattr(config, "atmosphere", None)
        if atm is None or not getattr(atm, "enabled", False):
            return 1.0
        fog = max(0.0, min(1.0, float(getattr(atm, "fog_strength", 0.0) or 0.0)))
        haze = max(0.0, min(1.0, float(getattr(atm, "haze_strength", 0.0) or 0.0)))
        rain = max(0.0, min(1.0, float(getattr(atm, "rain_density", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 1.0
    atten = 1.0
    if fog > 0:
        atten *= max(0.1, 1.0 - fog * 0.8)
    if rain > 0:
        atten *= max(0.2, 1.0 - rain * 0.6)
    if haze > 0:
        atten *= max(0.3, 1.0 - haze * 0.5)
    return atten


class OpticalLinkEngine:
    """Computes beam alignment and link state between terminals.

    The beam direction is ALWAYS Terminal A's optical axis (camera
    pointing direction). The angular error is the angle between this
    axis and the actual direction to Terminal B (beacon).

    When the camera tracks well → angular error is small → alignment
    is high → link is LOCKED.

    When the camera is misaligned → angular error grows → alignment
    drops → link degrades → eventually LOST.
    """

    def __init__(self) -> None:
        self.state = OpticalLinkState()

    def update(
        self,
        term_a: TerminalState,
        term_b: TerminalState,
        atmospheric_attenuation: float = 1.0,
    ) -> OpticalLinkState:
        """Update link given current terminal states.

        Args:
            term_a: Local terminal (camera/tracking system).
            term_b: Remote terminal (beacon).
            atmospheric_attenuation: 0.0–1.0 multiplier for atmospheric effects.
                1.0 = clear sky, 0.0 = fully opaque.
        """
        if not term_a.active or not term_b.active:
            self.state.status = LinkStatus.NO_LINK
            self.state.beam_alignment_percent = 0.0
            self.state.link_quality_percent = 0.0
            self.state.angular_error_deg = 0.0
            self.state.range_m = 0.0
            self._reset_hysteresis()
            return self.state

        # --- Geometry ---
        dx = term_b.x - term_a.x
        dy = term_b.y - term_a.y
        dz = term_b.z - term_a.z

        self.state.range_m = math.sqrt(dx*dx + dy*dy + dz*dz)
        if self.state.range_m < 0.001:
            self.state.range_m = 0.001

        # Direction to beacon (world space)
        nx = dx / self.state.range_m
        ny = dy / self.state.range_m
        nz = dz / self.state.range_m

        # Terminal A optical axis (from camera pan/tilt)
        ax, ay, az = term_a.optical_axis

        # Angular error = angle between optical axis and beacon direction
        dot = ax*nx + ay*ny + az*nz
        dot = max(-1.0, min(1.0, dot))
        self.state.angular_error_deg = math.degrees(math.acos(dot))

        # --- Beam alignment (smooth decay) ---
        # Gaussian-like decay centered at 0 error
        # At 0° error: ~100%, at threshold: ~37%, beyond: approaches 0
        sigma = self.state.alignment_threshold_deg
        error_rad = math.radians(self.state.angular_error_deg)
        sigma_rad = math.radians(sigma)
        alignment = math.exp(-0.5 * (error_rad / sigma_rad) ** 2)
        self.state.beam_alignment_percent = alignment * 100.0

        # --- Distance factor (inverse-square heuristic) ---
        # Normalized to 1000m reference range
        # At 1000m: factor = 1.0; at 2000m: factor ≈ 0.25
        ref_range = 1000.0
        dist_factor = min(1.0, (ref_range / max(self.state.range_m, 1.0)) ** 1.5)

        # --- Link quality ---
        atmo = max(0.0, min(1.0, atmospheric_attenuation))
        self.state.link_quality_percent = (
            self.state.beam_alignment_percent * dist_factor * atmo
        )

        # --- Status state machine with hysteresis ---
        err = self.state.angular_error_deg
        locked_thresh = self.state.alignment_threshold_deg
        degraded_thresh = self.state.degraded_threshold_deg

        if err <= locked_thresh:
            # Good alignment → candidate for LOCKED
            self.state._locked_count += 1
            self.state._degraded_count = 0
            self.state._lost_count = 0

            if self.state._locked_count >= self.state.hysteresis_frames:
                if self.state.status in (LinkStatus.NO_LINK, LinkStatus.SEARCHING,
                                          LinkStatus.ALIGNING, LinkStatus.DEGRADED) or self.state.status in (LinkStatus.LOST, LinkStatus.REACQUIRING):
                    self.state.status = LinkStatus.LOCKED

        elif err <= degraded_thresh:
            # Moderate alignment → step toward lock, or hold degraded.
            self.state._degraded_count += 1
            self.state._locked_count = 0
            self.state._lost_count = 0

            if self.state._degraded_count >= self.state.hysteresis_frames:
                if self.state.status == LinkStatus.LOCKED:
                    self.state.status = LinkStatus.DEGRADED
                elif self.state.status in (LinkStatus.NO_LINK, LinkStatus.SEARCHING):
                    self.state.status = LinkStatus.ALIGNING
                elif self.state.status == LinkStatus.LOST:
                    # Improving but not yet locked → reacquiring.
                    self.state.status = LinkStatus.REACQUIRING
                # ALIGNING / DEGRADED / REACQUIRING hold while moderate.

        else:
            # Poor alignment → acquisition search, then LOST. LOST holds
            # until the error improves (see branches above).
            self.state._lost_count += 1
            self.state._locked_count = 0
            self.state._degraded_count = 0

            if self.state.status in (LinkStatus.NO_LINK, LinkStatus.SEARCHING):
                # Acquisition in progress, not yet a tracked-then-lost link.
                self.state.status = LinkStatus.SEARCHING

            if self.state._lost_count >= self.state.hysteresis_frames:
                if self.state.status in (LinkStatus.LOCKED, LinkStatus.DEGRADED,
                                          LinkStatus.ALIGNING, LinkStatus.SEARCHING,
                                          LinkStatus.REACQUIRING):
                    self.state.status = LinkStatus.LOST
                # If already LOST, stay LOST (REACQUIRING requires improvement)

        return self.state

    def _reset_hysteresis(self) -> None:
        self.state._locked_count = 0
        self.state._degraded_count = 0
        self.state._lost_count = 0
