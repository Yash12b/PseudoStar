import pytest

from fsoc_tracker.simulation.communication import CommunicationEngine, MessageStatus
from fsoc_tracker.simulation.optical_link import (
    LinkStatus,
    OpticalLinkEngine,
    atmospheric_attenuation_from_disturbance,
)
from fsoc_tracker.simulation.terminal import TerminalState


def test_terminal_optical_axis():
    term = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    # Default is looking along +Z
    axis = term.optical_axis
    assert abs(axis[0] - 0.0) < 1e-6
    assert abs(axis[1] - 0.0) < 1e-6
    assert abs(axis[2] - 1.0) < 1e-6

    # Turn right 90 degrees (+X)
    term.yaw_deg = 90
    axis = term.optical_axis
    assert abs(axis[0] - 1.0) < 1e-6
    assert abs(axis[1] - 0.0) < 1e-6
    assert abs(axis[2] - 0.0) < 1e-6

def test_optical_link_angular_error():
    engine = OpticalLinkEngine()
    
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=0, y=0, z=1000) # Straight ahead
    
    # Hysteresis requires multiple aligned frames before LOCKED
    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert abs(state.angular_error_deg) < 1e-6
    assert state.beam_alignment_percent == 100.0
    assert state.status == LinkStatus.LOCKED
    
    # Move term_b slightly
    term_b.x = 1.0 # sin(angle) ~ 1/1000 rad ~ 0.057 deg
    state = engine.update(term_a, term_b)
    assert state.angular_error_deg > 0.05
    assert state.beam_alignment_percent > 0.0
    assert state.status == LinkStatus.LOCKED # Still within threshold, hysteresis keeps LOCKED

    # Move term_b outside degraded threshold — need sustained misalignment for LOST
    # First: transition from LOCKED → DEGRADED (5 frames), then DEGRADED → LOST (5 frames)
    term_b.x = 50.0 # ~ 2.87 deg > 2.0 deg degraded threshold
    for _ in range(12):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOST
    assert state.beam_alignment_percent < 1.0  # Gaussian decay, approaches 0

def test_communication_simulation_time_and_bidirectional():
    comm = CommunicationEngine()
    comm.transmission_latency_s = 0.5
    
    # Create message A -> B
    msg1 = comm.create_message("TERM_A", "TERM_B", "HELLO", 10.0)
    comm.send_message(msg1)
    
    assert msg1.status == MessageStatus.QUEUED
    
    # Mock link state — run enough updates for hysteresis to reach LOCKED
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=0, y=0, z=100)
    for _ in range(6):
        link = engine.update(term_a, term_b)
    
    # Process queue at t=10.1
    comm.update(link, 10.1)
    assert msg1.status == MessageStatus.TRANSMITTING
    
    # Process queue at t=10.6 (0.5s passed)
    comm.update(link, 10.6)
    assert msg1.status == MessageStatus.DELIVERED
    
    # Create message B -> A
    msg2 = comm.create_message("TERM_B", "TERM_A", "ACK", 11.0)
    comm.send_message(msg2)
    comm.update(link, 11.0)
    assert msg2.status == MessageStatus.TRANSMITTING
    
    # Break link — run enough updates for hysteresis to reach LOST
    term_b.x = 50.0
    
    for _ in range(12):
        link = engine.update(term_a, term_b)
        comm.update(link, 11.0 + _ * 0.01)
    assert msg2.status == MessageStatus.FAILED


def test_target_separation():
    # Demonstrating conceptual separation
    selected_object_id = 5
    designated_beacon_id = 5
    tracked_target_id = 2

    assert selected_object_id == designated_beacon_id
    assert designated_beacon_id != tracked_target_id


def test_optical_link_searching_then_lost():
    # Fresh engine with sustained poor alignment: acquisition is
    # SEARCHING first, then LOST after hysteresis.
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=50, y=0, z=100)  # ~26.6 deg error

    state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.SEARCHING

    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOST


def test_optical_link_fresh_moderate_aligning():
    # Fresh engine with sustained moderate error converges to ALIGNING.
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=1.5, y=0, z=100)  # ~0.86 deg, degraded band

    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.ALIGNING


def test_optical_link_lost_improves_to_reacquiring_then_locked():
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=50, y=0, z=100)

    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOST

    # Improve into the degraded band: LOST -> REACQUIRING.
    term_b.x = 1.5
    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.REACQUIRING

    # Realign fully: REACQUIRING -> LOCKED.
    term_b.x = 0.0
    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOCKED


def test_optical_link_reacquiring_regresses_to_lost():
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=0, y=0, z=0, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=50, y=0, z=100)

    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOST

    term_b.x = 1.5
    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.REACQUIRING

    term_b.x = 50.0
    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOST


def test_atmospheric_attenuation_helper():
    from fsoc_tracker.disturbances.config import (
        AtmosphereConfig,
        AtmosphereMode,
        DisturbanceConfig,
    )

    assert atmospheric_attenuation_from_disturbance(None) == 1.0
    assert atmospheric_attenuation_from_disturbance(DisturbanceConfig()) == 1.0

    # Disturbance master switch off -> fail open.
    cfg = DisturbanceConfig(
        enabled=False,
        atmosphere=AtmosphereConfig(
            enabled=True, mode=AtmosphereMode.FOG, fog_strength=0.7
        ),
    )
    assert atmospheric_attenuation_from_disturbance(cfg) == 1.0

    # Real nested fields (fog_strength etc.) drive the attenuation.
    # This guards against regressions to flat cfg.fog / cfg.rain access.
    cfg = DisturbanceConfig(
        enabled=True,
        atmosphere=AtmosphereConfig(
            enabled=True, mode=AtmosphereMode.FOG, fog_strength=0.5
        ),
    )
    atten = atmospheric_attenuation_from_disturbance(cfg)
    assert atten == pytest.approx(0.6)

    clear = DisturbanceConfig(
        enabled=True,
        atmosphere=AtmosphereConfig(enabled=True, mode=AtmosphereMode.CLEAR),
    )
    assert atmospheric_attenuation_from_disturbance(clear) == 1.0


def test_inactive_terminal_reports_no_link_without_stale_values():
    # No beacon (e.g. VIDEO/LIVE monocular): the engine must report
    # NO_LINK with zeroed metrics — never a fabricated range.
    engine = OpticalLinkEngine()
    term_a = TerminalState(x=1000, y=1000, z=50, yaw_deg=0, pitch_deg=0)
    term_b = TerminalState(x=1000, y=1000, z=500)

    for _ in range(6):
        state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.LOCKED

    term_b.active = False
    state = engine.update(term_a, term_b)
    assert state.status == LinkStatus.NO_LINK
    assert state.range_m == 0.0
    assert state.beam_alignment_percent == 0.0
    assert state.link_quality_percent == 0.0
    assert state.angular_error_deg == 0.0


def test_terminal_view_optical_axis_matches_terminal_state():
    # The GUI beam direction (TerminalView) must use the same convention
    # as the physics-side TerminalState: yaw=0/pitch=0 looks along +Z.
    from fsoc_tracker.gui.state import TerminalView

    for yaw, pitch in [(0, 0), (90, 0), (0, 30), (-45, -10), (2.0, -1.5)]:
        view = TerminalView(yaw_deg=yaw, pitch_deg=pitch)
        sim = TerminalState(yaw_deg=yaw, pitch_deg=pitch)
        for v, w in zip(view.optical_axis, sim.optical_axis):
            assert v == pytest.approx(w)
