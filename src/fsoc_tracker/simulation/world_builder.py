"""Generic configurable 3D virtual world builder.

Replaces the Scene 1-15 preset system for normal simulation operation.
A world is a :class:`ScenarioConfig` built explicitly: Terminal A pose,
an arbitrary list of independent beacons, one primary designation,
disturbance settings, bounds, and seeds. Nothing is prerecorded; every
beacon owns procedural motion state evaluated per frame by the engine.

Randomized generation is deterministic per master seed. Placement never
auto-aims the terminal at any beacon.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from fsoc_tracker.simulation.scenario import (
    BeaconConfig,
    ScenarioConfig,
    TerminalConfig,
)
from fsoc_tracker.simulation.trajectory.registry import TrajectoryRegistry

_REGISTRY = TrajectoryRegistry()

# Motion models guaranteed by the trajectory registry.
REQUIRED_MOTIONS = ("straight_line", "circular", "figure_8", "random")


def _available_motions() -> list[str]:
    names = _REGISTRY.available
    return sorted(names() if callable(names) else names)


def available_motions() -> list[str]:
    """Motion model names supported by the trajectory registry."""
    return _available_motions()


def create_world(
    width: float = 2000.0,
    height: float = 2000.0,
    depth: float = 2000.0,
    seed: int = 42,
    terminal_x: float = 1000.0,
    terminal_y: float = 1000.0,
    terminal_z: float = 50.0,
    terminal_yaw_deg: float = 0.0,
    terminal_pitch_deg: float = 0.0,
    name: str = "Custom World",
) -> ScenarioConfig:
    """Create an empty world with Terminal A placed but no beacons."""
    return ScenarioConfig(
        name=name,
        description="User-configured generic world",
        terminal=TerminalConfig(
            x=terminal_x, y=terminal_y, z=terminal_z,
            yaw_deg=terminal_yaw_deg, pitch_deg=terminal_pitch_deg,
        ),
        beacons=[],
        primary_beacon_id=None,
        world_width=width,
        world_height=height,
        world_depth=depth,
        seed=seed,
    )


def clear_world(cfg: ScenarioConfig) -> ScenarioConfig:
    """Remove all beacons and the primary designation (keeps terminal)."""
    cfg.beacons = []
    cfg.primary_beacon_id = None
    return cfg


# ---------------------------------------------------------------------------
# Terminal A
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), lo), hi)


def place_terminal_a(
    cfg: ScenarioConfig, x: float, y: float, z: float,
) -> TerminalConfig:
    """Place Terminal A at an arbitrary in-bounds location.

    Orientation is never touched: placing the terminal does not aim it
    at anything.
    """
    cfg.terminal.x = _clamp(x, 0.0, cfg.world_width)
    cfg.terminal.y = _clamp(y, 0.0, cfg.world_height)
    cfg.terminal.z = _clamp(z, 0.0, cfg.world_depth)
    return cfg.terminal


def move_terminal_a(
    cfg: ScenarioConfig, dx: float, dy: float, dz: float,
) -> TerminalConfig:
    """Translate Terminal A by a delta, clamped to world bounds."""
    t = cfg.terminal
    return place_terminal_a(cfg, t.x + dx, t.y + dy, t.z + dz)


def set_terminal_a_orientation(
    cfg: ScenarioConfig, yaw_deg: float, pitch_deg: float,
) -> TerminalConfig:
    """Set Terminal A orientation independently of any beacon."""
    cfg.terminal.yaw_deg = float(yaw_deg)
    cfg.terminal.pitch_deg = float(pitch_deg)
    return cfg.terminal


def set_terminal_motion(
    cfg: ScenarioConfig, motion: str = "drift", **params: float,
) -> TerminalConfig:
    """Give Terminal A itself a motion (e.g. two moving satellites).

    "static" (default) keeps today's fixed-tripod behavior. "drift"
    integrates constant velocity + attitude rates every engine step.
    Accepted params: vx, vy, vz (world units/s), yaw_rate_deg_s,
    pitch_rate_deg_s, roll_rate_deg_s. Unknown motion names raise
    ValueError immediately (never drift silently on a typo).
    """
    from fsoc_tracker.simulation.platform import PLATFORM_MOTIONS
    if motion not in PLATFORM_MOTIONS:
        raise ValueError(f"Unknown platform motion: {motion!r}")
    cfg.terminal.platform_motion = motion
    cfg.terminal.platform_motion_params = {k: float(v)
                                           for k, v in params.items()}
    return cfg.terminal


def get_terminal_a_state(cfg: ScenarioConfig) -> dict[str, float]:
    """Read back Terminal A pose for verification."""
    t = cfg.terminal
    return {"x": t.x, "y": t.y, "z": t.z,
            "yaw_deg": t.yaw_deg, "pitch_deg": t.pitch_deg}


# ---------------------------------------------------------------------------
# Beacons
# ---------------------------------------------------------------------------

def _next_beacon_id(cfg: ScenarioConfig) -> int:
    # Monotonic counter: IDs are never reused (tracking identity safety),
    # even after removals. Falls back to max+1 for hand-built configs.
    used = [b.beacon_id for b in cfg.beacons]
    bid = max(cfg.next_beacon_id, max(used, default=-1) + 1)
    cfg.next_beacon_id = bid + 1
    return bid


def add_beacon(
    cfg: ScenarioConfig,
    x: float, y: float, z: float,
    motion: str = "straight_line",
    motion_params: dict[str, Any] | None = None,
    seed: int = 42,
    size_px: float = 10.0,
    brightness: float = 1.0,
    shape: str = "square",
    code: str = "",
    active: bool = True,
    is_primary: bool = False,
) -> BeaconConfig:
    """Create an independent beacon. IDs auto-increment, never reused."""
    if motion not in _available_motions():
        raise ValueError(f"Unknown motion model: {motion}")
    beacon = BeaconConfig(
        beacon_id=_next_beacon_id(cfg),
        x0=float(x), y0=float(y), z0=float(z),
        trajectory=motion,
        trajectory_params=dict(motion_params or {}),
        seed=int(seed),
        size_px=float(size_px),
        brightness=float(brightness),
        shape=str(shape),
        code=str(code or ""),
        is_primary=bool(is_primary),
        active=bool(active),
    )
    cfg.beacons.append(beacon)
    if beacon.is_primary and cfg.primary_beacon_id is None:
        cfg.primary_beacon_id = beacon.beacon_id
    return beacon


def remove_beacon(cfg: ScenarioConfig, beacon_id: int) -> bool:
    """Remove a beacon. Clears primary designation if it was primary."""
    for i, b in enumerate(cfg.beacons):
        if b.beacon_id == beacon_id:
            del cfg.beacons[i]
            if cfg.primary_beacon_id == beacon_id:
                cfg.primary_beacon_id = None
            return True
    return False


def move_beacon(
    cfg: ScenarioConfig, beacon_id: int, x: float, y: float, z: float,
) -> bool:
    """Reposition a beacon's spawn/initial position (motion untouched)."""
    b = cfg.get_beacon(beacon_id)
    if b is None:
        return False
    b.x0, b.y0, b.z0 = float(x), float(y), float(z)
    return True


def set_beacon_motion(
    cfg: ScenarioConfig,
    beacon_id: int,
    motion: str,
    motion_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> bool:
    """Replace one beacon's motion model. Other beacons are untouched."""
    if motion not in _available_motions():
        raise ValueError(f"Unknown motion model: {motion}")
    b = cfg.get_beacon(beacon_id)
    if b is None:
        return False
    b.trajectory = motion
    b.trajectory_params = dict(motion_params or {})
    if seed is not None:
        b.seed = int(seed)
    return True


def set_manual_control(cfg: ScenarioConfig, beacon_id: int) -> bool:
    """Switch a beacon to user/keyboard control (keeps its position)."""
    b = cfg.get_beacon(beacon_id)
    if b is None:
        return False
    return set_beacon_motion(
        cfg, beacon_id, "user_controlled",
        {"x0": b.x0, "y0": b.y0, "z0": b.z0})


# ---------------------------------------------------------------------------
# Primary beacon
# ---------------------------------------------------------------------------

def set_primary_beacon(cfg: ScenarioConfig, beacon_id: int) -> bool:
    """Designate exactly one primary beacon.

    Drops the old relationship, establishes the new one. Trajectories
    of all beacons are untouched and nothing moves the camera.
    """
    target = cfg.get_beacon(beacon_id)
    if target is None:
        return False
    for b in cfg.beacons:
        b.is_primary = (b.beacon_id == beacon_id)
    cfg.primary_beacon_id = beacon_id
    return True


def clear_primary_beacon(cfg: ScenarioConfig) -> None:
    """Remove the primary designation (beacons keep moving)."""
    cfg.primary_beacon_id = None
    for b in cfg.beacons:
        b.is_primary = False


# ---------------------------------------------------------------------------
# Disturbances (scenario-level shorthand; pipeline consumes separately)
# ---------------------------------------------------------------------------

def set_disturbance(
    cfg: ScenarioConfig,
    preset: str = "clear",
    jitter_px: float = 0.0,
    noise_sigma: float = 0.0,
) -> None:
    """Configure scenario disturbances independently of any scene ID."""
    cfg.disturbances.enabled = preset != "clear"
    cfg.disturbances.preset = preset
    cfg.disturbances.jitter_px = float(jitter_px)
    cfg.disturbances.noise_sigma = float(noise_sigma)


def clear_disturbance(cfg: ScenarioConfig) -> None:
    """Disable all scenario disturbances."""
    set_disturbance(cfg)


# ---------------------------------------------------------------------------
# Reset / restart
# ---------------------------------------------------------------------------

def reset_world(cfg: ScenarioConfig) -> ScenarioConfig:
    """Return a pristine copy of the world configuration.

    Restores initial beacon/terminal/disturbance configuration. Runtime
    state (tracker, AI, search, ROI, link) is owned by the caller, which
    must reset those subsystems alongside ``engine.load_scenario(cfg)``
    (see ``restart_simulation``).
    """
    return copy.deepcopy(cfg)


def restart_simulation(engine: Any, cfg: ScenarioConfig) -> None:
    """Reload a world configuration into a live engine deterministically."""
    engine.load_scenario(cfg)


# ---------------------------------------------------------------------------
# Randomized generation
# ---------------------------------------------------------------------------

def _bearing_deg(
    tx: float, ty: float, tz: float,
    px: float, py: float, pz: float,
    yaw_deg: float, pitch_deg: float,
) -> tuple[float, float]:
    """Angular offset of a target from boresight in degrees."""
    dx, dy, dz = tx - px, ty - py, tz - pz
    yaw_off = math.degrees(math.atan2(dx, dz)) - yaw_deg
    horiz = math.hypot(dx, dz)
    pitch_off = math.degrees(math.atan2(dy, horiz)) - pitch_deg
    return yaw_off, pitch_off


_DIFFICULTY_BANDS = {
    # (min_abs_yaw_off_deg, max_abs_yaw_off_deg)
    "easy": (0.2, 1.0),     # inside FOV, moderately off-center
    "normal": (1.0, 2.5),   # significantly off-center
    "hard": (2.5, 8.0),     # near boundary or outside initial FOV
}


def randomize_world(
    master_seed: int = 12345,
    n_beacons: int = 5,
    world_bounds: tuple[float, float, float] = (2000.0, 2000.0, 2000.0),
    min_beacon_separation: float = 30.0,
    min_terminal_beacon_separation: float = 100.0,
    difficulty: str = "normal",
    motion_pool: tuple[str, ...] = (
        "straight_line", "circular", "figure_8", "random",
        "sinusoidal", "spiral",
    ),
) -> ScenarioConfig:
    """Generate a complete world deterministically from a master seed.

    Terminal A and beacons are placed within bounds with enforced
    minimum separations (no tiny-cluster spawns). Each beacon draws an
    independent motion model, parameters, and seed. The primary beacon
    is placed per difficulty band relative to boresight — never aimed
    at, and acquisition is never trivially centered.
    """
    import random

    if difficulty not in _DIFFICULTY_BANDS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    for motion in motion_pool:
        if motion not in _available_motions():
            raise ValueError(f"Unknown motion model: {motion}")

    rng = random.Random(master_seed)
    w, h, d = world_bounds
    lo_yaw, hi_yaw = _DIFFICULTY_BANDS[difficulty]

    cfg = create_world(width=w, height=h, depth=d, seed=master_seed,
                       name=f"Random World (seed {master_seed})")

    # Terminal A: random pose, random orientation.
    tx = rng.uniform(w * 0.2, w * 0.8)
    ty = rng.uniform(h * 0.3, h * 0.7)
    tz = rng.uniform(0.0, d * 0.2)
    place_terminal_a(cfg, tx, ty, tz)
    set_terminal_a_orientation(
        cfg, rng.uniform(-30.0, 30.0), rng.uniform(-10.0, 10.0))

    placed: list[tuple[float, float, float]] = [(tx, ty, tz)]

    def separated(x: float, y: float, z: float) -> bool:
        for i, (px, py, pz) in enumerate(placed):
            dist = math.dist((x, y, z), (px, py, pz))
            need = min_terminal_beacon_separation if i == 0 else min_beacon_separation
            if dist < need:
                return False
        return True

    def in_bounds(x: float, y: float, z: float) -> bool:
        return 0.0 <= x <= w and 0.0 <= y <= h and 0.0 <= z <= d

    def direction(yaw_off_deg: float, pitch_off_deg: float,
                  rng_range: float) -> tuple[float, float, float] | None:
        """Position at an angular offset from boresight and a range.

        Returns None when the point falls outside world bounds.
        """
        yaw = math.radians(cfg.terminal.yaw_deg + yaw_off_deg)
        pitch = math.radians(cfg.terminal.pitch_deg + pitch_off_deg)
        dx = math.sin(yaw) * math.cos(pitch) * rng_range
        dy = math.sin(pitch) * rng_range
        dz = math.cos(yaw) * math.cos(pitch) * rng_range
        x, y, z = tx + dx, ty + dy, tz + dz
        if not in_bounds(x, y, z):
            return None
        return (x, y, z)

    # Primary first: bearing in the difficulty band, near the boresight
    # plane (tight elevation), at a visible range. Never centered, never
    # aimed at — the terminal keeps its random orientation.
    primary = None
    for _ in range(2000):
        yaw_off = rng.uniform(lo_yaw, hi_yaw) * rng.choice((-1.0, 1.0))
        pitch_off = rng.uniform(-1.0, 1.0)
        rng_range = rng.uniform(200.0, 800.0)
        pos = direction(yaw_off, pitch_off, rng_range)
        if pos is None or not separated(*pos):
            continue
        primary = pos
        break
    if primary is None:  # deterministic fallback, still off-center
        primary = (tx + 0.03 * w, ty, min(tz + 400.0, d))
    px, py, pz = primary
    placed.append(primary)

    motion = rng.choice(list(motion_pool))
    first = add_beacon(
        cfg, px, py, pz, motion=motion,
        motion_params=_default_motion_params(motion, px, py, pz, rng),
        seed=rng.randint(0, 99999), is_primary=True,
    )
    cfg.primary_beacon_id = first.beacon_id

    # Distractors: independent models, params, seeds. Kept within a
    # plausible band around the boresight plane so they can actually
    # cross the FOV (fully invisible ones prove nothing).
    for _ in range(max(0, n_beacons - 1)):
        pos = None
        for _ in range(2000):
            yaw_off = rng.uniform(-15.0, 15.0)
            pitch_off = rng.uniform(-8.0, 8.0)
            rng_range = rng.uniform(150.0, 1200.0)
            cand = direction(yaw_off, pitch_off, rng_range)
            if cand is None or not separated(*cand):
                continue
            pos = cand
            break
        if pos is None:
            continue
        placed.append(pos)
        motion = rng.choice(list(motion_pool))
        add_beacon(
            cfg, pos[0], pos[1], pos[2], motion=motion,
            motion_params=_default_motion_params(
                motion, pos[0], pos[1], pos[2], rng),
            seed=rng.randint(0, 99999),
            brightness=round(rng.uniform(0.5, 1.0), 2),
        )
    return cfg


def _default_motion_params(
    motion: str, x: float, y: float, z: float, rng: Any,
) -> dict[str, Any]:
    """Sane procedural parameters placing motion around (x, y, z)."""
    if motion == "straight_line":
        return {"x0": x, "y0": y, "z0": z,
                "vx": rng.uniform(-1.0, 1.0),
                "vy": rng.uniform(-0.5, 0.5)}
    if motion == "circular":
        return {"cx": x, "cy": y, "cz": z,
                "radius": rng.uniform(2.0, 15.0),
                "angular_speed_rad_s": rng.uniform(0.1, 0.5)}
    if motion == "figure_8":
        return {"cx": x, "cy": y, "cz": z,
                "amplitude_x": rng.uniform(3.0, 12.0),
                "amplitude_y": rng.uniform(2.0, 8.0)}
    if motion == "random":
        return {"x0": x, "y0": y, "z0": z,
                "seed": rng.randint(0, 99999)}
    if motion == "sinusoidal":
        return {"cx": x, "cy": y, "cz": z,
                "amplitude_x": rng.uniform(3.0, 12.0),
                "amplitude_y": rng.uniform(2.0, 8.0),
                "freq_x": rng.uniform(0.1, 0.4),
                "freq_y": rng.uniform(0.1, 0.4)}
    if motion == "spiral":
        return {"cx": x, "cy": y, "cz": z,
                "radius_start": rng.uniform(2.0, 10.0),
                "radius_growth": rng.uniform(0.5, 2.0),
                "angular_speed_rad_s": rng.uniform(0.1, 0.4)}
    return {"x0": x, "y0": y, "z0": z}


# ---------------------------------------------------------------------------
# Benchmark worlds: deterministic generated configurations (no presets)
# ---------------------------------------------------------------------------

_BENCHMARK_PROFILES = (
    "nominal", "multi", "distractor", "coded", "moving", "loss", "noise",
    "fog", "jitter", "fast",
)


def build_benchmark_world(profile: str, seed: int = 42) -> ScenarioConfig:
    """Deterministic generated world for benchmarking (replaces scenes).

    ``profile`` selects the stress dimension; ``seed`` makes the run
    reproducible. No prerecorded behavior: motion is procedural per
    beacon with independent seeds derived from the master seed.
    """
    import random

    if profile not in _BENCHMARK_PROFILES:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    rng = random.Random(seed)
    cfg = create_world(seed=seed, name=f"Benchmark/{profile}/{seed}")
    place_terminal_a(cfg, 1000.0, 1000.0, 50.0)
    set_terminal_a_orientation(cfg, 0.0, 0.0)

    if profile == "nominal":
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.3, "vy": 0.2},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
    elif profile == "multi":
        for i, (radius, speed) in enumerate(
                ((2.0, 0.3), (3.0, 0.2), (1.5, 0.4))):
            add_beacon(
                cfg, 1000.0, 1000.0, 500.0, motion="circular",
                motion_params={"cx": 1000.0, "cy": 1000.0, "cz": 500.0,
                               "radius": radius,
                               "angular_speed_rad_s": speed},
                seed=rng.randint(0, 99999), is_primary=(i == 0))
        cfg.primary_beacon_id = 0
    elif profile == "distractor":
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.1, "vy": 0.02},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
        for i in range(1, 6):
            add_beacon(
                cfg, 1000.0 + (i - 3) * 1.5, 1000.0 + (i % 3 - 1) * 1.0,
                500.0 + i * 20, motion="straight_line",
                motion_params={"vx": 0.05 * (i + 1), "vy": 0.01},
                seed=rng.randint(0, 99999), brightness=0.8)
    elif profile == "coded":
        # Coded-identity duel: primary blinks CODE_A, five equal decoys
        # blink the exact inverse CODE_B (same size/brightness/geometry
        # class as distractor). Only the temporal code separates them.
        from fsoc_tracker.tracking.identity import CODE_A, CODE_B
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.1, "vy": 0.02},
                   seed=rng.randint(0, 99999), is_primary=True, code=CODE_A)
        cfg.primary_beacon_id = 0
        for i in range(1, 6):
            add_beacon(
                cfg, 1000.0 + (i - 3) * 1.5, 1000.0 + (i % 3 - 1) * 1.0,
                500.0 + i * 20, motion="straight_line",
                motion_params={"vx": 0.05 * (i + 1), "vy": 0.01},
                seed=rng.randint(0, 99999), brightness=1.0, code=CODE_B)
    elif profile == "moving":
        # Two moving endpoints: Terminal A drifts laterally and yaws
        # slowly while the beacon flies its own straight line. Rates
        # stay inside the 5 deg/s gimbal authority with margin.
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.3, "vy": 0.2},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
        set_terminal_motion(cfg, "drift", vx=0.3, yaw_rate_deg_s=0.2)
    elif profile == "loss":
        # Fast beacon plus harness-applied temporal disappearance windows
        # (the runner enables them for this profile; see benchmark runner).
        add_beacon(cfg, 998.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 998.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 3.0, "vy": 0.0, "vz": 0.0},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
        set_disturbance(cfg, preset="clear")
    elif profile in ("noise", "fog", "jitter"):
        add_beacon(cfg, 1000.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 1000.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 0.3, "vy": 0.2},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
        set_disturbance(cfg, preset=profile)
    elif profile == "fast":
        # Fast lateral crosser for lag/lead-angle evaluation: enters near
        # the FOV edge at ~240 px/s, inside the 5 deg/s mount capability.
        add_beacon(cfg, 985.0, 1000.0, 500.0, motion="straight_line",
                   motion_params={"x0": 985.0, "y0": 1000.0, "z0": 500.0,
                                  "vx": 10.0, "vy": 0.0, "vz": 0.0},
                   seed=rng.randint(0, 99999), is_primary=True)
        cfg.primary_beacon_id = 0
        set_disturbance(cfg, preset="clear")
    return cfg


def make_experiment_world(
    motion: str = "straight_line",
    seed: int = 42,
    target_size_px: float = 10.0,
    disturbance: str = "clear",
    name: str | None = None,
    width: float = 2000.0,
    height: float = 2000.0,
    depth: float = 2000.0,
) -> ScenarioConfig:
    """Build a reproducible single-beacon experiment from motion + seed.

    Replaces numbered SIH scene tables. Placement is explicit; primary
    is designated by the caller via ``set_primary_beacon``.
    """
    cfg = create_world(
        width=width, height=height, depth=depth, seed=seed,
        name=name or f"experiment/{motion}/{seed}",
    )
    params = _default_motion_params(motion, 1000.0, 1000.0, 100.0, __import__("random").Random(seed))
    beacon = add_beacon(
        cfg, 1000.0, 1000.0, 100.0,
        motion=motion, motion_params=params,
        seed=seed, size_px=target_size_px, is_primary=True,
    )
    set_primary_beacon(cfg, beacon.beacon_id)
    if disturbance and disturbance != "clear":
        set_disturbance(cfg, preset=disturbance)
    return cfg


def benchmark_profiles() -> tuple[str, ...]:
    """Available benchmark world profiles."""
    return _BENCHMARK_PROFILES
