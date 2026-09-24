"""Scenario configuration for the simulation test environment.

Defines the configurable test environment:
  - Terminal A position and initial orientation
  - Any number of independent beacons, each with its own trajectory, seed, size, brightness
  - Primary beacon selection (mission target)
  - Disturbance settings

The scenario is the user-facing configuration layer.  The simulation engine
consumes it to set up the world.  Ground truth is NOT exposed through this
layer — it exists only in the engine internals for offline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TerminalConfig:
    """Configuration for Terminal A (the tracking camera).

    Attributes:
        x, y, z: World position of the terminal.
        yaw_deg: Initial yaw (pan) in degrees. 0 = facing +Z.
        pitch_deg: Initial tilt in degrees. 0 = level.
        hfov_deg: Horizontal field of view.
        vfov_deg: Vertical field of view.
        max_pan_speed_deg_s: Maximum pan rotation speed.
        max_tilt_speed_deg_s: Maximum tilt rotation speed.
    """

    x: float = 1000.0
    y: float = 1000.0
    z: float = 50.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    hfov_deg: float = 4.0
    vfov_deg: float = 3.0
    max_pan_speed_deg_s: float = 5.0
    max_tilt_speed_deg_s: float = 5.0
    # Platform motion (Terminal A itself moves, e.g. two satellites):
    # "static" (default, today's behavior) or "drift" (constant
    # velocity + constant attitude rates, stepped by the engine).
    # platform_motion_params keys: vx, vy, vz (world units/s),
    # yaw_rate_deg_s, pitch_rate_deg_s, roll_rate_deg_s.
    platform_motion: str = "static"
    platform_motion_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeaconConfig:
    """Configuration for a single beacon target.

    Each beacon has its own independent trajectory, seed, and visual
    properties.  Changing one beacon's configuration never affects others.

    Attributes:
        beacon_id: Unique identifier. Auto-assigned if -1.
        x0, y0, z0: Initial world position.
        trajectory: Trajectory type name (e.g. "straight_line", "circular").
        trajectory_params: Parameters passed to the trajectory constructor.
        seed: Random seed for this beacon's trajectory. Ensures independence.
        size_px: Visual size in pixels (rendered diameter).
        brightness: Brightness multiplier (0.0-1.0).
        is_primary: Whether this is the mission target beacon.
        active: Whether this beacon is active in the simulation.
    """

    beacon_id: int = -1
    x0: float = 1000.0
    y0: float = 1000.0
    z0: float = 500.0
    trajectory: str = "straight_line"
    trajectory_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    size_px: float = 10.0
    brightness: float = 1.0
    # Beacon spot shape: "square" (PS default), "circular", or "spot"
    # (soft Gaussian). Rendered per target by the sensor renderer.
    shape: str = "square"
    # Temporal identity code (binary string, e.g. "10110010"): brightness
    # is multiplied by the code bit each frame. "" = steady ON (default,
    # today's behavior). Distinct codes let the tracker reject decoys.
    code: str = ""
    is_primary: bool = False
    active: bool = True


@dataclass
class DisturbanceConfig:
    """Atmospheric disturbance configuration.

    Attributes:
        enabled: Whether disturbances are active.
        preset: Preset name ("clear", "fog", "rain", etc.).
        jitter_px: Platform jitter amplitude in pixels.
        noise_sigma: Sensor noise standard deviation.
    """

    enabled: bool = False
    preset: str = "clear"
    jitter_px: float = 0.0
    noise_sigma: float = 0.0


@dataclass
class ScenarioConfig:
    """Complete scenario configuration for the simulation test environment.

    This is the user-facing configuration object.  It describes:
      - The world
      - Terminal A (position, orientation, camera)
      - All beacons (each independent)
      - Which beacon is the primary mission target
      - Disturbances

    The engine consumes this to set up the simulation.  Ground truth is
    NOT part of this config — it exists only in the engine internals.

    Usage::

        scenario = ScenarioConfig(
            terminal=TerminalConfig(yaw_deg=10.0),  # start misaligned
            beacons=[
                BeaconConfig(x0=500, y0=500, z0=800, trajectory="circular",
                             seed=1, is_primary=True),
                BeaconConfig(x0=1500, y0=1200, z0=300, trajectory="random",
                             seed=2, is_primary=False),
            ],
        )
    """

    name: str = "Untitled Scenario"
    description: str = ""
    terminal: TerminalConfig = field(default_factory=TerminalConfig)
    beacons: list[BeaconConfig] = field(default_factory=list)
    primary_beacon_id: int | None = None
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    world_width: float = 2000.0
    world_height: float = 2000.0
    world_depth: float = 2000.0
    seed: int = 42
    next_beacon_id: int = 0

    def __post_init__(self) -> None:
        # Auto-assign beacon IDs if not set
        for i, b in enumerate(self.beacons):
            if b.beacon_id < 0:
                b.beacon_id = i
        # Monotonic counter never reuses IDs (tracking identity safety).
        used = [b.beacon_id for b in self.beacons]
        self.next_beacon_id = max(max(used, default=-1) + 1,
                                  self.next_beacon_id)

        # Auto-select primary beacon if not set
        if self.primary_beacon_id is None:
            for b in self.beacons:
                if b.is_primary:
                    self.primary_beacon_id = b.beacon_id
                    break

        # Mark the primary beacon
        if self.primary_beacon_id is not None:
            for b in self.beacons:
                if b.beacon_id == self.primary_beacon_id:
                    b.is_primary = True

    @property
    def primary_beacon(self) -> BeaconConfig | None:
        """Return the primary beacon config, or None."""
        if self.primary_beacon_id is None:
            return None
        for b in self.beacons:
            if b.beacon_id == self.primary_beacon_id:
                return b
        return None

    def get_beacon(self, beacon_id: int) -> BeaconConfig | None:
        """Return a beacon config by ID."""
        for b in self.beacons:
            if b.beacon_id == beacon_id:
                return b
        return None

    def add_beacon(self, **kwargs: Any) -> BeaconConfig:
        """Add a beacon and return its config."""
        bid = kwargs.pop("beacon_id", self.next_beacon_id)
        beacon = BeaconConfig(beacon_id=bid, **kwargs)
        self.beacons.append(beacon)
        self.next_beacon_id = max(self.next_beacon_id, bid + 1)
        if beacon.is_primary and self.primary_beacon_id is None:
            self.primary_beacon_id = beacon.beacon_id
        return beacon
