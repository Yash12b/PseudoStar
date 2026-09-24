"""Tests for trajectory implementations."""

from __future__ import annotations

import math

import pytest

from fsoc_tracker.simulation.trajectory.circular import CircularTrajectory
from fsoc_tracker.simulation.trajectory.figure_eight import FigureEightTrajectory
from fsoc_tracker.simulation.trajectory.random import RandomTrajectory
from fsoc_tracker.simulation.trajectory.sinusoidal import SinusoidalTrajectory
from fsoc_tracker.simulation.trajectory.spiral import SpiralTrajectory
from fsoc_tracker.simulation.trajectory.straight_line import StraightLineTrajectory


class TestStraightLine:
    def test_position_at_zero(self) -> None:
        t = StraightLineTrajectory(x0=100, y0=200, z0=300, vx=10, vy=20, vz=30)
        assert t.position(0) == (100, 200, 300)

    def test_position_linear(self) -> None:
        t = StraightLineTrajectory(x0=0, y0=0, z0=0, vx=10, vy=0, vz=0)
        px, py, pz = t.position(5.0)
        assert px == pytest.approx(50.0)
        assert py == pytest.approx(0.0)
        assert pz == pytest.approx(0.0)

    def test_velocity_constant(self) -> None:
        t = StraightLineTrajectory(vx=10, vy=20, vz=30)
        for time in [0.0, 1.0, 5.0, 100.0]:
            assert t.velocity(time) == (10, 20, 30)

    def test_acceleration_zero(self) -> None:
        t = StraightLineTrajectory()
        for time in [0.0, 1.0, 5.0]:
            assert t.acceleration(time) == (0.0, 0.0, 0.0)

    def test_speed(self) -> None:
        t = StraightLineTrajectory(vx=3, vy=4, vz=0)
        assert t.speed() == pytest.approx(5.0)

    def test_duration(self) -> None:
        t = StraightLineTrajectory(duration=10.0)
        assert t.duration == 10.0
        assert not t.is_finished(5.0)
        assert t.is_finished(10.0)
        assert t.is_finished(15.0)

    def test_infinite_duration(self) -> None:
        t = StraightLineTrajectory()
        assert t.duration is None
        assert not t.is_finished(100000.0)

    def test_serialization_roundtrip(self) -> None:
        t = StraightLineTrajectory(x0=10, y0=20, vx=5, vy=10)
        d = t.to_dict()
        t2 = StraightLineTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)


class TestCircular:
    def test_position_at_zero(self) -> None:
        t = CircularTrajectory(cx=100, cy=200, cz=50, radius=30, phase_rad=0)
        px, py, pz = t.position(0)
        assert px == pytest.approx(130.0)  # cx + r*cos(0)
        assert py == pytest.approx(200.0)  # cy + r*sin(0)
        assert pz == pytest.approx(50.0)

    def test_radius_preserved(self) -> None:
        t = CircularTrajectory(cx=1000, cy=1000, radius=200)
        for time in [0.0, 0.5, 1.0, 2.5, 10.0]:
            px, py, _ = t.position(time)
            r = math.sqrt((px - 1000) ** 2 + (py - 1000) ** 2)
            assert r == pytest.approx(200.0, rel=1e-10)

    def test_velocity_orthogonal_to_position(self) -> None:
        t = CircularTrajectory(cx=0, cy=0, radius=100, angular_speed_rad_s=1.0)
        for time in [0.0, 0.5, 1.0]:
            px, py, _ = t.position(time)
            vx, vy, _ = t.velocity(time)
            dot = px * vx + py * vy
            assert dot == pytest.approx(0.0, abs=1e-8)

    def test_acceleration_centripetal(self) -> None:
        t = CircularTrajectory(cx=0, cy=0, radius=100, angular_speed_rad_s=2.0)
        for time in [0.0, 0.5, 1.0]:
            ax, ay, _ = t.acceleration(time)
            a_mag = math.sqrt(ax**2 + ay**2)
            expected = 100 * 4.0  # r * omega^2
            assert a_mag == pytest.approx(expected, rel=1e-8)

    def test_full_period(self) -> None:
        omega = math.pi / 2.0  # period = 4s
        t = CircularTrajectory(cx=500, cy=500, radius=100, angular_speed_rad_s=omega)
        p0 = t.position(0)
        p4 = t.position(4.0)
        for a, b in zip(p0, p4):
            assert a == pytest.approx(b, rel=1e-8)

    def test_xz_plane(self) -> None:
        t = CircularTrajectory(cx=0, cy=0, cz=50, radius=100, plane="xz")
        px, py, pz = t.position(0)
        assert px == pytest.approx(100.0)
        assert py == pytest.approx(0.0)
        assert pz == pytest.approx(50.0)

    def test_serialization_roundtrip(self) -> None:
        t = CircularTrajectory(cx=500, cy=500, radius=150)
        d = t.to_dict()
        t2 = CircularTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)


class TestFigureEight:
    def test_position_at_zero(self) -> None:
        t = FigureEightTrajectory(cx=1000, cy=1000, amplitude_x=100, amplitude_y=80, phase_x_rad=0, phase_y_rad=0)
        px, py, _ = t.position(0)
        assert px == pytest.approx(1000.0)
        assert py == pytest.approx(1000.0)

    def test_symmetry(self) -> None:
        t = FigureEightTrajectory(cx=0, cy=0, amplitude_x=100, amplitude_y=80)
        # At t and t + half_period, the figure-8 should have symmetry
        p1 = t.position(0)
        p2 = t.position(math.pi / t._omega)
        # x should be same (sin(omega*t) vs sin(omega*t + pi) = -sin)
        # But at t=0 both are 0, so check a non-zero time
        p3 = t.position(1.0)
        p4 = t.position(1.0 + math.pi / t._omega)
        # x should be negated
        assert p3[0] == pytest.approx(-p4[0], abs=1e-8)

    def test_bounds_respected(self) -> None:
        t = FigureEightTrajectory(cx=1000, cy=1000, amplitude_x=100, amplitude_y=80)
        for i in range(100):
            px, py, _ = t.position(i * 0.1)
            assert 900 <= px <= 1100
            assert 920 <= py <= 1080

    def test_serialization_roundtrip(self) -> None:
        t = FigureEightTrajectory(cx=500, cy=500, amplitude_x=100, amplitude_y=60)
        d = t.to_dict()
        t2 = FigureEightTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)


class TestRandom:
    def test_deterministic(self) -> None:
        t1 = RandomTrajectory(seed=42)
        t2 = RandomTrajectory(seed=42)
        for i in range(50):
            assert t1.position(i * 0.5) == t2.position(i * 0.5)

    def test_different_seeds_differ(self) -> None:
        t1 = RandomTrajectory(seed=42)
        t2 = RandomTrajectory(seed=99)
        # At some point they should diverge
        diverged = False
        for i in range(1, 50):
            p1 = t1.position(i * 0.5)
            p2 = t2.position(i * 0.5)
            if abs(p1[0] - p2[0]) > 0.1:
                diverged = True
                break
        assert diverged

    def test_bounds_respected(self) -> None:
        t = RandomTrajectory(x_min=0, x_max=2000, y_min=0, y_max=2000, seed=42)
        for i in range(100):
            px, py, _ = t.position(i * 0.5)
            assert -10 <= px <= 2010  # small margin for interpolation
            assert -10 <= py <= 2010

    def test_reset(self) -> None:
        t = RandomTrajectory(seed=42)
        p1 = t.position(2.0)
        t.reset()
        p2 = t.position(2.0)
        assert p1 == p2

    def test_serialization_roundtrip(self) -> None:
        t = RandomTrajectory(seed=42, num_waypoints=10)
        d = t.to_dict()
        t2 = RandomTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)


class TestSpiral:
    def test_initial_radius(self) -> None:
        t = SpiralTrajectory(cx=500, cy=500, radius_start=50, radius_growth=0)
        px, py, _ = t.position(0)
        r = math.sqrt((px - 500) ** 2 + (py - 500) ** 2)
        assert r == pytest.approx(50.0)

    def test_expanding(self) -> None:
        t = SpiralTrajectory(cx=500, cy=500, radius_start=50, radius_growth=10)
        r0 = math.sqrt((t.position(0)[0] - 500) ** 2 + (t.position(0)[1] - 500) ** 2)
        r5 = math.sqrt((t.position(5)[0] - 500) ** 2 + (t.position(5)[1] - 500) ** 2)
        assert r5 > r0

    def test_duration(self) -> None:
        t = SpiralTrajectory(duration=10.0)
        assert t.duration == 10.0
        assert t.is_finished(11.0)

    def test_serialization_roundtrip(self) -> None:
        t = SpiralTrajectory(radius_start=100, radius_growth=5)
        d = t.to_dict()
        t2 = SpiralTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)


class TestSinusoidal:
    def test_position_at_zero(self) -> None:
        t = SinusoidalTrajectory(cx=1000, cy=1000, amplitude_x=100, amplitude_y=80)
        px, py, _ = t.position(0)
        assert px == pytest.approx(1000.0)
        assert py == pytest.approx(1000.0)

    def test_amplitude_bounds(self) -> None:
        t = SinusoidalTrajectory(cx=1000, cy=1000, amplitude_x=100, amplitude_y=80)
        for i in range(200):
            px, py, _ = t.position(i * 0.1)
            assert 900 <= px <= 1100
            assert 920 <= py <= 1080

    def test_serialization_roundtrip(self) -> None:
        t = SinusoidalTrajectory(amplitude_x=50, amplitude_y=40)
        d = t.to_dict()
        t2 = SinusoidalTrajectory.from_dict(d)
        assert t.position(1.0) == t2.position(1.0)
