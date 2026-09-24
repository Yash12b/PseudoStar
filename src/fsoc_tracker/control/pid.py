"""PID controller with anti-windup, derivative filtering, and deadband.

Reusable, FPS-independent PID controller for single-axis control.
"""

from __future__ import annotations

import math


class PIDController:
    """Single-axis PID controller with anti-windup and derivative filtering.

    u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * d_filtered(e)/dt

    Usage::

        pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
        output = pid.update(error, dt)
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.1,
        output_limit: float = 5.0,
        integral_limit: float = 10.0,
        derivative_filter_alpha: float = 0.5,
        deadband: float = 0.0,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._output_limit = output_limit
        self._integral_limit = integral_limit
        self._derivative_filter_alpha = derivative_filter_alpha
        self._deadband = deadband

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_filtered_derivative: float = 0.0
        self._initialized: bool = False
        self._saturated: bool = False
        self._in_deadband: bool = False

        # Diagnostics
        self._p_term: float = 0.0
        self._i_term: float = 0.0
        self._d_term: float = 0.0
        self._raw_derivative: float = 0.0
        self._filtered_derivative: float = 0.0
        self._output_before_sat: float = 0.0

    @property
    def kp(self) -> float:
        return self._kp

    @kp.setter
    def kp(self, value: float) -> None:
        self._kp = value

    @property
    def ki(self) -> float:
        return self._ki

    @ki.setter
    def ki(self, value: float) -> None:
        self._ki = value

    @property
    def kd(self) -> float:
        return self._kd

    @kd.setter
    def kd(self, value: float) -> None:
        self._kd = value

    @property
    def p_term(self) -> float:
        return self._p_term

    @property
    def i_term(self) -> float:
        return self._i_term

    @property
    def d_term(self) -> float:
        return self._d_term

    @property
    def raw_derivative(self) -> float:
        return self._raw_derivative

    @property
    def filtered_derivative(self) -> float:
        return self._filtered_derivative

    @property
    def output_before_saturation(self) -> float:
        return self._output_before_sat

    @property
    def saturated(self) -> bool:
        return self._saturated

    @property
    def in_deadband(self) -> bool:
        return self._in_deadband

    @property
    def integral(self) -> float:
        return self._integral

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd

    def set_output_limit(self, limit: float) -> None:
        self._output_limit = limit

    def set_integral_limit(self, limit: float) -> None:
        self._integral_limit = limit

    def set_deadband(self, deadband: float) -> None:
        self._deadband = deadband

    def update(self, error: float, dt: float) -> float:
        """Update PID with current error and dt.

        Args:
            error: Current error (desired - actual).
            dt: Time step in seconds. Must be positive.

        Returns:
            Control output, clamped to [-output_limit, output_limit].
        """
        if dt <= 0:
            return 0.0

        if math.isnan(error) or math.isinf(error):
            return 0.0

        # Deadband: hold zero output AND clear the integrator so no
        # stale integral resumes as a kick on exit (anti-hunt).
        if abs(error) < self._deadband:
            self._in_deadband = True
            self._integral = 0.0
            self._p_term = 0.0
            self._i_term = 0.0
            self._d_term = 0.0
            self._raw_derivative = 0.0
            self._filtered_derivative = 0.0
            self._output_before_sat = 0.0
            return 0.0
        self._in_deadband = False

        # P term
        self._p_term = self._kp * error

        # I term with anti-windup
        self._integral += error * dt
        self._integral = max(-self._integral_limit, min(self._integral_limit, self._integral))
        self._i_term = self._ki * self._integral

        # D term with derivative filtering
        if self._initialized:
            self._raw_derivative = (error - self._prev_error) / dt
            alpha = self._derivative_filter_alpha
            self._filtered_derivative = (
                alpha * self._prev_filtered_derivative
                + (1.0 - alpha) * self._raw_derivative
            )
        else:
            self._raw_derivative = 0.0
            self._filtered_derivative = 0.0
            self._initialized = True

        self._d_term = self._kd * self._filtered_derivative

        # Output
        output = self._p_term + self._i_term + self._d_term
        self._output_before_sat = output

        # Saturate
        if output > self._output_limit:
            output = self._output_limit
            self._saturated = True
            # Anti-windup: prevent integral from growing in saturated direction
            if error > 0:
                self._integral -= error * dt
                self._i_term = self._ki * self._integral
        elif output < -self._output_limit:
            output = -self._output_limit
            self._saturated = True
            if error < 0:
                self._integral -= error * dt
                self._i_term = self._ki * self._integral
        else:
            self._saturated = False

        self._prev_error = error
        self._prev_filtered_derivative = self._filtered_derivative

        return output

    def reset(self) -> None:
        """Reset PID state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_filtered_derivative = 0.0
        self._initialized = False
        self._saturated = False
        self._in_deadband = False
        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0
        self._raw_derivative = 0.0
        self._filtered_derivative = 0.0
        self._output_before_sat = 0.0
