"""Kalman filter for 2D constant-velocity tracking.

State vector:
    x = [px, py, vx, vy]^T

Transition model (constant velocity):
    px(k+1) = px(k) + vx(k) * dt
    py(k+1) = py(k) + vy(k) * dt
    vx(k+1) = vx(k)
    vy(k+1) = vy(k)

Measurement model:
    z = [px, py]^T

All dt values are derived from timestamps.  No fixed FPS is assumed.
"""

from __future__ import annotations

import numpy as np

from fsoc_tracker.tracking.config import TrackerConfig


class KalmanFilter2D:
    """2D constant-velocity Kalman filter.

    Pure NumPy implementation.  No external dependencies.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._state_dim = 4   # [px, py, vx, vy]
        self._meas_dim = 2    # [px, py]

        # State vector: [px, py, vx, vy]
        self._x = np.zeros((self._state_dim, 1), dtype=np.float64)

        # State covariance
        self._P = np.eye(self._state_dim, dtype=np.float64)

        # Measurement matrix H (maps state to measurement)
        self._H = np.zeros((self._meas_dim, self._state_dim), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0

        # Measurement noise R
        self._R = np.diag([
            self._config.measurement_noise_x,
            self._config.measurement_noise_y,
        ]).astype(np.float64)

        # Innovation (for diagnostics)
        self._innovation = np.zeros((self._meas_dim, 1), dtype=np.float64)
        self._innovation_cov = np.eye(self._meas_dim, dtype=np.float64)
        self._S_inv = np.eye(self._meas_dim, dtype=np.float64)

        self._initialized = False
        self._last_timestamp_s: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        """Current estimated position (px, py)."""
        return float(self._x[0, 0]), float(self._x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/s."""
        return float(self._x[2, 0]), float(self._x[3, 0])

    @property
    def state_vector(self) -> np.ndarray:
        """Full state vector [px, py, vx, vy]^T."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """State covariance matrix P."""
        return self._P.copy()

    @property
    def position_uncertainty(self) -> tuple[float, float]:
        """Position standard deviations (sigma_x, sigma_y)."""
        return float(np.sqrt(self._P[0, 0])), float(np.sqrt(self._P[1, 1]))

    @property
    def innovation(self) -> tuple[float, float]:
        """Last innovation (measurement residual) (dx, dy)."""
        return float(self._innovation[0, 0]), float(self._innovation[1, 0])

    @property
    def innovation_mahal(self) -> float:
        """Mahalanobis distance of last innovation."""
        innov = self._innovation
        return float(innov.T @ self._S_inv @ innov)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, position: tuple[float, float], timestamp_s: float) -> None:
        """Initialize filter state with first measurement.

        Sets position to measurement, velocity to zero.
        """
        self._x[0, 0] = position[0]
        self._x[1, 0] = position[1]
        self._x[2, 0] = 0.0
        self._x[3, 0] = 0.0

        self._P = np.diag([
            self._config.initial_position_uncertainty,
            self._config.initial_position_uncertainty,
            self._config.initial_velocity_uncertainty,
            self._config.initial_velocity_uncertainty,
        ]).astype(np.float64)

        self._last_timestamp_s = timestamp_s
        self._initialized = True

    def predict(self, dt: float) -> tuple[float, float]:
        """Predict state forward by dt seconds.

        Args:
            dt: Time step in seconds.  Must be positive.

        Returns:
            Predicted position (px, py).

        Raises:
            ValueError: If dt is not positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not self._initialized:
            raise RuntimeError("Filter not initialized")

        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Process noise Q
        q_cfg = self._config
        dt2 = dt * dt / 2.0
        Q = np.array([
            [dt2*dt2 * q_cfg.process_noise_pos, 0,                  dt2 * q_cfg.process_noise_pos, 0],
            [0,                  dt2*dt2 * q_cfg.process_noise_pos,  0,                  dt2 * q_cfg.process_noise_pos],
            [dt2 * q_cfg.process_noise_pos, 0,                dt * q_cfg.process_noise_vel,  0],
            [0,                  dt2 * q_cfg.process_noise_pos,  0,                  dt * q_cfg.process_noise_vel],
        ], dtype=np.float64)

        # Predict
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)

        return float(self._x[0, 0]), float(self._x[1, 0])

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """Update state with a measurement.

        Args:
            measurement: Measured position (px, py).

        Returns:
            Updated position (px, py).
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized")

        z = np.array(measurement, dtype=np.float64).reshape(self._meas_dim, 1)

        # Innovation
        y = z - self._H @ self._x
        self._innovation = y

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R
        self._innovation_cov = S

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
            self._S_inv = S_inv
            K = self._P @ self._H.T @ S_inv
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            S_inv = np.linalg.pinv(S)
            self._S_inv = S_inv
            K = self._P @ self._H.T @ S_inv

        # Update state
        self._x = self._x + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self._state_dim, dtype=np.float64) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T
        # Enforce symmetry (roundoff drifts P asymmetric over thousands
        # of updates; asymmetric P corrupts Mahalanobis gating).
        self._P = 0.5 * (self._P + self._P.T)

        return float(self._x[0, 0]), float(self._x[1, 0])

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._x = np.zeros((self._state_dim, 1), dtype=np.float64)
        self._P = np.eye(self._state_dim, dtype=np.float64)
        self._innovation = np.zeros((self._meas_dim, 1), dtype=np.float64)
        self._innovation_cov = np.eye(self._meas_dim, dtype=np.float64)
        self._S_inv = np.eye(self._meas_dim, dtype=np.float64)
        self._initialized = False
        self._last_timestamp_s = 0.0

    def mahalanobis_distance(self, measurement: tuple[float, float]) -> float:
        """Compute Mahalanobis distance of a measurement from current state.

        Does NOT modify filter state.
        """
        z = np.array(measurement, dtype=np.float64).reshape(self._meas_dim, 1)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        return float((y.T @ S_inv @ y).item())
