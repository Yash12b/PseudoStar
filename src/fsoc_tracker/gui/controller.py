"""Application controller — clean facade between GUI and subsystems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ApplicationController:
    """Thin facade. GUI calls these methods; worker does the work."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = self._default_config()

    def _default_config(self) -> dict[str, Any]:
        return {
            "mode": "simulation",
            "camera_id": 0,
            "trajectory": "straight_line",
            "trajectory_params": {},
            "target_size": 10.0,
            "camera_width": 640,
            "camera_height": 480,
            "hfov": 4.0,
            "vfov": 3.0,
            "max_pan_rate": 5.0,
            "max_tilt_rate": 5.0,
            "world_width": 2000.0,
            "world_height": 2000.0,
            "seed": 42,
            "sim_dt": 1.0 / 30.0,
            "disturbance_enabled": False,
            "disturbance_preset": "clear",
            "perception_backend": "classical",
            "confidence_threshold": 0.3,
        }

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def set_mode(self, mode: str) -> None:
        self._config["mode"] = mode

    def set_trajectory(self, traj: str, params: dict[str, Any] | None = None) -> None:
        self._config["trajectory"] = traj
        if params is not None:
            self._config["trajectory_params"] = params

    def set_camera(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._config:
                self._config[k] = v

    def set_disturbance(self, enabled: bool, preset: str = "clear") -> None:
        self._config["disturbance_enabled"] = enabled
        self._config["disturbance_preset"] = preset

    def set_perception(self, backend: str, threshold: float = 0.3) -> None:
        self._config["perception_backend"] = backend
        self._config["confidence_threshold"] = threshold

    def save_config(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._config, f, indent=2)

    def load_config(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(p) as f:
            self._config = json.load(f)

    def get_worker_config(self) -> dict[str, Any]:
        return dict(self._config)
