"""Control subsystem — closed-loop coarse pointing control.

Modules:
    config    — ControllerConfig and ControlMode
    pid       — PIDController (reusable, with anti-windup, derivative filter)
    command   — ControlCommand and ControlTelemetry
    controller — CoarsePointingController (main class) and CameraActuator
"""

from fsoc_tracker.control.command import ControlCommand, ControlTelemetry
from fsoc_tracker.control.config import ControllerConfig, ControlMode
from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.control.pid import PIDController

__all__ = [
    "ControllerConfig",
    "ControlMode",
    "ControlCommand",
    "ControlTelemetry",
    "PIDController",
    "CoarsePointingController",
    "CameraActuator",
]
