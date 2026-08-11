"""Action execution (Phase F: mock dry-run + optional Windows SendInput)."""

from actuator.base import ActuatorBackend
from actuator.factory import create_actuator
from actuator.mock import MockActuator

__all__ = ["ActuatorBackend", "MockActuator", "create_actuator"]
