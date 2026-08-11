"""Action execution (Phase B: mock / dry-run only)."""

from actuator.base import ActuatorBackend
from actuator.mock import MockActuator

__all__ = ["ActuatorBackend", "MockActuator"]
