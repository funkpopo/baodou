"""Actuator backend factory."""

from __future__ import annotations

from core.config import AppConfig
from core.errors import ConfigError

from actuator.base import ActuatorBackend


def create_actuator(config: AppConfig) -> ActuatorBackend:
    backend = (config.actuator.backend or "mock").lower()
    if backend == "mock":
        from actuator.mock import MockActuator

        return MockActuator(config)
    if backend in ("win", "windows", "sendinput"):
        from actuator.win_input import WinActuator

        return WinActuator(config)
    raise ConfigError(f"未知 actuator.backend: {backend}", details={"backend": backend})
