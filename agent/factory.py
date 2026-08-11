"""Agent / TaskAgent factory."""

from __future__ import annotations

from collections.abc import Callable

from core.config import AppConfig
from core.models import ActionPreview, ActionStep, SafetyDecision

from agent.base import AgentBackend
from agent.planner import create_planner
from agent.runtime import ConfirmCallback, TaskAgent


def create_agent_backend(config: AppConfig, inference: object | None = None) -> AgentBackend:
    return create_planner(config, inference=inference)


def create_task_agent(
    config: AppConfig,
    *,
    confirm_callback: ConfirmCallback
    | Callable[[ActionPreview, ActionStep, SafetyDecision], bool]
    | None = None,
    auto_confirm: bool | None = None,
    dry_run: bool | None = None,
) -> TaskAgent:
    """Build a fully wired TaskAgent from config (+ optional overrides)."""
    if auto_confirm is not None:
        config.agent.auto_confirm = auto_confirm
    if dry_run is not None:
        config.actuator.dry_run = dry_run
    return TaskAgent(config, confirm_callback=confirm_callback)
