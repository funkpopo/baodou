"""Agent backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ActionPlan, ScreenObservation, UIVisionResult


class AgentBackend(ABC):
    @abstractmethod
    def plan(
        self,
        user_goal: str,
        vision: UIVisionResult,
        observation: ScreenObservation,
        *,
        trace_id: str = "",
    ) -> ActionPlan:
        """Build a structured ActionPlan from goal + UI context."""
