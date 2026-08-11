"""Actuator interface — real OS input is Phase F."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ActionResult, ActionStep, UIVisionResult, VerificationResult


class ActuatorBackend(ABC):
    @abstractmethod
    def execute(
        self,
        step: ActionStep,
        vision: UIVisionResult,
        *,
        trace_id: str = "",
    ) -> ActionResult: ...

    @abstractmethod
    def verify(
        self,
        step: ActionStep,
        action: ActionResult,
        *,
        trace_id: str = "",
    ) -> VerificationResult: ...
