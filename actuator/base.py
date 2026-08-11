"""Actuator interface — structured OS input (Phase F)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import (
    ActionResult,
    ActionStep,
    UIElement,
    UIVisionResult,
    VerificationResult,
)


class ActuatorBackend(ABC):
    @abstractmethod
    def execute(
        self,
        step: ActionStep,
        vision: UIVisionResult,
        *,
        trace_id: str = "",
        coordinate_confirmed: bool = False,
        prior_element: UIElement | None = None,
    ) -> ActionResult: ...

    @abstractmethod
    def verify(
        self,
        step: ActionStep,
        action: ActionResult,
        *,
        trace_id: str = "",
        vision_before: UIVisionResult | None = None,
        vision_after: UIVisionResult | None = None,
        change_score: float | None = None,
        frame_id_before: str | None = None,
        frame_id_after: str | None = None,
    ) -> VerificationResult: ...
