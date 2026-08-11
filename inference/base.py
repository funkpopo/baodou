"""Inference backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import InferenceResponse, ScreenFrame, UIVisionResult


class InferenceBackend(ABC):
    @abstractmethod
    def health(self) -> bool:
        """Return True if backend is ready."""

    @abstractmethod
    def observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
    ) -> InferenceResponse:
        """Produce ScreenObservation (and optionally a plan sketch)."""

    def close(self) -> None:
        return None
