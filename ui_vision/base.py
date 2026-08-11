"""UI vision backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ScreenFrame, UIVisionResult


class UIVisionBackend(ABC):
    @abstractmethod
    def recognize(self, frame: ScreenFrame, *, trace_id: str = "") -> UIVisionResult:
        """Return structured UIElement list for a frame."""
