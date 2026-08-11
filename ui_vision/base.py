"""UI vision backend + recognizer plugin interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import BBox, ScreenFrame, UIElement, UIVisionResult
from PIL import Image


class UIRecognizer(ABC):
    """One detection source (UIA, OCR, rules, future app adapters)."""

    name: str = "base"

    @abstractmethod
    def recognize(
        self,
        frame: ScreenFrame,
        image: Image.Image | None = None,
        *,
        roi: BBox | None = None,
        trace_id: str = "",
    ) -> list[UIElement]:
        """Return elements in **physical screen pixels** (may be empty on failure)."""

    def close(self) -> None:
        return None


class UIVisionBackend(ABC):
    """Full vision backend: one or more sources → fused UIVisionResult."""

    @abstractmethod
    def recognize(
        self,
        frame: ScreenFrame,
        *,
        trace_id: str = "",
        image: Image.Image | None = None,
        roi: BBox | None = None,
        goal: str | None = None,
    ) -> UIVisionResult:
        """Return structured UIElement list for a frame."""

    def close(self) -> None:
        return None


class RecognizerContext:
    """Optional shared context passed into pipeline (window hwnd, goal, …)."""

    def __init__(self, **kwargs: Any) -> None:
        self.data = dict(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
