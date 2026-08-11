"""Capture backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ScreenFrame


class CaptureBackend(ABC):
    @abstractmethod
    def capture(self, *, trace_id: str = "") -> ScreenFrame:
        """Capture one frame; must set frame_id and dimensions."""

    def close(self) -> None:
        return None
