"""Inference backend interface (Phase E)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from core.models import InferenceResponse, ScreenFrame, UIVisionResult


class InferenceBackend(ABC):
    @abstractmethod
    def health(self) -> bool:
        """Return True if backend is ready."""

    def ensure_ready(self) -> dict[str, Any]:
        """Load / warmup if needed. Default: health probe only."""
        return {"healthy": self.health()}

    @abstractmethod
    def observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
        mode: str = "observe_plan",
        include_image: bool = True,
    ) -> InferenceResponse:
        """Produce validated ScreenObservation and optional ActionPlan."""

    def chat(
        self,
        *,
        user_text: str,
        system: str | None = None,
        image_b64: str | None = None,
        max_tokens: int | None = None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        """Low-level text/image chat (raw). Optional for backends."""
        raise NotImplementedError

    def stream_observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
        mode: str = "observe_plan",
        include_image: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield stream chunks; final chunk has validated result or error.

        Only a complete, schema-valid JSON payload may set ``ready_for_action=True``.
        """
        # Default: non-streaming single shot wrapped as one final event.
        resp = self.observe(
            frame,
            vision,
            user_goal=user_goal,
            trace_id=trace_id,
            mode=mode,
            include_image=include_image,
        )
        yield {
            "type": "final",
            "ready_for_action": bool(resp.ok and resp.plan is not None),
            "response": resp,
            "raw_text": resp.raw_text,
        }

    def cancel_current(self) -> None:
        """Best-effort cancel in-flight request (HTTP has limited support)."""
        return None

    def close(self) -> None:
        return None
