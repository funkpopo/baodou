"""Timeout / busy / invalid-output degrade strategies (Phase E).

When the model is busy, times out, or returns illegal output, we never pass
raw text to the actuator. Options:
  - return last trusted observation (caller-supplied cache)
  - return a safe empty plan asking for user confirmation
  - mark response ok=False with structured error
"""

from __future__ import annotations

from core.models import (
    ActionPlan,
    InferenceResponse,
    ScreenFrame,
    ScreenObservation,
    UIVisionResult,
)


def degraded_response(
    *,
    request_id: str,
    trace_id: str,
    frame: ScreenFrame,
    vision: UIVisionResult,
    user_goal: str,
    reason: str,
    error_code: str,
    last_observation: ScreenObservation | None = None,
    latency_ms: float | None = None,
    raw_text: str = "",
) -> InferenceResponse:
    """Build a non-executable but structured fallback response."""
    if last_observation is not None:
        obs = last_observation.model_copy(
            update={
                "trace_id": trace_id,
                "frame_id": frame.frame_id,
                "notes": f"degraded: using last trusted observation ({reason})",
                "confidence": min(last_observation.confidence, 0.4),
            }
        )
    else:
        # Minimal trusted state from UI vision only — no model claims.
        clickable = [e for e in vision.elements if e.clickable][:6]
        obs = ScreenObservation(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            observation=(
                f"Model unavailable or invalid ({reason}). "
                f"UI layer reports {len(vision.elements)} elements, "
                f"{len(clickable)} clickable. Goal: {user_goal}"
            )[:2000],
            ui_elements=clickable or vision.elements[:6],
            notes=f"degraded:{reason}",
            confidence=0.2,
            model_name="degraded",
            latency_ms=latency_ms,
        )

    # Empty plan — force user confirm path; never auto-act.
    plan = ActionPlan(
        trace_id=trace_id,
        goal=user_goal,
        steps=[],
        stop_if=["model_degraded", "target_missing"],
    )
    return InferenceResponse(
        request_id=request_id,
        trace_id=trace_id,
        ok=False,
        observation=obs,
        plan=plan,
        raw_text=raw_text[:4000],
        latency_ms=latency_ms,
        error_code=error_code,
        error_message=reason,
    )
