"""Developer diagnostic bundle: frames, UI tree, OCR notes, prompt, plan, verify (Phase H)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.models import (
    ActionPlan,
    ActionPreview,
    ActionResult,
    ScreenFrame,
    ScreenObservation,
    TaskContext,
    UIVisionResult,
    UserCorrection,
    VerificationResult,
)

from frontend.corrections import CorrectionStore
from frontend.metrics import MetricsCollector


@dataclass
class DiagnosticBundle:
    """Snapshot of internal state for the developer view (no raw secrets)."""

    task: dict[str, Any] = field(default_factory=dict)
    frame: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    elements: list[dict[str, Any]] = field(default_factory=list)
    observation: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    previews: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    activity: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    prompt_version: str = ""
    model_version: str = ""
    raw_notes: str = ""
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "frame": self.frame,
            "vision": self.vision,
            "elements": self.elements,
            "observation": self.observation,
            "plan": self.plan,
            "previews": self.previews,
            "actions": self.actions,
            "verifications": self.verifications,
            "corrections": self.corrections,
            "metrics": self.metrics,
            "activity": self.activity,
            "events": self.events,
            "prompt_version": self.prompt_version,
            "model_version": self.model_version,
            "raw_notes": self.raw_notes,
            "error": self.error,
        }


def build_diagnostics(
    *,
    task: TaskContext | None = None,
    frame: ScreenFrame | None = None,
    vision: UIVisionResult | None = None,
    observation: ScreenObservation | None = None,
    plan: ActionPlan | None = None,
    previews: list[ActionPreview] | None = None,
    actions: list[ActionResult] | None = None,
    verifications: list[VerificationResult] | None = None,
    corrections: CorrectionStore | list[UserCorrection] | None = None,
    metrics: MetricsCollector | None = None,
    activity: dict[str, Any] | None = None,
    events: list[Any] | None = None,
    error: Any = None,
    prompt_version: str = "",
    model_version: str = "",
    max_elements: int = 48,
) -> DiagnosticBundle:
    """Assemble a redacted diagnostic view from session state."""
    corr_list: list[UserCorrection]
    if corrections is None:
        corr_list = []
    elif isinstance(corrections, CorrectionStore):
        corr_list = corrections.to_list()
    else:
        corr_list = list(corrections)

    elements: list[dict[str, Any]] = []
    if vision is not None:
        for el in vision.elements[:max_elements]:
            elements.append(
                {
                    "element_id": el.element_id,
                    "type": el.type.value,
                    "text": (el.text or "")[:60],
                    "bbox": el.bbox.model_dump(),
                    "confidence": el.confidence,
                    "clickable": el.clickable,
                    "editable": el.editable,
                    "source": el.source,
                    "needs_review": el.needs_review,
                }
            )

    evt_out: list[dict[str, Any]] = []
    for e in events or []:
        if hasattr(e, "model_dump"):
            d = e.model_dump(mode="json")
            # Drop large payloads
            payload = d.get("payload") or {}
            if isinstance(payload, dict) and "image_b64" in payload:
                payload = {**payload, "image_b64": "<omitted>"}
                d["payload"] = payload
            evt_out.append(d)
        elif isinstance(e, dict):
            evt_out.append(e)

    err_dict = None
    if error is not None:
        err_dict = error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)[:200]}

    notes_parts: list[str] = []
    if vision and vision.notes:
        notes_parts.append(f"vision: {vision.notes[:200]}")
    if observation and observation.notes:
        notes_parts.append(f"obs: {observation.notes[:200]}")
    if observation and observation.observation:
        notes_parts.append(f"screen: {observation.observation[:240]}")

    return DiagnosticBundle(
        task=task.log_summary() if task else {},
        frame=frame.log_summary() if frame else None,
        vision=vision.log_summary() if vision else None,
        elements=elements,
        observation=observation.log_summary() if observation else None,
        plan=plan.log_summary() if plan else None,
        previews=[p.log_summary() for p in (previews or [])],
        actions=[a.log_summary() for a in (actions or [])],
        verifications=[v.log_summary() for v in (verifications or [])],
        corrections=[c.log_summary() for c in corr_list],
        metrics=metrics.snapshot().log_summary() if metrics else {},
        activity=activity or {},
        events=evt_out[-40:],
        prompt_version=prompt_version,
        model_version=model_version,
        raw_notes=" | ".join(notes_parts),
        error=err_dict,
    )
