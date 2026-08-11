"""Post-action verification helpers (Phase F)."""

from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import (
    ActionResult,
    ActionStep,
    ActionType,
    UIVisionResult,
    VerificationResult,
)

_log = get_logger("actuator.verify")


def verify_step(
    step: ActionStep,
    action: ActionResult,
    *,
    config: AppConfig,
    vision_before: UIVisionResult | None = None,
    vision_after: UIVisionResult | None = None,
    change_score: float | None = None,
    frame_id_before: str | None = None,
    frame_id_after: str | None = None,
    trace_id: str = "",
) -> VerificationResult:
    """Validate action outcome using success flag + optional screen/UI signals."""
    expected = step.expected_change or f"{step.action.value} success"
    if not action.success:
        ver = VerificationResult(
            trace_id=trace_id,
            step_id=step.step_id,
            passed=False,
            expected=expected,
            actual=action.message,
            message="动作未成功，验证失败",
            change_score=change_score,
            frame_id_before=frame_id_before,
            frame_id_after=frame_id_after,
        )
        log_event(_log, "actuator.verify", **ver.log_summary())
        return ver

    # Wait / reidentify / none: success is enough.
    if step.action in (ActionType.NONE, ActionType.WAIT, ActionType.REIDENTIFY, ActionType.MOVE):
        ver = VerificationResult(
            trace_id=trace_id,
            step_id=step.step_id,
            passed=True,
            expected=expected,
            actual=action.message,
            message="ok (no screen delta required)",
            change_score=change_score,
            frame_id_before=frame_id_before,
            frame_id_after=frame_id_after,
        )
        log_event(_log, "actuator.verify", **ver.log_summary())
        return ver

    target_present: bool | None = None
    notes: list[str] = []

    if vision_after is not None and step.target_element_id:
        el = vision_after.by_id(step.target_element_id)
        if el is None and action.resolved_element_id:
            el = vision_after.by_id(action.resolved_element_id)
        # Fuzzy: any element with same text from before
        if el is None and vision_before is not None:
            before_el = vision_before.by_id(step.target_element_id)
            if before_el is not None:
                for cand in vision_after.elements:
                    if cand.type == before_el.type and (cand.text or "") == (before_el.text or ""):
                        el = cand
                        break
        target_present = el is not None
        if el is not None:
            notes.append(f"target present id={el.element_id}")
        else:
            notes.append("target not in post-vision (may be ok if UI navigated away)")

    # Soft change signal
    changed_ok = True
    if change_score is not None and config.actuator.verify_require_change:
        if change_score < config.actuator.verify_change_threshold:
            # Dry-run often has zero screen change — don't hard-fail dry runs.
            if action.dry_run:
                notes.append(
                    f"change_score={change_score:.4f} below threshold but dry_run soft-pass"
                )
            else:
                changed_ok = False
                notes.append(f"change_score={change_score:.4f} below threshold")
        else:
            notes.append(f"change_score={change_score:.4f} ok")

    # Frame identity: after action we expect a new verify frame id when re-captured.
    if frame_id_before and frame_id_after and frame_id_before == frame_id_after:
        notes.append("same frame_id before/after (no re-capture)")

    passed = changed_ok and action.success
    message = "; ".join(notes) if notes else ("verify ok" if passed else "verify failed")

    ver = VerificationResult(
        trace_id=trace_id,
        step_id=step.step_id,
        passed=passed,
        expected=expected,
        actual=action.message,
        message=message,
        change_score=change_score,
        target_still_present=target_present,
        frame_id_before=frame_id_before,
        frame_id_after=frame_id_after,
    )
    log_event(_log, "actuator.verify", **ver.log_summary())
    return ver
