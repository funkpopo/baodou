"""Task agent runtime: observe → plan → confirm → execute → verify (Phase F).

State machine:
  idle → observing → planning → awaiting_confirmation → executing → verifying
       → completed | failed | paused | cancelled
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from actuator.base import ActuatorBackend
from actuator.factory import create_actuator
from capture.base import CaptureBackend
from capture.factory import create_capture
from core.cancel import CancellationToken, get_global_token
from core.config import AppConfig
from core.errors import ActuatorError, BaodouError, CancelledError, ErrorCode
from core.logging import get_logger, log_event, new_trace_id, trace_scope
from core.models import (
    ActionPlan,
    ActionPreview,
    ActionResult,
    ActionStep,
    ActionType,
    FrameKind,
    PipelineEvent,
    PipelineEventKind,
    RecoveryAction,
    RiskLevel,
    SafetyDecision,
    ScreenFrame,
    ScreenObservation,
    StepRecord,
    TaskContext,
    TaskState,
    UIElement,
    UIVisionResult,
    VerificationResult,
)
from inference.base import InferenceBackend
from safety.policy import SafetyPolicy
from ui_vision.base import UIVisionBackend
from ui_vision.factory import create_ui_vision

from agent.base import AgentBackend
from agent.planner import create_planner
from agent.preview import build_plan_previews, build_step_preview
from agent.recovery import decide_recovery
from agent.state import is_terminal, transition

_log = get_logger("agent.runtime")

ConfirmCallback = Callable[[ActionPreview, ActionStep, SafetyDecision], bool]


@dataclass
class TaskRunResult:
    """Full result of one agent.run() invocation."""

    trace_id: str
    task: TaskContext
    frame: ScreenFrame | None = None
    vision: UIVisionResult | None = None
    observation: ScreenObservation | None = None
    plan: ActionPlan | None = None
    previews: list[ActionPreview] = field(default_factory=list)
    safety_decisions: list[SafetyDecision] = field(default_factory=list)
    actions: list[ActionResult] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    events: list[PipelineEvent] = field(default_factory=list)
    error: BaodouError | None = None
    ok: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "task": self.task.log_summary(),
            "frame": self.frame.log_summary() if self.frame else None,
            "vision": self.vision.log_summary() if self.vision else None,
            "observation": self.observation.log_summary() if self.observation else None,
            "plan": self.plan.log_summary() if self.plan else None,
            "previews": [p.log_summary() for p in self.previews],
            "safety": [s.log_summary() for s in self.safety_decisions],
            "actions": [a.log_summary() for a in self.actions],
            "verifications": [v.log_summary() for v in self.verifications],
            "step_records": [r.model_dump(mode="json") for r in self.task.step_records],
            "error": self.error.to_dict() if self.error else None,
            "events": [e.model_dump(mode="json") for e in self.events],
        }


class TaskAgent:
    """Orchestrates capture → vision → inference → plan → safety → act → verify."""

    def __init__(
        self,
        config: AppConfig,
        *,
        token: CancellationToken | None = None,
        capture: CaptureBackend | None = None,
        vision: UIVisionBackend | None = None,
        inference: InferenceBackend | None = None,
        planner: AgentBackend | None = None,
        actuator: ActuatorBackend | None = None,
        safety: SafetyPolicy | None = None,
        confirm_callback: ConfirmCallback | None = None,
    ) -> None:
        self.config = config
        self.token = token or get_global_token()
        self.capture = capture or create_capture(config)
        self.vision = vision or create_ui_vision(config)
        self.inference = inference
        if self.inference is None:
            from inference.http_client import create_inference

            self.inference = create_inference(config)
        self.planner = planner or create_planner(config, inference=self.inference)
        self.actuator = actuator or create_actuator(config)
        self.safety = safety or SafetyPolicy(config)
        self.confirm_callback = confirm_callback
        self._owns_backends = capture is None  # close only if we created them loosely

    # ------------------------------------------------------------------ helpers
    def _emit(
        self,
        result: TaskRunResult,
        kind: PipelineEventKind,
        payload: dict[str, Any],
        *,
        error: BaodouError | None = None,
    ) -> None:
        evt = PipelineEvent(
            trace_id=result.trace_id,
            kind=kind,
            payload=payload,
            error_code=error.code.value if error else None,
            error_message=error.message if error else None,
        )
        result.events.append(evt)
        fields = {"kind": kind.value, **payload}
        if error:
            fields.update(error.to_dict())
            log_event(_log, f"agent.{kind.value}", level=40, **fields)
        else:
            log_event(_log, f"agent.{kind.value}", **fields)

    def _set_state(self, task: TaskContext, new: TaskState) -> None:
        task.touch(transition(task.state, new))

    def _capture_frame(
        self,
        *,
        kind: FrameKind,
        trace_id: str,
        force: bool = True,
    ) -> tuple[ScreenFrame, Any]:
        """Return (meta, image_or_None). Supports packet and simple capture backends."""
        image = None
        if hasattr(self.capture, "capture_packet"):
            packet = self.capture.capture_packet(  # type: ignore[attr-defined]
                kind=kind, force=force, encode=True
            )
            frame = packet.meta
            frame.trace_id = frame.trace_id or trace_id
            image = getattr(packet, "image", None)
            if kind == FrameKind.MODEL and hasattr(packet, "attach_b64"):
                with contextlib.suppress(Exception):
                    packet.attach_b64()
            return frame, image
        frame = self.capture.capture(trace_id=trace_id)
        frame.frame_kind = kind
        return frame, image

    def _recognize(
        self,
        frame: ScreenFrame,
        *,
        trace_id: str,
        image: Any = None,
        goal: str | None = None,
    ) -> UIVisionResult:
        return self.vision.recognize(
            frame,
            trace_id=trace_id,
            image=image,
            goal=goal,
        )

    def _change_score(self, before: ScreenFrame, after: ScreenFrame) -> float | None:
        """Best-effort change score from frame metadata or hashes."""
        if before.change_score is not None:
            return float(before.change_score)
        if after.change_score is not None:
            return float(after.change_score)
        if before.pixel_hash and after.pixel_hash:
            return 0.0 if before.pixel_hash == after.pixel_hash else 1.0
        if before.frame_id != after.frame_id:
            return 0.05  # unknown but re-captured
        return None

    def _should_auto_confirm(self, step: ActionStep, decision: SafetyDecision) -> bool:
        if not decision.requires_confirmation:
            return True
        if not self.config.agent.auto_confirm:
            return False
        # Never auto-confirm high risk or blocked
        if decision.risk == RiskLevel.HIGH:
            return False
        return step.risk != RiskLevel.HIGH

    def _confirm(
        self,
        preview: ActionPreview,
        step: ActionStep,
        decision: SafetyDecision,
    ) -> bool:
        if self._should_auto_confirm(step, decision):
            return True
        if self.confirm_callback is not None:
            return bool(self.confirm_callback(preview, step, decision))
        # No callback and not auto → pause path (caller handles)
        return False

    # ------------------------------------------------------------------ public API
    def preview(self, user_goal: str, *, trace_id: str | None = None) -> TaskRunResult:
        """Observe + plan + build previews without executing."""
        return self.run(
            user_goal,
            trace_id=trace_id,
            execute=False,
            confirmed=False,
        )

    def run(
        self,
        user_goal: str,
        *,
        trace_id: str | None = None,
        execute: bool = True,
        confirmed: bool = False,
        max_steps: int | None = None,
    ) -> TaskRunResult:
        t0 = time.perf_counter()
        tid = trace_id or new_trace_id()
        task = TaskContext(user_goal=user_goal, trace_id=tid, state=TaskState.IDLE)
        if confirmed:
            task.confirmed = True
        result = TaskRunResult(trace_id=tid, task=task)
        max_steps = max_steps or self.config.agent.max_steps

        with trace_scope(tid):
            try:
                self.token.check()
                self._set_state(task, TaskState.OBSERVING)

                # 1. Capture (model frame for planning)
                frame, image = self._capture_frame(kind=FrameKind.MODEL, trace_id=tid, force=True)
                result.frame = frame
                task.frame_id = frame.frame_id
                self._emit(result, PipelineEventKind.FRAME, frame.log_summary())
                self.token.check()

                # 2. UI vision
                vision = self._recognize(frame, trace_id=tid, image=image, goal=user_goal)
                result.vision = vision
                self._emit(result, PipelineEventKind.VISION, vision.log_summary())
                self.token.check()

                # 3. Inference observation
                inf = self.inference.observe(
                    frame,
                    vision,
                    user_goal=user_goal,
                    trace_id=tid,
                    mode="observe_plan",
                    include_image=bool(frame.image_b64 or frame.image_path),
                )
                if not inf.ok or inf.observation is None:
                    # Soft: still allow mock planning from vision alone
                    from core.models import ScreenObservation as SO

                    observation = SO(
                        trace_id=tid,
                        frame_id=frame.frame_id,
                        observation=f"(degraded) goal={user_goal}",
                        confidence=0.3,
                        notes=inf.error_message or "inference degraded",
                    )
                else:
                    observation = inf.observation
                result.observation = observation
                self._emit(
                    result,
                    PipelineEventKind.INFERENCE,
                    inf.log_summary()
                    if hasattr(inf, "log_summary")
                    else {"ok": bool(getattr(inf, "ok", False))},
                )
                self.token.check()

                # 4. Plan
                self._set_state(task, TaskState.PLANNING)
                plan = self._make_plan(
                    user_goal,
                    vision,
                    observation,
                    frame=frame,
                    trace_id=tid,
                    inf_plan=getattr(inf, "plan", None),
                )
                if len(plan.steps) > max_steps:
                    plan.steps = plan.steps[:max_steps]
                result.plan = plan
                task.plan_id = plan.plan_id
                self._emit(result, PipelineEventKind.PLAN, plan.log_summary())

                # Previews for all steps
                previews = build_plan_previews(plan, vision)
                result.previews = previews
                for p in previews:
                    log_event(_log, "agent.preview", **p.log_summary())

                if not plan.steps:
                    self._set_state(task, TaskState.COMPLETED)
                    result.ok = True
                    self._emit(
                        result, PipelineEventKind.DONE, {"reason": "no_steps", **task.log_summary()}
                    )
                    return result

                if not execute:
                    self._set_state(task, TaskState.AWAITING_CONFIRMATION)
                    self._emit(
                        result,
                        PipelineEventKind.DONE,
                        {"reason": "preview_only", "step_count": len(plan.steps)},
                    )
                    result.ok = True
                    return result

                # 5. Execute steps with confirm / relocate / verify / recovery
                self._run_steps(result, plan, vision, frame)
            except CancelledError as exc:
                try:
                    self._set_state(task, TaskState.CANCELLED)
                except BaodouError:
                    task.touch(TaskState.CANCELLED)
                result.error = exc
                self._emit(result, PipelineEventKind.CANCEL, exc.to_dict(), error=exc)
            except BaodouError as exc:
                if not is_terminal(task.state):
                    try:
                        self._set_state(
                            task,
                            TaskState.PAUSED
                            if exc.code
                            in {
                                ErrorCode.CONFIRMATION_REQUIRED,
                                ErrorCode.TARGET_STALE,
                                ErrorCode.ELEMENT_NOT_FOUND,
                            }
                            else TaskState.FAILED,
                        )
                    except BaodouError:
                        task.touch(TaskState.FAILED)
                result.error = exc
                task.last_error = exc.message
                self._emit(result, PipelineEventKind.ERROR, exc.to_dict(), error=exc)
            except Exception as exc:  # noqa: BLE001
                err = BaodouError(ErrorCode.INTERNAL, str(exc), cause=exc)
                task.touch(TaskState.FAILED)
                result.error = err
                task.last_error = err.message
                self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
            finally:
                result.elapsed_ms = (time.perf_counter() - t0) * 1000
                log_event(
                    _log,
                    "agent.finished",
                    ok=result.ok,
                    elapsed_ms=result.elapsed_ms,
                    **task.log_summary(),
                    event_count=len(result.events),
                )
        return result

    def _make_plan(
        self,
        user_goal: str,
        vision: UIVisionResult,
        observation: ScreenObservation,
        *,
        frame: ScreenFrame,
        trace_id: str,
        inf_plan: ActionPlan | None,
    ) -> ActionPlan:
        # Prefer validated inference plan only in inference backend mode.
        if (
            inf_plan is not None
            and inf_plan.steps is not None
            and self.config.agent.backend == "inference"
        ):
            plan = inf_plan
            plan.trace_id = plan.trace_id or trace_id
            return plan
        # Planner may accept frame kw (InferencePlanner) or not (MockAgent)
        try:
            return self.planner.plan(  # type: ignore[call-arg]
                user_goal,
                vision,
                observation,
                trace_id=trace_id,
                frame=frame,
            )
        except TypeError:
            return self.planner.plan(user_goal, vision, observation, trace_id=trace_id)

    def _run_steps(
        self,
        result: TaskRunResult,
        plan: ActionPlan,
        vision: UIVisionResult,
        frame: ScreenFrame,
    ) -> None:
        task = result.task
        tid = result.trace_id
        current_vision = vision
        current_frame = frame
        prior_elements: dict[str, UIElement] = {e.element_id: e for e in vision.elements}
        i = 0
        while i < len(plan.steps):
            self.token.check()
            step = plan.steps[i]
            task.step_index = i
            record = StepRecord(
                step_id=step.step_id,
                index=i,
                action=step.action,
                state="pending",
            )
            task.step_records.append(record)

            # --- Preview ---
            preview = build_step_preview(step, current_vision, plan=plan)
            record.preview = preview
            record.state = "preview"
            if preview not in result.previews:
                # Keep step-local; full list already built — refresh this index
                if i < len(result.previews):
                    result.previews[i] = preview
                else:
                    result.previews.append(preview)
            log_event(_log, "agent.step_preview", **preview.log_summary(), index=i)

            # Coordinate warning → force confirmation
            if preview.uses_coordinates and self.config.agent.coordinate_requires_confirm:
                step.requires_confirmation = True

            # --- Safety ---
            decision = self.safety.evaluate(step, plan)
            record.safety = decision
            result.safety_decisions.append(decision)
            self._emit(
                result,
                PipelineEventKind.SAFETY,
                {**decision.log_summary(), "step_id": step.step_id},
            )

            if not decision.allowed:
                record.state = "failed"
                record.error_code = ErrorCode.RISK_BLOCKED.value
                record.error_message = decision.reason
                task.steps_failed += 1
                err = BaodouError(
                    ErrorCode.RISK_BLOCKED if decision.blocked_by else ErrorCode.PERMISSION_DENIED,
                    decision.reason,
                )
                result.error = err
                self._set_state(task, TaskState.FAILED)
                self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                return

            # --- Confirmation ---
            needs = decision.requires_confirmation or step.requires_confirmation
            if needs and not task.confirmed:
                self._set_state(task, TaskState.AWAITING_CONFIRMATION)
                ok_confirm = self._confirm(preview, step, decision)
                if ok_confirm:
                    task.auto_confirmed = self.config.agent.auto_confirm
                    task.confirmed = True  # session-level for remaining low-risk if auto
                    # For non-auto, confirm only this step — reset for next if not auto
                    if not self.config.agent.auto_confirm and self.confirm_callback:
                        task.confirmed = False  # per-step via callback already returned True
                        record.state = "confirmed"
                    else:
                        record.state = "confirmed"
                else:
                    # Pause for user
                    task.pause_reason = (
                        f"需要确认: {preview.summary} | 影响: {preview.expected_impact}"
                    )
                    record.state = "paused"
                    record.error_code = ErrorCode.CONFIRMATION_REQUIRED.value
                    record.error_message = task.pause_reason
                    err = BaodouError(
                        ErrorCode.CONFIRMATION_REQUIRED,
                        task.pause_reason,
                        details={
                            "preview": preview.log_summary(),
                            "step_id": step.step_id,
                        },
                    )
                    result.error = err
                    self._set_state(task, TaskState.PAUSED)
                    self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                    return
            else:
                record.state = "confirmed"

            # --- Re-identify before action ---
            if self.config.agent.reidentify_before_action and step.action not in (
                ActionType.NONE,
                ActionType.WAIT,
                ActionType.KEY,
                ActionType.HOTKEY,
            ):
                current_frame, image = self._capture_frame(
                    kind=FrameKind.VERIFY, trace_id=tid, force=True
                )
                current_vision = self._recognize(
                    current_frame,
                    trace_id=tid,
                    image=image,
                    goal=task.user_goal,
                )
                result.vision = current_vision
                result.frame = current_frame
                task.frame_id = current_frame.frame_id
                record.frame_id_before = current_frame.frame_id
                self._emit(
                    result,
                    PipelineEventKind.VISION,
                    {**current_vision.log_summary(), "phase": "pre_action"},
                )

            # Refresh preview after relocate frame
            preview = build_step_preview(step, current_vision, plan=plan)
            record.preview = preview
            if preview.warnings and any("未找到" in w or "不可见" in w for w in preview.warnings):
                # Target missing on fresh frame
                rec_action, rec_note = decide_recovery(
                    config=self.config,
                    task=task,
                    step=step,
                    error_code=ErrorCode.TARGET_STALE.value,
                    reason="; ".join(preview.warnings),
                )
                handled = self._apply_recovery(
                    result,
                    record,
                    rec_action,
                    rec_note,
                    plan=plan,
                    step_index=i,
                )
                if handled == "return":
                    return
                if handled == "retry":
                    continue
                if handled == "skip":
                    i += 1
                    task.confirmed = self.config.agent.auto_confirm and task.confirmed
                    continue

            # --- Execute ---
            self._set_state(task, TaskState.EXECUTING)
            prior = None
            if step.target_element_id:
                prior = prior_elements.get(step.target_element_id)
            coord_ok = bool(
                task.confirmed or self.config.agent.auto_confirm or step.allow_coordinate_fallback
            )
            try:
                action = self.actuator.execute(
                    step,
                    current_vision,
                    trace_id=tid,
                    coordinate_confirmed=coord_ok,
                    prior_element=prior,
                )
            except ActuatorError as exc:
                record.state = "failed"
                record.error_code = exc.code.value
                record.error_message = exc.message
                rec_action, rec_note = decide_recovery(
                    config=self.config,
                    task=task,
                    step=step,
                    error_code=exc.code.value,
                    reason=exc.message,
                )
                handled = self._apply_recovery(
                    result,
                    record,
                    rec_action,
                    rec_note,
                    plan=plan,
                    step_index=i,
                    error=exc,
                )
                if handled == "return":
                    return
                if handled == "retry":
                    task.recovery_attempts += 1
                    continue
                if handled == "skip":
                    i += 1
                    continue
                result.error = exc
                self._set_state(task, TaskState.FAILED)
                self._emit(result, PipelineEventKind.ERROR, exc.to_dict(), error=exc)
                return

            record.action_result = action
            record.state = "executed"
            result.actions.append(action)
            self._emit(result, PipelineEventKind.ACTION, action.log_summary())

            if action.resolved_element_id and current_vision.by_id(action.resolved_element_id):
                prior_elements[action.resolved_element_id] = current_vision.by_id(  # type: ignore[assignment]
                    action.resolved_element_id
                )

            if not action.success:
                rec_action, rec_note = decide_recovery(
                    config=self.config,
                    task=task,
                    step=step,
                    action=action,
                    reason=action.message,
                )
                handled = self._apply_recovery(
                    result, record, rec_action, rec_note, plan=plan, step_index=i
                )
                if handled == "return":
                    return
                if handled == "retry":
                    task.recovery_attempts += 1
                    continue
                if handled == "skip":
                    i += 1
                    continue

            # Settle + verify capture
            settle = self.config.agent.post_action_settle_ms / 1000.0
            if settle > 0:
                time.sleep(min(settle, 1.0))

            self._set_state(task, TaskState.VERIFYING)
            frame_before_id = record.frame_id_before or current_frame.frame_id
            vision_before = current_vision

            after_frame, after_image = self._capture_frame(
                kind=FrameKind.VERIFY, trace_id=tid, force=True
            )
            after_vision = self._recognize(
                after_frame, trace_id=tid, image=after_image, goal=task.user_goal
            )
            change = self._change_score(current_frame, after_frame)
            record.frame_id_after = after_frame.frame_id

            verification = self.actuator.verify(
                step,
                action,
                trace_id=tid,
                vision_before=vision_before,
                vision_after=after_vision,
                change_score=change,
                frame_id_before=frame_before_id,
                frame_id_after=after_frame.frame_id,
            )
            record.verification = verification
            result.verifications.append(verification)
            self._emit(result, PipelineEventKind.VERIFICATION, verification.log_summary())

            current_frame = after_frame
            current_vision = after_vision
            result.frame = after_frame
            result.vision = after_vision
            task.frame_id = after_frame.frame_id
            for el in after_vision.elements:
                prior_elements[el.element_id] = el

            if not verification.passed:
                rec_action, rec_note = decide_recovery(
                    config=self.config,
                    task=task,
                    step=step,
                    action=action,
                    verification=verification,
                    reason=verification.message,
                )
                handled = self._apply_recovery(
                    result, record, rec_action, rec_note, plan=plan, step_index=i
                )
                if handled == "return":
                    return
                if handled == "retry":
                    task.recovery_attempts += 1
                    continue
                if handled == "skip":
                    record.state = "skipped"
                    task.steps_skipped += 1
                    i += 1
                    # reset per-step confirm unless auto
                    if not self.config.agent.auto_confirm:
                        task.confirmed = False
                    continue

            record.state = "verified"
            task.steps_done += 1
            task.recovery_attempts = 0  # reset on success
            i += 1
            # Per-step confirmation unless auto_confirm
            if not self.config.agent.auto_confirm:
                task.confirmed = False

        self._set_state(task, TaskState.COMPLETED)
        result.ok = True
        self._emit(
            result,
            PipelineEventKind.DONE,
            {
                **task.log_summary(),
                "steps_total": len(plan.steps),
            },
        )

    def _apply_recovery(
        self,
        result: TaskRunResult,
        record: StepRecord,
        action: RecoveryAction,
        note: str,
        *,
        plan: ActionPlan,
        step_index: int,
        error: BaodouError | None = None,
    ) -> str:
        """Apply recovery. Returns 'return' | 'retry' | 'skip' | 'fail'."""
        task = result.task
        record.recovery = action
        record.recovery_note = note
        log_event(
            _log,
            "agent.recovery",
            recovery=action.value,
            note=note,
            step_id=record.step_id,
            attempts=task.recovery_attempts,
        )

        if action == RecoveryAction.REIDENTIFY:
            task.recovery_attempts += 1
            # Force fresh observe on next loop iteration
            try:
                frame, image = self._capture_frame(
                    kind=FrameKind.VISION, trace_id=result.trace_id, force=True
                )
                vision = self._recognize(
                    frame, trace_id=result.trace_id, image=image, goal=task.user_goal
                )
                result.frame = frame
                result.vision = vision
                task.frame_id = frame.frame_id
                self._emit(
                    result,
                    PipelineEventKind.VISION,
                    {**vision.log_summary(), "phase": "recovery_reidentify"},
                )
            except Exception as exc:  # noqa: BLE001
                log_event(_log, "agent.reidentify_failed", error=str(exc))
            return "retry"

        if action == RecoveryAction.RETRY_STEP:
            task.recovery_attempts += 1
            return "retry"

        if action == RecoveryAction.SKIP_STEP:
            record.state = "skipped"
            task.steps_skipped += 1
            return "skip"

        if action == RecoveryAction.GO_BACK:
            if step_index > 0:
                # Caller uses retry but we can't easily decrement from here;
                # treat as pause with note.
                task.pause_reason = f"建议回退: {note}"
                record.state = "paused"
                self._set_state(task, TaskState.PAUSED)
                err = error or BaodouError(
                    ErrorCode.VERIFICATION_FAILED,
                    task.pause_reason,
                )
                result.error = err
                self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                return "return"
            return "retry"

        if action == RecoveryAction.PAUSE:
            task.pause_reason = note
            record.state = "paused"
            self._set_state(task, TaskState.PAUSED)
            err = error or BaodouError(
                ErrorCode.CONFIRMATION_REQUIRED if "确认" in note else ErrorCode.TARGET_STALE,
                note,
            )
            result.error = err
            task.last_error = note
            self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
            return "return"

        # FAIL
        record.state = "failed"
        task.steps_failed += 1
        self._set_state(task, TaskState.FAILED)
        err = error or BaodouError(ErrorCode.ACTION_FAILED, note)
        result.error = err
        task.last_error = note
        self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
        return "return"

    def close(self) -> None:
        for obj in (self.vision, self.capture, self.inference):
            close = getattr(obj, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
