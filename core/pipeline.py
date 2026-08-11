"""End-to-end mock pipeline: capture → vision → inference → plan → safety → act → verify.

Phase B acceptance: one trace_id correlates the full chain in structured logs.
"""

from __future__ import annotations

import time
from typing import Any

from actuator.mock import MockActuator
from agent.mock import MockAgent
from capture.mock import MockCapture
from inference.mock import MockInference
from safety.policy import SafetyPolicy
from ui_vision.mock import MockUIVision

from core.cancel import CancellationToken, get_global_token
from core.config import AppConfig
from core.errors import BaodouError, CancelledError, ErrorCode
from core.logging import get_logger, log_event, new_trace_id, trace_scope
from core.models import (
    ActionPlan,
    ActionResult,
    PipelineEvent,
    PipelineEventKind,
    SafetyDecision,
    ScreenFrame,
    ScreenObservation,
    TaskContext,
    TaskState,
    UIVisionResult,
    VerificationResult,
)

_log = get_logger("core.pipeline")


class PipelineResult:
    def __init__(self, trace_id: str, task: TaskContext) -> None:
        self.trace_id = trace_id
        self.task = task
        self.frame: ScreenFrame | None = None
        self.vision: UIVisionResult | None = None
        self.observation: ScreenObservation | None = None
        self.plan: ActionPlan | None = None
        self.safety: SafetyDecision | None = None
        self.action: ActionResult | None = None
        self.verification: VerificationResult | None = None
        self.events: list[PipelineEvent] = []
        self.error: BaodouError | None = None
        self.ok: bool = False
        self.elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "task_state": self.task.state.value,
            "frame": self.frame.log_summary() if self.frame else None,
            "vision": self.vision.log_summary() if self.vision else None,
            "observation": self.observation.log_summary() if self.observation else None,
            "plan": self.plan.log_summary() if self.plan else None,
            "safety": self.safety.log_summary() if self.safety else None,
            "action": self.action.log_summary() if self.action else None,
            "verification": self.verification.log_summary() if self.verification else None,
            "error": self.error.to_dict() if self.error else None,
            "events": [e.model_dump(mode="json") for e in self.events],
        }


class MockPipeline:
    """Composable pipeline using mock backends (Phase B)."""

    def __init__(self, config: AppConfig, token: CancellationToken | None = None) -> None:
        self.config = config
        self.token = token or get_global_token()
        self.capture = MockCapture(config)
        self.vision = MockUIVision(config)
        self.inference = MockInference(config)
        self.agent = MockAgent(config)
        self.safety = SafetyPolicy(config)
        self.actuator = MockActuator(config)

    def _emit(
        self,
        result: PipelineResult,
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
        level_fields = {"kind": kind.value, **payload}
        if error:
            level_fields.update(error.to_dict())
            log_event(_log, f"pipeline.{kind.value}", level=40, **level_fields)
        else:
            log_event(_log, f"pipeline.{kind.value}", **level_fields)

    def run(self, user_goal: str, *, trace_id: str | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        tid = trace_id or new_trace_id()
        task = TaskContext(user_goal=user_goal, trace_id=tid, state=TaskState.IDLE)
        result = PipelineResult(tid, task)

        with trace_scope(tid):
            try:
                self.token.check()
                task.touch(TaskState.OBSERVING)

                # 1. Capture
                frame = self.capture.capture(trace_id=tid)
                result.frame = frame
                task.frame_id = frame.frame_id
                self._emit(result, PipelineEventKind.FRAME, frame.log_summary())
                self.token.check()

                # 2. UI vision
                vision = self.vision.recognize(frame, trace_id=tid)
                result.vision = vision
                self._emit(result, PipelineEventKind.VISION, vision.log_summary())
                self.token.check()

                # 3. Inference (observation)
                inf = self.inference.observe(frame, vision, user_goal=user_goal, trace_id=tid)
                if not inf.ok or inf.observation is None:
                    raise BaodouError(
                        ErrorCode(inf.error_code or ErrorCode.INFERENCE_FAILED.value),
                        inf.error_message or "推理失败",
                    )
                result.observation = inf.observation
                self._emit(result, PipelineEventKind.INFERENCE, inf.log_summary())
                self.token.check()

                # 4. Plan
                task.touch(TaskState.PLANNING)
                plan = self.agent.plan(user_goal, vision, inf.observation, trace_id=tid)
                result.plan = plan
                task.plan_id = plan.plan_id
                self._emit(result, PipelineEventKind.PLAN, plan.log_summary())
                self.token.check()

                if not plan.steps:
                    task.touch(TaskState.COMPLETED)
                    result.ok = True
                    self._emit(result, PipelineEventKind.DONE, {"reason": "no_steps"})
                    return result

                step = plan.steps[0]

                # 5. Safety
                decision = self.safety.evaluate(step, plan)
                result.safety = decision
                self._emit(result, PipelineEventKind.SAFETY, decision.log_summary())
                if not decision.allowed:
                    task.touch(TaskState.FAILED)
                    err = BaodouError(
                        ErrorCode.RISK_BLOCKED
                        if decision.blocked_by
                        else ErrorCode.PERMISSION_DENIED,
                        decision.reason,
                    )
                    result.error = err
                    self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                    return result

                if decision.requires_confirmation:
                    task.touch(TaskState.AWAITING_CONFIRMATION)
                    # Phase B: dry-run auto-continues so the full chain is visible in logs.
                    # Real confirmation UI is Phase F/H; non-dry-run stops here.
                    if not self.config.actuator.dry_run:
                        err = BaodouError(
                            ErrorCode.CONFIRMATION_REQUIRED,
                            "需要用户确认后才能执行",
                            details={"mode": self.config.safety.default_mode},
                        )
                        task.touch(TaskState.PAUSED)
                        result.error = err
                        self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                        return result
                    log_event(
                        _log,
                        "pipeline.auto_confirm_dry_run",
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        risk=step.risk.value,
                    )

                # 6. Actuate (mock / dry-run)
                self.token.check()
                task.touch(TaskState.EXECUTING)
                action = self.actuator.execute(step, vision, trace_id=tid)
                result.action = action
                self._emit(result, PipelineEventKind.ACTION, action.log_summary())
                if not action.success:
                    task.touch(TaskState.FAILED)
                    err = BaodouError(ErrorCode.ACTION_FAILED, action.message)
                    result.error = err
                    self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                    return result

                # 7. Verify
                task.touch(TaskState.VERIFYING)
                verification = self.actuator.verify(step, action, trace_id=tid)
                result.verification = verification
                self._emit(result, PipelineEventKind.VERIFICATION, verification.log_summary())
                if not verification.passed:
                    task.touch(TaskState.FAILED)
                    err = BaodouError(ErrorCode.VERIFICATION_FAILED, verification.message)
                    result.error = err
                    self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
                    return result

                task.touch(TaskState.COMPLETED)
                result.ok = True
                self._emit(
                    result,
                    PipelineEventKind.DONE,
                    {
                        "task_id": task.task_id,
                        "plan_id": plan.plan_id,
                        "steps_done": 1,
                    },
                )
            except CancelledError as exc:
                task.touch(TaskState.CANCELLED)
                result.error = exc
                self._emit(result, PipelineEventKind.CANCEL, exc.to_dict(), error=exc)
            except BaodouError as exc:
                task.touch(TaskState.FAILED)
                result.error = exc
                self._emit(result, PipelineEventKind.ERROR, exc.to_dict(), error=exc)
            except Exception as exc:  # noqa: BLE001
                err = BaodouError(ErrorCode.INTERNAL, str(exc), cause=exc)
                task.touch(TaskState.FAILED)
                result.error = err
                self._emit(result, PipelineEventKind.ERROR, err.to_dict(), error=err)
            finally:
                result.elapsed_ms = (time.perf_counter() - t0) * 1000
                log_event(
                    _log,
                    "pipeline.finished",
                    ok=result.ok,
                    elapsed_ms=result.elapsed_ms,
                    task_state=task.state.value,
                    event_count=len(result.events),
                )
        return result
