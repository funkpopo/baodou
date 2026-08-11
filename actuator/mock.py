"""Mock actuator: validates element_id existence, never moves mouse/keyboard."""

from __future__ import annotations

import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import ActuatorError, ErrorCode
from core.logging import get_logger, log_event
from core.models import (
    ActionResult,
    ActionStep,
    ActionType,
    UIVisionResult,
    VerificationResult,
)

from actuator.base import ActuatorBackend

_log = get_logger("actuator.mock")


class MockActuator(ActuatorBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._history: list[ActionResult] = []

    def execute(
        self,
        step: ActionStep,
        vision: UIVisionResult,
        *,
        trace_id: str = "",
    ) -> ActionResult:
        get_global_token().check()
        t0 = time.perf_counter()

        if step.action == ActionType.NONE:
            result = ActionResult(
                trace_id=trace_id,
                step_id=step.step_id,
                action=step.action,
                success=True,
                dry_run=True,
                message="no-op",
                latency_ms=0.0,
            )
            self._history.append(result)
            return result

        if step.target_element_id:
            found = next(
                (e for e in vision.elements if e.element_id == step.target_element_id),
                None,
            )
            if found is None:
                raise ActuatorError(
                    f"目标元素不存在: {step.target_element_id}",
                    code=ErrorCode.TARGET_INVALID,
                    details={"element_id": step.target_element_id},
                )
            if not found.visible:
                raise ActuatorError(
                    f"目标元素不可见: {step.target_element_id}",
                    code=ErrorCode.TARGET_STALE,
                )

        time.sleep(0.005)
        msg = (
            f"[dry_run={self.config.actuator.dry_run}] "
            f"{step.action.value} target={step.target_element_id} text={step.text!r}"
        )
        result = ActionResult(
            trace_id=trace_id,
            step_id=step.step_id,
            action=step.action,
            success=True,
            dry_run=self.config.actuator.dry_run,
            message=msg,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self._history.append(result)
        log_event(_log, "actuator.execute", **result.log_summary())
        return result

    def verify(
        self,
        step: ActionStep,
        action: ActionResult,
        *,
        trace_id: str = "",
    ) -> VerificationResult:
        get_global_token().check()
        passed = action.success
        ver = VerificationResult(
            trace_id=trace_id,
            step_id=step.step_id,
            passed=passed,
            expected=step.expected_change or "action success",
            actual=action.message,
            message="mock verify ok" if passed else "mock verify failed",
        )
        log_event(_log, "actuator.verify", **ver.log_summary())
        return ver
