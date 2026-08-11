"""Mock actuator: validates targets, relocates, never injects OS input."""

from __future__ import annotations

import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import ActuatorError
from core.logging import get_logger, log_event
from core.models import (
    ActionResult,
    ActionStep,
    ActionType,
    UIElement,
    UIVisionResult,
    VerificationResult,
)

from actuator.base import ActuatorBackend
from actuator.rate_limit import ActionRateLimiter
from actuator.relocate import relocate_target
from actuator.verify import verify_step

_log = get_logger("actuator.mock")


class MockActuator(ActuatorBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._history: list[ActionResult] = []
        self._limiter = ActionRateLimiter(config.actuator.max_actions_per_minute)
        self._prior_elements: dict[str, UIElement] = {}

    def execute(
        self,
        step: ActionStep,
        vision: UIVisionResult,
        *,
        trace_id: str = "",
        coordinate_confirmed: bool = False,
        prior_element: UIElement | None = None,
    ) -> ActionResult:
        get_global_token().check()
        t0 = time.perf_counter()
        self._limiter.check()

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

        if step.action == ActionType.WAIT:
            ms = step.wait_ms or 10
            time.sleep(min(ms, 50) / 1000.0)
            result = ActionResult(
                trace_id=trace_id,
                step_id=step.step_id,
                action=step.action,
                success=True,
                dry_run=True,
                message=f"wait {ms}ms (mock)",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._history.append(result)
            log_event(_log, "actuator.execute", **result.log_summary())
            return result

        if step.action == ActionType.REIDENTIFY:
            result = ActionResult(
                trace_id=trace_id,
                step_id=step.step_id,
                action=step.action,
                success=True,
                dry_run=True,
                message="reidentify (mock marker)",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._history.append(result)
            return result

        if step.action in (ActionType.KEY, ActionType.HOTKEY) and not step.target_element_id:
            # Keyboard-only: no relocate needed.
            self._limiter.record()
            result = ActionResult(
                trace_id=trace_id,
                step_id=step.step_id,
                action=step.action,
                success=True,
                dry_run=True,
                message=f"[dry_run] {step.action.value} keys={step.keys}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._history.append(result)
            log_event(_log, "actuator.execute", **result.log_summary())
            return result

        prior = prior_element
        if prior is None and step.target_element_id:
            prior = self._prior_elements.get(step.target_element_id)

        try:
            located = relocate_target(
                step,
                vision,
                self.config,
                prior_element=prior,
                coordinate_confirmed=coordinate_confirmed,
            )
        except ActuatorError:
            raise

        if located.element is not None:
            self._prior_elements[located.element.element_id] = located.element

        time.sleep(0.005)
        self._limiter.record()
        msg = (
            f"[dry_run={self.config.actuator.dry_run}] "
            f"{step.action.value} target={located.element_id or step.target_element_id} "
            f"point={located.point} method={located.method} text={step.text!r}"
        )
        result = ActionResult(
            trace_id=trace_id,
            step_id=step.step_id,
            action=step.action,
            success=True,
            dry_run=True,
            message=msg,
            latency_ms=(time.perf_counter() - t0) * 1000,
            resolved_point=located.point,
            resolved_element_id=located.element_id or step.target_element_id,
            relocated=located.relocated,
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
        vision_before: UIVisionResult | None = None,
        vision_after: UIVisionResult | None = None,
        change_score: float | None = None,
        frame_id_before: str | None = None,
        frame_id_after: str | None = None,
    ) -> VerificationResult:
        get_global_token().check()
        return verify_step(
            step,
            action,
            config=self.config,
            vision_before=vision_before,
            vision_after=vision_after,
            change_score=change_score,
            frame_id_before=frame_id_before,
            frame_id_after=frame_id_after,
            trace_id=trace_id,
        )
