"""Failure recovery policies (Phase F)."""

from __future__ import annotations

from core.config import AppConfig
from core.models import (
    ActionResult,
    ActionStep,
    RecoveryAction,
    TaskContext,
    VerificationResult,
)


def decide_recovery(
    *,
    config: AppConfig,
    task: TaskContext,
    step: ActionStep,
    action: ActionResult | None = None,
    verification: VerificationResult | None = None,
    error_code: str | None = None,
    reason: str = "",
) -> tuple[RecoveryAction, str]:
    """Choose recovery for a failed / stalled step.

    Priority:
      1. target missing / stale → reidentify (bounded) then pause
      2. optional step → skip
      3. action failed → retry once via reidentify, then pause
      4. verify failed → reidentify or pause per config
      5. otherwise → fail
    """
    attempts = task.recovery_attempts
    max_attempts = max(0, config.agent.max_recovery_attempts)
    code = (error_code or "").lower()
    note = reason or ""

    target_issues = {
        "target_stale",
        "target_invalid",
        "element_not_found",
        "target_missing",
    }
    if code in target_issues or "missing" in note.lower() or "stale" in note.lower():
        if attempts < max_attempts and config.agent.reidentify_before_action:
            return RecoveryAction.REIDENTIFY, note or "目标失效，重新识别"
        if config.agent.pause_on_target_missing:
            return RecoveryAction.PAUSE, note or "目标消失，暂停等待用户"
        return RecoveryAction.FAIL, note or "目标失效且不可恢复"

    if step.optional and (
        (action is not None and not action.success)
        or (verification is not None and not verification.passed)
        or code
    ):
        return RecoveryAction.SKIP_STEP, note or "可选步骤失败，跳过"

    if action is not None and not action.success:
        if attempts < max_attempts:
            return RecoveryAction.REIDENTIFY, note or "执行失败，重新定位后重试"
        return RecoveryAction.PAUSE, note or "执行失败，暂停"

    if verification is not None and not verification.passed:
        if attempts < max_attempts:
            return RecoveryAction.REIDENTIFY, note or "验证失败，重新识别"
        if config.agent.pause_on_verify_fail:
            return RecoveryAction.PAUSE, note or "验证失败，暂停请求用户处理"
        return RecoveryAction.FAIL, note or "验证失败"

    if attempts < max_attempts:
        return RecoveryAction.RETRY_STEP, note or "重试当前步骤"

    return RecoveryAction.FAIL, note or "无法恢复"
