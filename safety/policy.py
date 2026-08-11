"""Risk evaluation and hard blocks (Phase B baseline)."""

from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import ActionPlan, ActionStep, RiskLevel, SafetyDecision

_log = get_logger("safety.policy")

_RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


class SafetyPolicy:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(self, step: ActionStep, plan: ActionPlan) -> SafetyDecision:
        risk = step.risk
        text_blob = " ".join(
            [
                plan.goal,
                step.text or "",
                step.target_element_id or "",
                step.expected_change,
            ]
        ).lower()

        for kw in self.config.safety.sensitive_keywords:
            if kw.lower() in text_blob:
                risk = RiskLevel.HIGH
                break

        mode = self.config.safety.default_mode

        if risk == RiskLevel.HIGH and self.config.safety.block_high_risk:
            decision = SafetyDecision(
                allowed=False,
                requires_confirmation=True,
                risk=risk,
                reason="高风险操作被硬编码拦截",
                blocked_by="block_high_risk",
            )
            log_event(_log, "safety.decision", **decision.log_summary())
            return decision

        if mode == "read_only":
            # Observation-only mode: any real action needs confirmation path.
            decision = SafetyDecision(
                allowed=True,
                requires_confirmation=True,
                risk=risk,
                reason="只读模式：执行需确认",
                blocked_by=None,
            )
            log_event(_log, "safety.decision", **decision.log_summary())
            return decision

        require_at = RiskLevel(self.config.safety.require_confirmation_below)
        needs_confirm = _RISK_ORDER[risk] >= _RISK_ORDER[require_at]
        if mode == "confirm_all":
            needs_confirm = True
        if mode == "allow_low" and risk == RiskLevel.LOW:
            needs_confirm = False

        if step.requires_confirmation:
            needs_confirm = True

        decision = SafetyDecision(
            allowed=True,
            requires_confirmation=needs_confirm,
            risk=risk,
            reason="通过策略检查",
        )
        log_event(_log, "safety.decision", **decision.log_summary())
        return decision
