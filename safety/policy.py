"""Risk evaluation, hard blocks, confirmation policy (Phase G)."""

from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import (
    ActionPlan,
    ActionStep,
    RiskCategory,
    RiskLevel,
    SafetyDecision,
    ScreenFrame,
    UIVisionResult,
)

from safety.risk import classify_step, level_to_category, max_category, risk_order
from safety.targets import check_targets
from safety.threats import ThreatReport, scan_coordinate_spoof, scan_plan

_log = get_logger("safety.policy")


class SafetyPolicy:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(
        self,
        step: ActionStep,
        plan: ActionPlan,
        *,
        frame: ScreenFrame | None = None,
        vision: UIVisionResult | None = None,
        screen_texts: list[str] | None = None,
        threat_report: ThreatReport | None = None,
    ) -> SafetyDecision:
        """
        Full Phase G gate for one step.

        Order:
          1. Action whitelist
          2. Risk classification + sensitive keywords
          3. Target allow/deny
          4. Threat findings (injection / coord spoof)
          5. High-risk hard block
          6. Mode → confirmation / auto-exec
        """
        cfg = self.config.safety
        rules: list[str] = []

        # --- 1. Action whitelist ---
        action_name = step.action.value
        if action_name not in cfg.action_whitelist:
            return self._deny(
                reason=f"动作不在白名单: {action_name}",
                blocked_by="action_whitelist",
                rules=["action_whitelist"],
            )

        # --- 2. Classify risk ---
        cat, risk, class_rules = classify_step(
            step,
            plan=plan,
            sensitive_keywords=cfg.sensitive_keywords,
        )
        rules.extend(class_rules)

        # --- 3. Target lists ---
        target = check_targets(
            cfg=cfg,
            step=step,
            frame=frame,
            vision=vision,
            extra_text=plan.goal,
        )
        if not target.allowed:
            return self._deny(
                reason=target.reason,
                blocked_by=target.blocked_by or "target_block",
                rules=rules + [target.blocked_by or "target_block"],
            )
        rules.append("targets_ok")

        # --- 4. Threats ---
        report = threat_report
        if report is None and cfg.block_prompt_injection:
            report = scan_plan(plan, cfg=cfg, screen_texts=screen_texts)

        if report is not None:
            for f in report.findings:
                rules.append(f"threat:{f.threat_id}:{f.source}")

            # Screen-sourced high injection → block entire action path
            if cfg.ignore_screen_instructions:
                screen_high = [f for f in report.findings if f.severity == "high" and f.source == "screen"]
                if screen_high:
                    f0 = screen_high[0]
                    return self._deny(
                        reason=f0.message,
                        blocked_by=f"threat:{f0.threat_id}",
                        rules=rules,
                    )

            # Per-step coordinate absurd → hard block
            local_high = [f for f in scan_coordinate_spoof(step) if f.severity == "high"]
            if local_high:
                f0 = local_high[0]
                return self._deny(
                    reason=f0.message,
                    blocked_by=f"threat:{f0.threat_id}",
                    rules=rules + [f"threat:{f0.threat_id}"],
                )

            # Medium coordinate spoof → force confirm + elevate
            local_med = [f for f in scan_coordinate_spoof(step) if f.severity == "medium"]
            if local_med:
                step.requires_confirmation = True
                cat = max_category(cat, RiskCategory.MEDIUM)
                if risk_order(risk) < risk_order(RiskLevel.MEDIUM):
                    risk = RiskLevel.MEDIUM
                rules.append("coord_spoof_confirm")

        # --- 5. High-risk hard block ---
        if risk == RiskLevel.HIGH and cfg.block_high_risk:
            return self._deny(
                reason="高风险操作被硬编码拦截",
                blocked_by="block_high_risk",
                rules=rules + ["block_high_risk"],
            )

        # --- 6. Mode → confirmation ---
        mode = cfg.default_mode
        require_at = RiskLevel(cfg.require_confirmation_below)
        needs_confirm = risk_order(risk) >= risk_order(require_at)
        auto_ok = False

        if cat == RiskCategory.OBSERVE and step.action.value in ("none", "wait", "reidentify"):
            needs_confirm = False
            auto_ok = True
            rules.append("observe_no_confirm")

        if mode == "read_only":
            if cat != RiskCategory.OBSERVE:
                needs_confirm = True
                auto_ok = False
                rules.append("read_only_confirm")
            reason = "只读模式：执行需确认" if needs_confirm else "只读/观察步骤"
        elif mode == "confirm_all":
            needs_confirm = True
            auto_ok = False
            rules.append("confirm_all")
            reason = "全部确认模式"
        elif mode == "allow_low":
            if risk == RiskLevel.LOW and cat in (RiskCategory.OBSERVE, RiskCategory.LOW):
                needs_confirm = False
                auto_ok = True
                rules.append("allow_low_auto")
            else:
                needs_confirm = True
                auto_ok = False
                rules.append("allow_low_need_confirm")
            reason = "低风险可自动；中高风险需确认"
        else:
            reason = "通过策略检查"

        # Medium+ always confirm (Phase G explicit rule)
        if risk_order(risk) >= risk_order(RiskLevel.MEDIUM):
            needs_confirm = True
            auto_ok = False
            rules.append("medium_plus_confirm")

        if step.requires_confirmation:
            needs_confirm = True
            auto_ok = False
            rules.append("step_requires_confirmation")

        if risk == RiskLevel.HIGH:
            cat = RiskCategory.HIGH
        else:
            cat = max_category(cat, level_to_category(risk))

        decision = SafetyDecision(
            allowed=True,
            requires_confirmation=needs_confirm,
            risk=risk,
            category=cat,
            reason=reason,
            blocked_by=None,
            rules_hit=rules[:16],
            auto_executable=bool(auto_ok and not needs_confirm),
        )
        log_event(_log, "safety.decision", **decision.log_summary())
        return decision

    def evaluate_plan_threats(
        self,
        plan: ActionPlan,
        *,
        screen_texts: list[str] | None = None,
    ) -> ThreatReport:
        return scan_plan(plan, cfg=self.config.safety, screen_texts=screen_texts)

    def _deny(
        self,
        *,
        reason: str,
        blocked_by: str,
        rules: list[str],
    ) -> SafetyDecision:
        decision = SafetyDecision(
            allowed=False,
            requires_confirmation=True,
            risk=RiskLevel.HIGH,
            category=RiskCategory.HIGH,
            reason=reason,
            blocked_by=blocked_by,
            rules_hit=rules[:16],
            auto_executable=False,
        )
        log_event(_log, "safety.decision", **decision.log_summary())
        return decision
