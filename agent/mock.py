"""Mock agent: rule-based multi-step plans from keywords + UI elements."""

from __future__ import annotations

from core.cancel import get_global_token
from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import (
    ActionPlan,
    ActionStep,
    ActionType,
    ElementType,
    RiskLevel,
    ScreenObservation,
    UIVisionResult,
)

from agent.base import AgentBackend

_log = get_logger("agent.mock")


class MockAgent(AgentBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def plan(
        self,
        user_goal: str,
        vision: UIVisionResult,
        observation: ScreenObservation,
        *,
        trace_id: str = "",
    ) -> ActionPlan:
        get_global_token().check()
        goal_l = user_goal.lower()
        steps: list[ActionStep] = []
        risk_max = RiskLevel.LOW

        # Read-only goals → empty plan (observe only).
        read_only_hints = ("读取", "描述", "看看", "是什么", "summarize", "describe", "read")
        if any(h in goal_l for h in read_only_hints) and not any(
            h in goal_l for h in ("点击", "输入", "click", "type", "打开")
        ):
            plan = ActionPlan(
                trace_id=trace_id,
                goal=user_goal,
                steps=[],
                stop_if=["target_missing"],
                risk_max=RiskLevel.LOW,
            )
            log_event(_log, "agent.plan", **plan.log_summary(), mode="observe_only")
            return plan

        # Prefer search button for click-ish goals.
        target = None
        if any(h in goal_l for h in ("搜索", "search")):
            target = next((e for e in vision.elements if e.element_id == "btn_search_01"), None)
            if target is None:
                target = next(
                    (
                        e
                        for e in vision.elements
                        if e.clickable
                        and ("搜索" in (e.text or "") or "search" in (e.text or "").lower())
                    ),
                    None,
                )
        if target is None and any(h in goal_l for h in ("确定", "ok", "确认")):
            target = next((e for e in vision.elements if e.element_id == "btn_ok_01"), None)
            if target is None:
                target = next(
                    (
                        e
                        for e in vision.elements
                        if e.clickable
                        and any(t in (e.text or "") for t in ("确定", "OK", "Ok", "确认"))
                    ),
                    None,
                )
        if target is None and any(h in goal_l for h in ("设置", "settings")):
            target = next(
                (
                    e
                    for e in vision.elements
                    if e.clickable
                    and ("设置" in (e.text or "") or "setting" in (e.text or "").lower())
                ),
                None,
            )
        if target is None:
            target = next((e for e in vision.elements if e.clickable), None)

        # Type step first when goal asks for input.
        if any(h in goal_l for h in ("输入", "type", "填写", "搜索")):
            inp = next(
                (e for e in vision.elements if e.type == ElementType.INPUT and e.editable),
                None,
            )
            if inp is None:
                inp = next((e for e in vision.elements if e.editable), None)
            if inp is not None and any(h in goal_l for h in ("输入", "type", "填写")):
                # Extract simple quoted text or fallback
                text = "mock query"
                for q in ('"', "“", "'"):
                    if q in user_goal:
                        parts = user_goal.split(q)
                        if len(parts) >= 3:
                            text = parts[1]
                            break
                steps.append(
                    ActionStep(
                        action=ActionType.TYPE,
                        target_element_id=inp.element_id,
                        text=text,
                        risk=RiskLevel.LOW,
                        requires_confirmation=True,
                        preconditions=["element.visible == true", "element.editable == true"],
                        expected_change="input text updated",
                        description="输入文本",
                        timeout_ms=self.config.agent.step_timeout_ms,
                    )
                )

        # Open browser style: wait + click
        if any(h in goal_l for h in ("打开", "open", "启动")):
            steps.insert(
                0,
                ActionStep(
                    action=ActionType.WAIT,
                    wait_ms=50,
                    risk=RiskLevel.LOW,
                    requires_confirmation=False,
                    expected_change="ready",
                    description="短暂等待",
                    timeout_ms=1000,
                ),
            )

        if target is not None and (
            any(h in goal_l for h in ("点击", "click", "打开", "按", "搜索")) or not steps
        ):
            risk = RiskLevel.LOW
            requires = True
            if any(k in goal_l for k in ("删除", "delete", "支付", "转账", "password", "密码")):
                risk = RiskLevel.HIGH
                risk_max = RiskLevel.HIGH
            steps.append(
                ActionStep(
                    action=ActionType.CLICK,
                    target_element_id=target.element_id,
                    risk=risk,
                    requires_confirmation=requires,
                    preconditions=["element.visible == true", "element.clickable == true"],
                    expected_change=f"interact with {target.element_id}",
                    description=f"点击 {target.text or target.element_id}",
                    timeout_ms=self.config.agent.step_timeout_ms,
                )
            )

        # Multi-step: if both type and click requested, keep order type→click
        # Deduplicate identical consecutive actions on same target
        deduped: list[ActionStep] = []
        for s in steps:
            if (
                deduped
                and deduped[-1].action == s.action
                and deduped[-1].target_element_id == s.target_element_id
                and deduped[-1].text == s.text
            ):
                continue
            deduped.append(s)
        steps = deduped[: self.config.agent.max_steps]

        if steps:
            risk_max = max(
                (s.risk for s in steps), key=lambda r: ["low", "medium", "high"].index(r.value)
            )

        plan = ActionPlan(
            trace_id=trace_id,
            goal=user_goal,
            steps=steps,
            stop_if=["target_missing", "window_changed", "confidence_below_threshold"],
            risk_max=risk_max,
        )
        log_event(_log, "agent.plan", **plan.log_summary())
        return plan
