"""Validate model output: schema, element existence, coords, action whitelist.

Illegal / incomplete / out-of-policy outputs are rejected and must never reach
the actuator without going through Safety (Phase F/G). Phase E only produces
validated InferenceResponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.errors import ErrorCode, InferenceError
from core.models import (
    ActionPlan,
    ActionStep,
    ActionType,
    BBox,
    Point,
    RiskLevel,
    ScreenFrame,
    ScreenObservation,
    UIElement,
    UIVisionResult,
)

from inference.prompts import ALLOWED_ACTIONS, SENSITIVE_ACTION_HINTS
from inference.schema import (
    ModelObservationOnly,
    ModelObservePlan,
    action_type_or_none,
    parse_model_payload,
    risk_or_low,
)


@dataclass
class ValidationIssue:
    code: str
    message: str
    path: str = ""
    fatal: bool = True


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    observation: ScreenObservation | None = None
    plan: ActionPlan | None = None
    model_payload: ModelObservePlan | ModelObservationOnly | None = None
    degraded: bool = False

    def fatal_messages(self) -> list[str]:
        return [i.message for i in self.issues if i.fatal]


def validate_model_output(
    raw: dict[str, Any] | list[Any] | None,
    *,
    frame: ScreenFrame,
    vision: UIVisionResult,
    user_goal: str,
    mode: str = "observe_plan",
    trace_id: str = "",
    model_name: str = "",
    latency_ms: float | None = None,
    raw_truncated: bool = False,
    allow_unknown_elements: bool = False,
    screen_bounds: BBox | None = None,
) -> ValidationResult:
    """Full validation pipeline. Returns ok=False if anything fatal fails."""
    issues: list[ValidationIssue] = []

    if raw is None:
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue("empty", "模型输出为空或不可解析", fatal=True)],
        )
    if isinstance(raw, list):
        issues.append(ValidationIssue("root_type", "根节点必须是 object", fatal=True))
        return ValidationResult(ok=False, issues=issues)
    if not isinstance(raw, dict):
        issues.append(ValidationIssue("root_type", f"根节点类型无效: {type(raw)}", fatal=True))
        return ValidationResult(ok=False, issues=issues)

    if raw_truncated:
        issues.append(
            ValidationIssue(
                "truncated",
                "输出被截断，拒绝进入操作层",
                fatal=True,
            )
        )

    try:
        payload = parse_model_payload(raw, mode=mode)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            ok=False,
            issues=[
                ValidationIssue(
                    "schema",
                    f"schema 校验失败: {exc}",
                    fatal=True,
                )
            ],
        )

    # Build observation (always, even if plan invalid — caller may degrade).
    known_ids = {e.element_id for e in vision.elements}
    ui_from_model: list[UIElement] = []
    # Prefer real vision elements referenced by id
    for cand in payload.ui_candidates[:12]:
        if cand.element_id and cand.element_id in known_ids:
            el = vision.by_id(cand.element_id)
            if el is not None:
                ui_from_model.append(el)
        elif cand.element_id and not allow_unknown_elements:
            issues.append(
                ValidationIssue(
                    "unknown_element",
                    f"ui_candidate 未知 element_id: {cand.element_id}",
                    path="ui_candidates",
                    fatal=False,
                )
            )

    if not ui_from_model:
        ui_from_model = vision.elements[:6]

    observation = ScreenObservation(
        trace_id=trace_id,
        frame_id=frame.frame_id,
        observation=(payload.observation or "")[:2000],
        ui_elements=ui_from_model,
        notes=(payload.notes or "")[:1000],
        confidence=float(payload.confidence),
        model_name=model_name,
        latency_ms=latency_ms,
        raw_truncated=raw_truncated,
    )

    plan: ActionPlan | None = None
    if isinstance(payload, ModelObservePlan) and mode != "observation":
        plan, plan_issues = _validate_plan(
            payload,
            vision=vision,
            user_goal=user_goal,
            frame=frame,
            trace_id=trace_id,
            allow_unknown_elements=allow_unknown_elements,
            screen_bounds=screen_bounds,
        )
        issues.extend(plan_issues)

    fatal = [i for i in issues if i.fatal]
    ok = len(fatal) == 0 and not raw_truncated
    return ValidationResult(
        ok=ok,
        issues=issues,
        observation=observation,
        plan=plan if ok else None,  # never pass plan if fatal issues
        model_payload=payload,
    )


def _validate_plan(
    payload: ModelObservePlan,
    *,
    vision: UIVisionResult,
    user_goal: str,
    frame: ScreenFrame,
    trace_id: str,
    allow_unknown_elements: bool,
    screen_bounds: BBox | None,
) -> tuple[ActionPlan, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    known = {e.element_id: e for e in vision.elements}
    bounds = screen_bounds or BBox(
        x=frame.origin_x,
        y=frame.origin_y,
        width=frame.physical_width or int(frame.width * frame.scale_x) or frame.width,
        height=frame.physical_height or int(frame.height * frame.scale_y) or frame.height,
    )

    steps: list[ActionStep] = []
    risk_max = RiskLevel.LOW
    goal = (payload.plan.goal or user_goal or "").strip() or user_goal

    for idx, raw_step in enumerate(payload.plan.steps[:8]):
        action_raw = (raw_step.action or "none").strip().lower()
        if action_raw not in ALLOWED_ACTIONS:
            issues.append(
                ValidationIssue(
                    "action_whitelist",
                    f"非法动作 '{action_raw}' 不在白名单",
                    path=f"plan.steps[{idx}].action",
                    fatal=True,
                )
            )
            continue

        action = action_type_or_none(action_raw)
        if action == ActionType.NONE and action_raw not in {"none", "wait", "reidentify"}:
            # action_type_or_none maps unknown → NONE; already handled by whitelist
            pass

        # Drag intentionally not in ALLOWED for MVP sketch; if ever added, keep confirm.
        risk = risk_or_low(raw_step.risk)
        requires = bool(raw_step.requires_confirmation)
        # Sensitive text in type / goal elevates risk
        blob = f"{goal} {raw_step.text or ''} {raw_step.expected_change or ''}".lower()
        if any(h in blob for h in SENSITIVE_ACTION_HINTS):
            risk = RiskLevel.HIGH
            requires = True

        target_id = raw_step.target_element_id
        target_point: Point | None = None

        if target_id:
            if target_id not in known:
                if allow_unknown_elements:
                    issues.append(
                        ValidationIssue(
                            "unknown_element",
                            f"目标 element_id 未知: {target_id}",
                            path=f"plan.steps[{idx}].target_element_id",
                            fatal=False,
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            "unknown_element",
                            f"目标 element_id 不存在于当前 UI 列表: {target_id}",
                            path=f"plan.steps[{idx}].target_element_id",
                            fatal=True,
                        )
                    )
                    continue
            else:
                el = known[target_id]
                if el.is_stale_for_frame(frame.frame_id):
                    issues.append(
                        ValidationIssue(
                            "stale_element",
                            f"目标已失效 (frame mismatch): {target_id}",
                            path=f"plan.steps[{idx}].target_element_id",
                            fatal=True,
                        )
                    )
                    continue
                if action in {
                    ActionType.CLICK,
                    ActionType.DOUBLE_CLICK,
                    ActionType.RIGHT_CLICK,
                    ActionType.TYPE,
                }:
                    if not el.visible:
                        issues.append(
                            ValidationIssue(
                                "element_not_visible",
                                f"目标不可见: {target_id}",
                                path=f"plan.steps[{idx}]",
                                fatal=True,
                            )
                        )
                        continue
                    if action != ActionType.TYPE and not el.clickable and not el.enabled:
                        issues.append(
                            ValidationIssue(
                                "element_not_interactive",
                                f"目标不可交互: {target_id}",
                                path=f"plan.steps[{idx}]",
                                fatal=False,
                            )
                        )

        if raw_step.target_point is not None:
            # Bare coordinates only allowed when no element_id OR as secondary.
            px, py = raw_step.target_point.x, raw_step.target_point.y
            if not bounds.contains(px, py):
                # Also accept image-space mistakenly? Still reject — must be physical.
                issues.append(
                    ValidationIssue(
                        "coord_oob",
                        f"坐标越界: ({px},{py}) not in {bounds.model_dump()}",
                        path=f"plan.steps[{idx}].target_point",
                        fatal=True,
                    )
                )
                continue
            if target_id is None and action in {
                ActionType.CLICK,
                ActionType.DOUBLE_CLICK,
                ActionType.RIGHT_CLICK,
                ActionType.MOVE,
            }:
                # Prefer element_id; bare point forces confirmation
                requires = True
                issues.append(
                    ValidationIssue(
                        "bare_coords",
                        "仅有裸坐标、无 element_id，强制确认",
                        path=f"plan.steps[{idx}].target_point",
                        fatal=False,
                    )
                )
            target_point = Point(x=px, y=py)

        if (
            action
            in {
                ActionType.CLICK,
                ActionType.DOUBLE_CLICK,
                ActionType.RIGHT_CLICK,
                ActionType.TYPE,
                ActionType.MOVE,
            }
            and target_id is None
            and target_point is None
            and action != ActionType.WAIT
        ):
            issues.append(
                ValidationIssue(
                    "missing_target",
                    f"动作 {action.value} 缺少 target_element_id / target_point",
                    path=f"plan.steps[{idx}]",
                    fatal=True,
                )
            )
            continue

        if action == ActionType.TYPE and not (raw_step.text or "").strip():
            issues.append(
                ValidationIssue(
                    "missing_text",
                    "type 动作缺少 text",
                    path=f"plan.steps[{idx}].text",
                    fatal=True,
                )
            )
            continue

        if action in {ActionType.KEY, ActionType.HOTKEY} and not raw_step.keys:
            issues.append(
                ValidationIssue(
                    "missing_keys",
                    f"{action.value} 动作缺少 keys",
                    path=f"plan.steps[{idx}].keys",
                    fatal=True,
                )
            )
            continue

        # Executable steps always confirm by default in Phase E (read-only policy later)
        if action not in {ActionType.NONE, ActionType.WAIT, ActionType.REIDENTIFY}:
            requires = True

        step = ActionStep(
            action=action,
            target_element_id=target_id,
            target_point=target_point,
            text=raw_step.text,
            keys=list(raw_step.keys or []),
            risk=risk,
            requires_confirmation=requires,
            preconditions=list(raw_step.preconditions or []),
            expected_change=raw_step.expected_change or "",
        )
        steps.append(step)
        if ["low", "medium", "high"].index(risk.value) > ["low", "medium", "high"].index(
            risk_max.value
        ):
            risk_max = risk

    # Drop pure none steps
    steps = [s for s in steps if s.action != ActionType.NONE]

    plan = ActionPlan(
        trace_id=trace_id,
        goal=goal,
        steps=steps,
        stop_if=list(payload.plan.stop_if or ["target_missing", "window_changed"]),
        risk_max=risk_max,
    )
    return plan, issues


def raise_if_invalid(result: ValidationResult) -> None:
    if result.ok:
        return
    msgs = result.fatal_messages() or [i.message for i in result.issues]
    raise InferenceError(
        "; ".join(msgs)[:500],
        code=ErrorCode.OUTPUT_SCHEMA_INVALID,
        details={
            "issues": [
                {"code": i.code, "message": i.message, "path": i.path} for i in result.issues
            ]
        },
    )
