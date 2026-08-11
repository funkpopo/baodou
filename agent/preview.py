"""Build user-visible action previews (Phase F)."""

from __future__ import annotations

from core.models import (
    ActionPlan,
    ActionPreview,
    ActionStep,
    ActionType,
    Point,
    UIElement,
    UIVisionResult,
)


def _element_label(el: UIElement | None) -> str:
    if el is None:
        return ""
    text = (el.text or el.name or "").strip()
    if text:
        return f'{el.type.value} "{text[:40]}"'
    return f"{el.type.value} ({el.element_id})"


def build_step_preview(
    step: ActionStep,
    vision: UIVisionResult,
    *,
    plan: ActionPlan | None = None,
) -> ActionPreview:
    el: UIElement | None = None
    if step.target_element_id:
        el = vision.by_id(step.target_element_id)

    point: Point | None = None
    uses_coords = False
    warnings: list[str] = []

    if el is not None:
        point = el.center
        if not el.visible:
            warnings.append("目标当前不可见")
        if not el.enabled:
            warnings.append("目标当前未启用")
        if (
            step.action
            in (
                ActionType.CLICK,
                ActionType.DOUBLE_CLICK,
                ActionType.RIGHT_CLICK,
                ActionType.DRAG,
            )
            and not el.clickable
        ):
            warnings.append("目标标记为不可点击")
        if step.action == ActionType.TYPE and not el.editable:
            warnings.append("目标标记为不可编辑")
        if el.needs_review or el.conflict:
            warnings.append("目标置信度低或存在冲突，建议人工确认")
    elif step.target_point is not None:
        point = step.target_point
        uses_coords = True
        warnings.append("使用裸坐标（无 element_id）— 需二次确认")
    elif step.target_element_id:
        warnings.append(f"element_id 在当前帧未找到: {step.target_element_id}")

    if step.allow_coordinate_fallback and step.target_element_id and el is None:
        uses_coords = True
        warnings.append("允许坐标回退，但当前无可用坐标")

    action = step.action
    target_desc = (
        _element_label(el)
        if el
        else (f"point=({point.x},{point.y})" if point else (step.target_element_id or "无目标"))
    )

    if action == ActionType.CLICK:
        summary = f"单击 {target_desc}"
    elif action == ActionType.DOUBLE_CLICK:
        summary = f"双击 {target_desc}"
    elif action == ActionType.RIGHT_CLICK:
        summary = f"右键 {target_desc}"
    elif action == ActionType.MOVE:
        summary = f"移动鼠标到 {target_desc}"
    elif action == ActionType.DRAG:
        end = step.end_point
        end_s = f"→ ({end.x},{end.y})" if end else "→ (未指定终点)"
        summary = f"拖拽 {target_desc} {end_s}"
    elif action == ActionType.SCROLL:
        summary = f"滚轮 dx={step.scroll_dx} dy={step.scroll_dy} @ {target_desc}"
    elif action == ActionType.TYPE:
        text_preview = (step.text or "")[:40]
        summary = f'在 {target_desc} 输入 "{text_preview}"'
    elif action == ActionType.KEY:
        summary = f"按键 {'+'.join(step.keys) or '(空)'}"
    elif action == ActionType.HOTKEY:
        summary = f"快捷键 {'+'.join(step.keys) or '(空)'}"
    elif action == ActionType.WAIT:
        summary = f"等待 {step.wait_ms or step.timeout_ms} ms"
    elif action == ActionType.REIDENTIFY:
        summary = "重新识别屏幕 UI"
    elif action == ActionType.NONE:
        summary = "无操作"
    else:
        summary = f"{action.value} {target_desc}"

    if step.description:
        summary = f"{step.description} — {summary}"

    impact = step.expected_change or "屏幕状态可能变化"
    if plan and plan.goal:
        impact = f"推进目标「{plan.goal[:40]}」: {impact}"

    return ActionPreview(
        step_id=step.step_id,
        action=action,
        summary=summary,
        target_element_id=step.target_element_id,
        target_text=(el.text or el.name or "") if el else "",
        target_type=el.type.value if el else "",
        target_bbox=el.bbox if el else None,
        target_point=point,
        risk=step.risk,
        requires_confirmation=step.requires_confirmation,
        expected_impact=impact,
        preconditions=list(step.preconditions),
        warnings=warnings,
        uses_coordinates=uses_coords,
    )


def build_plan_previews(plan: ActionPlan, vision: UIVisionResult) -> list[ActionPreview]:
    return [build_step_preview(s, vision, plan=plan) for s in plan.steps]
