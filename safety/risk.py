"""Risk classification helpers (Phase G).

Buckets:
  observe  — no OS input (empty plan / none / reidentify / wait-only)
  low      — click / move / scroll / right_click / double_click
  medium   — type / drag / key / hotkey (data modification or key chords)
  high     — sensitive keywords, irreversible patterns, elevated by policy
"""

from __future__ import annotations

from core.models import ActionPlan, ActionStep, ActionType, RiskCategory, RiskLevel

# Action → base category (before keyword elevation)
_ACTION_CATEGORY: dict[ActionType, RiskCategory] = {
    ActionType.NONE: RiskCategory.OBSERVE,
    ActionType.WAIT: RiskCategory.OBSERVE,
    ActionType.REIDENTIFY: RiskCategory.OBSERVE,
    ActionType.MOVE: RiskCategory.LOW,
    ActionType.CLICK: RiskCategory.LOW,
    ActionType.DOUBLE_CLICK: RiskCategory.LOW,
    ActionType.RIGHT_CLICK: RiskCategory.LOW,
    ActionType.SCROLL: RiskCategory.LOW,
    ActionType.DRAG: RiskCategory.MEDIUM,
    ActionType.TYPE: RiskCategory.MEDIUM,
    ActionType.KEY: RiskCategory.MEDIUM,
    ActionType.HOTKEY: RiskCategory.MEDIUM,
}

_CATEGORY_TO_LEVEL = {
    RiskCategory.OBSERVE: RiskLevel.LOW,
    RiskCategory.LOW: RiskLevel.LOW,
    RiskCategory.MEDIUM: RiskLevel.MEDIUM,
    RiskCategory.HIGH: RiskLevel.HIGH,
}

_LEVEL_TO_CATEGORY = {
    RiskLevel.LOW: RiskCategory.LOW,
    RiskLevel.MEDIUM: RiskCategory.MEDIUM,
    RiskLevel.HIGH: RiskCategory.HIGH,
}

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}

_CATEGORY_ORDER = {
    RiskCategory.OBSERVE: 0,
    RiskCategory.LOW: 1,
    RiskCategory.MEDIUM: 2,
    RiskCategory.HIGH: 3,
}


def risk_order(level: RiskLevel) -> int:
    return _RISK_ORDER.get(level, 0)


def category_order(cat: RiskCategory) -> int:
    return _CATEGORY_ORDER.get(cat, 0)


def max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if risk_order(a) >= risk_order(b) else b


def max_category(a: RiskCategory, b: RiskCategory) -> RiskCategory:
    return a if category_order(a) >= category_order(b) else b


def category_to_level(cat: RiskCategory) -> RiskLevel:
    return _CATEGORY_TO_LEVEL[cat]


def level_to_category(level: RiskLevel) -> RiskCategory:
    return _LEVEL_TO_CATEGORY.get(level, RiskCategory.LOW)


def base_category_for_action(action: ActionType) -> RiskCategory:
    return _ACTION_CATEGORY.get(action, RiskCategory.MEDIUM)


def classify_step(
    step: ActionStep,
    *,
    plan: ActionPlan | None = None,
    sensitive_keywords: list[str] | None = None,
) -> tuple[RiskCategory, RiskLevel, list[str]]:
    """
    Return (category, elevated_risk_level, rules_hit).
    Combines action type, declared step.risk, and sensitive keyword hits.
    """
    rules: list[str] = []
    cat = base_category_for_action(step.action)
    rules.append(f"action:{step.action.value}->{cat.value}")

    # Declared model/planner risk elevates category floor.
    # Pure observe actions (none/wait/reidentify) stay observe unless medium+.
    declared = level_to_category(step.risk)
    if cat == RiskCategory.OBSERVE:
        if risk_order(step.risk) >= risk_order(RiskLevel.MEDIUM):
            cat = declared
            rules.append(f"declared_risk:{step.risk.value}")
    elif category_order(declared) > category_order(cat):
        cat = declared
        rules.append(f"declared_risk:{step.risk.value}")

    text_blob = " ".join(
        [
            plan.goal if plan else "",
            step.text or "",
            step.description or "",
            step.target_element_id or "",
            step.expected_change or "",
            " ".join(step.keys or []),
        ]
    ).lower()

    kws = sensitive_keywords or []
    for kw in kws:
        if kw and kw.lower() in text_blob:
            cat = RiskCategory.HIGH
            rules.append(f"sensitive_keyword:{kw}")
            break

    # Hotkeys that look like system/admin
    dangerous_keys = {"delete", "del", "f4", "alt+f4", "ctrl+w", "win+l", "ctrl+shift+esc"}
    key_blob = "+".join(k.lower() for k in (step.keys or []))
    if step.action in (ActionType.KEY, ActionType.HOTKEY, ActionType.TYPE) and any(
        d in key_blob or d in text_blob for d in dangerous_keys
    ):
        cat = max_category(cat, RiskCategory.HIGH)
        rules.append("dangerous_key_chord")

    level = max_risk(step.risk, category_to_level(cat))
    if cat == RiskCategory.HIGH:
        level = RiskLevel.HIGH
    return cat, level, rules


def classify_plan(
    plan: ActionPlan,
    *,
    sensitive_keywords: list[str] | None = None,
) -> tuple[RiskCategory, RiskLevel]:
    if not plan.steps:
        return RiskCategory.OBSERVE, RiskLevel.LOW
    cat = RiskCategory.OBSERVE
    level = RiskLevel.LOW
    for step in plan.steps:
        c, r, _ = classify_step(step, plan=plan, sensitive_keywords=sensitive_keywords)
        cat = max_category(cat, c)
        level = max_risk(level, r)
    return cat, level
