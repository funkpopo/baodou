"""Protocol model unit tests."""

from __future__ import annotations

import pytest
from core.models import (
    PROTOCOL_VERSION,
    ActionPlan,
    ActionStep,
    ActionType,
    BBox,
    ElementType,
    RiskLevel,
    ScreenFrame,
    UIElement,
)
from pydantic import ValidationError


def test_protocol_version_set() -> None:
    assert PROTOCOL_VERSION.count(".") == 2


def test_bbox_center_and_contains() -> None:
    box = BBox(x=10, y=20, width=100, height=40)
    assert box.center() == (60, 40)
    assert box.contains(60, 40)
    assert not box.contains(5, 20)


def test_bbox_rejects_negative_size() -> None:
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, width=-1, height=10)


def test_ui_element_center_property() -> None:
    el = UIElement(
        element_id="btn_1",
        type=ElementType.BUTTON,
        bbox=BBox(x=0, y=0, width=10, height=10),
        confidence=0.9,
        clickable=True,
        source=["mock"],
    )
    assert el.center.x == 5 and el.center.y == 5


def test_action_plan_roundtrip() -> None:
    plan = ActionPlan(
        goal="点击搜索",
        steps=[
            ActionStep(
                action=ActionType.CLICK,
                target_element_id="btn_search_01",
                risk=RiskLevel.LOW,
                requires_confirmation=True,
            )
        ],
        risk_max=RiskLevel.LOW,
    )
    data = plan.model_dump()
    restored = ActionPlan.model_validate(data)
    assert restored.goal == plan.goal
    assert restored.steps[0].target_element_id == "btn_search_01"
    assert restored.protocol_version == PROTOCOL_VERSION


def test_screen_frame_log_summary_omits_pixels() -> None:
    frame = ScreenFrame(width=1280, height=720, image_b64="AAAA")
    summary = frame.log_summary()
    assert "image_b64" not in summary
    assert summary["has_image"] is True
