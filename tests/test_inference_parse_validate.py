"""Phase E: JSON parse, schema validate, whitelist, coords, degrade."""

from __future__ import annotations

from core.errors import ErrorCode
from core.models import (
    BBox,
    CaptureMode,
    ElementType,
    ScreenFrame,
    UIElement,
    UIVisionResult,
)
from inference.degrade import degraded_response
from inference.parse import extract_json, is_json_complete, try_repair_json
from inference.prompts import ALLOWED_ACTIONS, PROMPT_VERSION, build_user_message, get_prompt
from inference.schema import parse_model_payload
from inference.validate import validate_model_output


def _frame(**kwargs) -> ScreenFrame:
    base = dict(
        width=1280,
        height=720,
        origin_x=0,
        origin_y=0,
        physical_width=1920,
        physical_height=1080,
        scale_x=1.5,
        scale_y=1.5,
        mode=CaptureMode.PRIMARY,
    )
    base.update(kwargs)
    return ScreenFrame(**base)


def _vision(frame: ScreenFrame) -> UIVisionResult:
    els = [
        UIElement(
            element_id="btn_search_01",
            type=ElementType.BUTTON,
            text="搜索",
            bbox=BBox(x=100, y=40, width=80, height=32),
            confidence=0.95,
            clickable=True,
            visible=True,
            enabled=True,
            source=["mock"],
            frame_id=frame.frame_id,
        ),
        UIElement(
            element_id="inp_q_01",
            type=ElementType.INPUT,
            text="",
            bbox=BBox(x=200, y=40, width=200, height=32),
            confidence=0.9,
            editable=True,
            visible=True,
            enabled=True,
            source=["mock"],
            frame_id=frame.frame_id,
        ),
    ]
    return UIVisionResult(frame_id=frame.frame_id, elements=els, source="mock")


def test_extract_json_plain() -> None:
    raw = '{"observation": "hi", "confidence": 0.5}'
    obj, err = extract_json(raw)
    assert err is None
    assert isinstance(obj, dict)
    assert obj["observation"] == "hi"


def test_extract_json_fenced_and_thinking() -> None:
    raw = '<think>secret</think>\n```json\n{"a": 1}\n```\n'
    obj, err = extract_json(raw)
    assert err is None
    assert obj == {"a": 1}


def test_extract_json_repair_truncated() -> None:
    raw = '{"observation": "partial", "confidence": 0.7, "notes": "x"'
    repaired = try_repair_json(raw)
    assert isinstance(repaired, dict)
    assert repaired.get("observation") == "partial"


def test_is_json_complete() -> None:
    assert is_json_complete('{"a": 1}') is True
    assert is_json_complete('{"a": ') is False
    assert is_json_complete("") is False


def test_prompt_version_stable() -> None:
    assert PROMPT_VERSION
    p = get_prompt("observe_plan")
    assert "JSON" in p.system or "json" in p.system.lower()
    msg = build_user_message(
        user_goal="点击搜索",
        frame_id="frame-1",
        width=1280,
        height=720,
        ui_summary=[{"element_id": "btn_search_01", "type": "button", "text": "搜索"}],
    )
    assert "btn_search_01" in msg
    assert "点击搜索" in msg


def test_validate_good_plan() -> None:
    frame = _frame()
    vision = _vision(frame)
    raw = {
        "kind": "observe_plan",
        "observation": "search button visible",
        "confidence": 0.8,
        "notes": "",
        "ui_candidates": [
            {"element_id": "btn_search_01", "type": "button", "text": "搜索", "confidence": 0.9}
        ],
        "plan": {
            "goal": "点击搜索",
            "steps": [
                {
                    "action": "click",
                    "target_element_id": "btn_search_01",
                    "risk": "low",
                    "requires_confirmation": True,
                    "expected_change": "results",
                }
            ],
            "stop_if": ["target_missing"],
        },
        "needs_user_confirm": True,
    }
    v = validate_model_output(
        raw, frame=frame, vision=vision, user_goal="点击搜索", mode="observe_plan"
    )
    assert v.ok is True
    assert v.plan is not None
    assert v.plan.steps[0].target_element_id == "btn_search_01"
    assert v.plan.steps[0].requires_confirmation is True


def test_reject_unknown_action() -> None:
    frame = _frame()
    vision = _vision(frame)
    raw = {
        "kind": "observe_plan",
        "observation": "x",
        "confidence": 0.5,
        "notes": "",
        "ui_candidates": [],
        "plan": {
            "goal": "hack",
            "steps": [
                {
                    "action": "shell_exec",
                    "target_element_id": "btn_search_01",
                    "risk": "low",
                    "requires_confirmation": False,
                }
            ],
        },
        "needs_user_confirm": False,
    }
    v = validate_model_output(raw, frame=frame, vision=vision, user_goal="hack")
    assert v.ok is False
    assert any(i.code == "action_whitelist" for i in v.issues)
    assert v.plan is None  # must not leak
    assert "shell_exec" not in ALLOWED_ACTIONS


def test_reject_unknown_element_id() -> None:
    frame = _frame()
    vision = _vision(frame)
    raw = {
        "kind": "observe_plan",
        "observation": "x",
        "confidence": 0.5,
        "notes": "",
        "ui_candidates": [],
        "plan": {
            "goal": "click",
            "steps": [
                {
                    "action": "click",
                    "target_element_id": "btn_does_not_exist",
                    "risk": "low",
                    "requires_confirmation": True,
                }
            ],
        },
        "needs_user_confirm": True,
    }
    v = validate_model_output(raw, frame=frame, vision=vision, user_goal="click")
    assert v.ok is False
    assert any(i.code == "unknown_element" for i in v.issues)


def test_reject_coord_out_of_bounds() -> None:
    frame = _frame()
    vision = _vision(frame)
    raw = {
        "kind": "observe_plan",
        "observation": "x",
        "confidence": 0.5,
        "notes": "",
        "ui_candidates": [],
        "plan": {
            "goal": "click",
            "steps": [
                {
                    "action": "click",
                    "target_element_id": None,
                    "target_point": {"x": 99999, "y": 99999},
                    "risk": "low",
                    "requires_confirmation": True,
                }
            ],
        },
        "needs_user_confirm": True,
    }
    v = validate_model_output(raw, frame=frame, vision=vision, user_goal="click")
    assert v.ok is False
    assert any(i.code == "coord_oob" for i in v.issues)


def test_reject_truncated() -> None:
    frame = _frame()
    vision = _vision(frame)
    raw = {
        "kind": "observation",
        "observation": "cut off",
        "confidence": 0.5,
        "notes": "",
        "ui_candidates": [],
    }
    v = validate_model_output(
        raw, frame=frame, vision=vision, user_goal="描述", mode="observation", raw_truncated=True
    )
    assert v.ok is False
    assert any(i.code == "truncated" for i in v.issues)


def test_phase_a_suggested_action_shape() -> None:
    """Normalize older free-form shapes into protocol."""
    data = {
        "observation": "desktop",
        "confidence": 0.6,
        "notes": "ok",
        "ui_elements": [{"id": "e1", "type": "button", "text": "OK", "confidence": 0.5}],
        "suggested_action": {
            "action": "none",
            "target_element_id": None,
            "risk": "low",
            "requires_confirmation": True,
        },
    }
    payload = parse_model_payload(data, mode="observe_plan")
    assert payload.observation == "desktop"
    assert payload.plan.steps  # lifted suggested_action


def test_degraded_response_no_steps(config) -> None:
    frame = _frame()
    vision = _vision(frame)
    resp = degraded_response(
        request_id="llm-x",
        trace_id="tr-x",
        frame=frame,
        vision=vision,
        user_goal="点击",
        reason="timeout",
        error_code=ErrorCode.INFERENCE_TIMEOUT.value,
    )
    assert resp.ok is False
    assert resp.plan is not None
    assert resp.plan.steps == []
    assert resp.observation is not None
    assert resp.error_code == ErrorCode.INFERENCE_TIMEOUT.value
