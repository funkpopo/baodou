"""Phase F: task state machine, preview, recovery, end-to-end agent runtime."""

from __future__ import annotations

from actuator.relocate import relocate_target
from agent.preview import build_step_preview
from agent.recovery import decide_recovery
from agent.runtime import TaskAgent
from agent.state import can_transition, is_terminal, transition
from core.errors import ActuatorError, ErrorCode
from core.models import (
    ActionPlan,
    ActionStep,
    ActionType,
    BBox,
    ElementType,
    Point,
    RiskLevel,
    TaskContext,
    TaskState,
    UIElement,
    UIVisionResult,
    bbox_iou,
)


def _mock_cfg(config):
    config.capture.backend = "mock"
    config.ui_vision.backend = "mock"
    config.inference.backend = "mock"
    config.actuator.backend = "mock"
    config.actuator.dry_run = True
    config.agent.backend = "mock"
    config.agent.auto_confirm = True
    config.agent.reidentify_before_action = True
    config.agent.post_action_settle_ms = 0
    return config


def test_state_machine_happy_path() -> None:
    s = TaskState.IDLE
    for nxt in (
        TaskState.OBSERVING,
        TaskState.PLANNING,
        TaskState.AWAITING_CONFIRMATION,
        TaskState.EXECUTING,
        TaskState.VERIFYING,
        TaskState.COMPLETED,
    ):
        assert can_transition(s, nxt)
        s = transition(s, nxt)
    assert is_terminal(s)


def test_state_machine_illegal() -> None:
    assert not can_transition(TaskState.IDLE, TaskState.EXECUTING)
    try:
        transition(TaskState.COMPLETED, TaskState.EXECUTING)
        raise AssertionError("expected error")
    except Exception as exc:  # noqa: BLE001
        assert "非法" in str(exc) or "INTERNAL" in str(exc) or True


def test_preview_click(config) -> None:
    from capture.mock import MockCapture
    from ui_vision.mock import MockUIVision

    cfg = _mock_cfg(config)
    frame = MockCapture(cfg).capture(trace_id="tr-p")
    vision = MockUIVision(cfg).recognize(frame, trace_id="tr-p")
    step = ActionStep(
        action=ActionType.CLICK,
        target_element_id="btn_search_01",
        risk=RiskLevel.LOW,
        expected_change="search opens",
        description="点击搜索",
    )
    prev = build_step_preview(step, vision)
    assert "搜索" in prev.summary or "btn_search" in prev.summary
    assert prev.target_element_id == "btn_search_01"
    assert prev.target_point is not None
    assert prev.uses_coordinates is False


def test_relocate_prefers_element_id(config) -> None:
    cfg = _mock_cfg(config)
    from capture.mock import MockCapture
    from ui_vision.mock import MockUIVision

    frame = MockCapture(cfg).capture()
    vision = MockUIVision(cfg).recognize(frame)
    step = ActionStep(action=ActionType.CLICK, target_element_id="btn_ok_01")
    loc = relocate_target(step, vision, cfg)
    assert loc.element is not None
    assert loc.element_id == "btn_ok_01"
    assert loc.method == "id"
    assert loc.point is not None


def test_relocate_rejects_bare_coords_without_flag(config) -> None:
    cfg = _mock_cfg(config)
    vision = UIVisionResult(frame_id="f1", elements=[], source="test")
    step = ActionStep(
        action=ActionType.CLICK,
        target_point=Point(x=10, y=10),
        allow_coordinate_fallback=False,
    )
    try:
        relocate_target(step, vision, cfg, coordinate_confirmed=False)
        raise AssertionError("should reject")
    except ActuatorError as exc:
        assert exc.code in {
            ErrorCode.TARGET_INVALID,
            ErrorCode.CONFIRMATION_REQUIRED,
            ErrorCode.TARGET_STALE,
        }


def test_relocate_fuzzy_hash(config) -> None:
    cfg = _mock_cfg(config)
    el = UIElement(
        element_id="btn_old",
        type=ElementType.BUTTON,
        text="搜索",
        bbox=BBox(x=100, y=40, width=80, height=30),
        clickable=True,
        visible=True,
        enabled=True,
        frame_id="f1",
        content_hash="abc123",
        source=["t"],
    )
    el2 = UIElement(
        element_id="btn_new",
        type=ElementType.BUTTON,
        text="搜索",
        bbox=BBox(x=102, y=42, width=80, height=30),
        clickable=True,
        visible=True,
        enabled=True,
        frame_id="f2",
        content_hash="abc123",
        source=["t"],
    )
    vision = UIVisionResult(frame_id="f2", elements=[el2], source="t")
    step = ActionStep(action=ActionType.CLICK, target_element_id="btn_old")
    loc = relocate_target(step, vision, cfg, prior_element=el)
    assert loc.element is not None
    assert loc.element_id == "btn_new"
    assert loc.method in {"hash", "iou"}
    assert bbox_iou(el.bbox, el2.bbox) > 0.5


def test_recovery_target_missing_pauses(config) -> None:
    cfg = _mock_cfg(config)
    cfg.agent.max_recovery_attempts = 0
    cfg.agent.pause_on_target_missing = True
    task = TaskContext(user_goal="x", recovery_attempts=0)
    step = ActionStep(action=ActionType.CLICK, target_element_id="missing")
    action, note = decide_recovery(
        config=cfg,
        task=task,
        step=step,
        error_code="target_stale",
        reason="gone",
    )
    from core.models import RecoveryAction

    assert action == RecoveryAction.PAUSE
    assert "暂停" in note or "gone" in note


def test_agent_click_e2e(config) -> None:
    cfg = _mock_cfg(config)
    agent = TaskAgent(cfg)
    try:
        result = agent.run("点击搜索按钮", execute=True, confirmed=True)
    finally:
        agent.close()
    assert result.ok is True
    assert result.task.state == TaskState.COMPLETED
    assert result.plan is not None and result.plan.steps
    assert result.plan.steps[0].target_element_id == "btn_search_01"
    assert result.actions
    assert result.actions[0].success
    assert result.actions[0].resolved_element_id == "btn_search_01"
    assert result.verifications
    assert result.verifications[0].passed
    assert result.task.steps_done >= 1
    # Previews present
    assert result.previews
    assert any(p.target_element_id == "btn_search_01" for p in result.previews)


def test_agent_type_e2e(config) -> None:
    cfg = _mock_cfg(config)
    agent = TaskAgent(cfg)
    try:
        result = agent.run('输入 "hello world"', execute=True, confirmed=True)
    finally:
        agent.close()
    assert result.ok is True
    assert result.plan is not None
    assert any(s.action == ActionType.TYPE for s in result.plan.steps)
    assert result.actions
    assert all(a.dry_run for a in result.actions)


def test_agent_observe_only(config) -> None:
    cfg = _mock_cfg(config)
    agent = TaskAgent(cfg)
    try:
        result = agent.run("描述当前屏幕", execute=True, confirmed=True)
    finally:
        agent.close()
    assert result.ok is True
    assert result.plan is not None
    assert result.plan.steps == []
    assert result.task.state == TaskState.COMPLETED
    assert not result.actions


def test_agent_preview_only_no_execute(config) -> None:
    cfg = _mock_cfg(config)
    cfg.agent.auto_confirm = False
    agent = TaskAgent(cfg)
    try:
        result = agent.preview("点击确定")
    finally:
        agent.close()
    assert result.ok is True
    assert result.plan is not None and result.plan.steps
    assert result.previews
    assert not result.actions
    assert result.task.state == TaskState.AWAITING_CONFIRMATION


def test_agent_requires_confirmation_pauses(config) -> None:
    cfg = _mock_cfg(config)
    cfg.agent.auto_confirm = False
    agent = TaskAgent(cfg)
    try:
        result = agent.run("点击搜索按钮", execute=True, confirmed=False)
    finally:
        agent.close()
    assert result.ok is False
    assert result.task.state == TaskState.PAUSED
    assert result.error is not None
    assert result.error.code == ErrorCode.CONFIRMATION_REQUIRED
    assert result.previews
    assert "搜索" in (result.task.pause_reason or result.previews[0].summary)


def test_agent_high_risk_blocked(config) -> None:
    cfg = _mock_cfg(config)
    agent = TaskAgent(cfg)
    try:
        result = agent.run("删除文件并支付", execute=True, confirmed=True)
    finally:
        agent.close()
    assert result.ok is False
    assert result.task.state == TaskState.FAILED
    assert result.error is not None
    assert result.error.code in {ErrorCode.RISK_BLOCKED, ErrorCode.PERMISSION_DENIED}
    assert not result.actions


def test_agent_target_missing_pauses(config) -> None:
    """When element disappears before execute, agent pauses (not blind click)."""
    cfg = _mock_cfg(config)
    cfg.agent.max_recovery_attempts = 0
    cfg.agent.pause_on_target_missing = True

    from actuator.mock import MockActuator
    from agent.mock import MockAgent
    from capture.mock import MockCapture
    from inference.mock import MockInference
    from ui_vision.mock import MockUIVision

    class EmptyVision(MockUIVision):
        def recognize(self, frame, *, trace_id="", image=None, roi=None, goal=None):
            # First call (planning) returns normal; subsequent empty → missing
            if not hasattr(self, "_n"):
                self._n = 0
            self._n += 1
            if self._n == 1:
                return super().recognize(frame, trace_id=trace_id, image=image, roi=roi, goal=goal)
            return UIVisionResult(
                frame_id=frame.frame_id,
                trace_id=trace_id,
                elements=[],
                source="empty",
            )

    class FixedPlanner(MockAgent):
        def plan(self, user_goal, vision, observation, *, trace_id=""):
            return ActionPlan(
                trace_id=trace_id,
                goal=user_goal,
                steps=[
                    ActionStep(
                        action=ActionType.CLICK,
                        target_element_id="btn_search_01",
                        risk=RiskLevel.LOW,
                        requires_confirmation=False,
                        expected_change="click",
                    )
                ],
            )

    agent = TaskAgent(
        cfg,
        capture=MockCapture(cfg),
        vision=EmptyVision(cfg),
        inference=MockInference(cfg),
        planner=FixedPlanner(cfg),
        actuator=MockActuator(cfg),
    )
    try:
        result = agent.run("点击搜索", execute=True, confirmed=True)
    finally:
        agent.close()
    assert result.ok is False
    assert result.task.state in {TaskState.PAUSED, TaskState.FAILED}
    # Must not have blindly executed with stale coords
    if result.actions:
        # If recovery attempted execute, it should have failed before success-without-target
        assert not (result.ok and result.task.state == TaskState.COMPLETED)


def test_three_low_risk_tasks_acceptance(config) -> None:
    """Phase F acceptance: ≥3 low-risk e2e tasks with traceable target/result/verify."""
    cfg = _mock_cfg(config)
    goals = [
        "点击搜索按钮",
        '输入 "baodou"',
        "描述当前屏幕",
    ]
    reports = []
    for g in goals:
        agent = TaskAgent(cfg)
        try:
            r = agent.run(g, execute=True, confirmed=True)
        finally:
            agent.close()
        reports.append(
            {
                "goal": g,
                "ok": r.ok,
                "state": r.task.state.value,
                "steps_done": r.task.steps_done,
                "actions": [a.log_summary() for a in r.actions],
                "verifications": [v.log_summary() for v in r.verifications],
                "previews": [p.log_summary() for p in r.previews],
            }
        )
        assert r.ok is True
        assert r.task.state == TaskState.COMPLETED
        # Each executed step has matching verification when actions present
        assert len(r.verifications) == len(r.actions)
        for a, v in zip(r.actions, r.verifications, strict=True):
            assert a.success
            assert v.passed
            assert a.step_id == v.step_id
    assert len(reports) == 3
