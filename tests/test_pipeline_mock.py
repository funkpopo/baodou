"""End-to-end mock pipeline: full chain + trace correlation."""

from __future__ import annotations

from core.cancel import get_global_token
from core.errors import ErrorCode
from core.models import ActionPlan, ActionStep, ActionType, PipelineEventKind, RiskLevel, TaskState
from core.pipeline import MockPipeline
from safety.policy import SafetyPolicy


def test_pipeline_click_chain(config) -> None:
    pipe = MockPipeline(config)
    result = pipe.run("点击搜索按钮")
    kinds = [e.kind for e in result.events]
    # Full chain present
    assert PipelineEventKind.FRAME in kinds
    assert PipelineEventKind.VISION in kinds
    assert PipelineEventKind.INFERENCE in kinds
    assert PipelineEventKind.PLAN in kinds
    assert PipelineEventKind.SAFETY in kinds
    assert PipelineEventKind.ACTION in kinds
    assert PipelineEventKind.VERIFICATION in kinds
    # Single trace_id everywhere
    assert result.trace_id.startswith("tr-")
    assert all(e.trace_id == result.trace_id for e in result.events)
    assert result.frame is not None and result.frame.trace_id == result.trace_id
    assert result.vision is not None
    assert result.plan is not None
    assert result.plan.steps
    assert result.plan.steps[0].target_element_id == "btn_search_01"
    assert result.action is not None and result.action.dry_run is True
    assert result.verification is not None and result.verification.passed
    assert result.ok is True or result.task.state in {TaskState.COMPLETED, TaskState.PAUSED}


def test_pipeline_observe_only(config) -> None:
    pipe = MockPipeline(config)
    result = pipe.run("描述当前屏幕")
    assert result.ok is True
    assert result.plan is not None
    assert result.plan.steps == []
    assert result.task.state == TaskState.COMPLETED
    kinds = [e.kind for e in result.events]
    assert PipelineEventKind.DONE in kinds
    assert PipelineEventKind.ACTION not in kinds


def test_pipeline_high_risk_blocked(config) -> None:
    pipe = MockPipeline(config)
    result = pipe.run("删除重要文件并支付")
    assert result.ok is False
    assert result.safety is not None
    assert result.safety.allowed is False
    assert result.error is not None
    assert result.error.code in {ErrorCode.RISK_BLOCKED, ErrorCode.PERMISSION_DENIED}
    kinds = [e.kind for e in result.events]
    assert PipelineEventKind.ACTION not in kinds


def test_pipeline_cancel_midway(config) -> None:
    pipe = MockPipeline(config)
    tok = get_global_token()
    # Cancel before run
    tok.cancel("test")
    result = pipe.run("点击搜索按钮")
    assert result.task.state == TaskState.CANCELLED
    assert result.error is not None
    assert result.error.code == ErrorCode.CANCELLED


def test_modules_independent_with_mock(config) -> None:
    """Each module runs alone on mock data (Phase B acceptance)."""
    from actuator.mock import MockActuator
    from agent.mock import MockAgent
    from capture.mock import MockCapture
    from inference.mock import MockInference
    from ui_vision.mock import MockUIVision

    frame = MockCapture(config).capture(trace_id="tr-unit")
    vision = MockUIVision(config).recognize(frame, trace_id="tr-unit")
    assert len(vision.elements) >= 3
    inf = MockInference(config).observe(frame, vision, user_goal="test", trace_id="tr-unit")
    assert inf.ok and inf.observation is not None
    plan = MockAgent(config).plan("点击确定", vision, inf.observation, trace_id="tr-unit")
    assert plan.steps
    step = plan.steps[0]
    decision = SafetyPolicy(config).evaluate(step, plan)
    assert decision.allowed or decision.risk == RiskLevel.HIGH
    act = MockActuator(config)
    action = act.execute(step, vision, trace_id="tr-unit")
    assert action.success and action.dry_run
    ver = act.verify(step, action, trace_id="tr-unit")
    assert ver.passed


def test_safety_sensitive_keyword(config) -> None:
    policy = SafetyPolicy(config)
    plan = ActionPlan(goal="转账给陌生人", steps=[])
    step = ActionStep(action=ActionType.CLICK, target_element_id="btn_ok_01", risk=RiskLevel.LOW)
    decision = policy.evaluate(step, plan)
    assert decision.allowed is False
    assert decision.risk == RiskLevel.HIGH
