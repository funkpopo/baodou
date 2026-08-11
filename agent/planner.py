"""Planning backends: mock rules + optional inference plan (Phase F)."""

from __future__ import annotations

from core.cancel import get_global_token
from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import ActionPlan, ScreenFrame, ScreenObservation, UIVisionResult

from agent.base import AgentBackend
from agent.mock import MockAgent

_log = get_logger("agent.planner")


class InferencePlanner(AgentBackend):
    """Use model-produced plan when present; otherwise fall back to MockAgent."""

    def __init__(self, config: AppConfig, inference: object | None = None) -> None:
        self.config = config
        self.inference = inference
        self._mock = MockAgent(config)

    def plan(
        self,
        user_goal: str,
        vision: UIVisionResult,
        observation: ScreenObservation,
        *,
        trace_id: str = "",
        frame: ScreenFrame | None = None,
    ) -> ActionPlan:
        get_global_token().check()

        if self.inference is not None and frame is not None:
            try:
                observe = getattr(self.inference, "observe", None)
                if callable(observe):
                    resp = observe(
                        frame,
                        vision,
                        user_goal=user_goal,
                        trace_id=trace_id,
                        mode="observe_plan",
                        include_image=True,
                    )
                    plan = getattr(resp, "plan", None) if resp is not None else None
                    if plan is not None and getattr(resp, "ok", False):
                        plan.trace_id = plan.trace_id or trace_id
                        if len(plan.steps) > self.config.agent.max_steps:
                            plan.steps = plan.steps[: self.config.agent.max_steps]
                        log_event(
                            _log,
                            "agent.plan",
                            **plan.log_summary(),
                            source="inference",
                        )
                        return plan
            except Exception as exc:  # noqa: BLE001
                log_event(
                    _log,
                    "agent.plan_inference_failed",
                    error=str(exc),
                    fallback=self.config.agent.fallback_to_mock_plan,
                )
                if not self.config.agent.fallback_to_mock_plan:
                    raise

        plan = self._mock.plan(user_goal, vision, observation, trace_id=trace_id)
        log_event(_log, "agent.plan", **plan.log_summary(), source="mock_fallback")
        return plan


def create_planner(config: AppConfig, inference: object | None = None) -> AgentBackend:
    backend = (config.agent.backend or "mock").lower()
    if backend == "inference":
        return InferencePlanner(config, inference=inference)
    return MockAgent(config)
