"""Mock inference: deterministic structured observation + plan without GPU."""

from __future__ import annotations

import json
import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import (
    ActionPlan,
    ActionStep,
    ActionType,
    InferenceRequest,
    InferenceResponse,
    RiskLevel,
    ScreenFrame,
    ScreenObservation,
    UIVisionResult,
)

from inference.base import InferenceBackend
from inference.prompts import PROMPT_VERSION
from inference.validate import validate_model_output

_log = get_logger("inference.mock")


class MockInference(InferenceBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def health(self) -> bool:
        return True

    def ensure_ready(self) -> dict:
        return {"healthy": True, "backend": "mock", "prompt_version": PROMPT_VERSION}

    def observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
        mode: str = "observe_plan",
        include_image: bool = True,
    ) -> InferenceResponse:
        get_global_token().check()
        t0 = time.perf_counter()
        req = InferenceRequest(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            task=user_goal,
            ui_summary=[
                {
                    "element_id": e.element_id,
                    "type": e.type.value,
                    "text": e.text,
                    "clickable": e.clickable,
                }
                for e in vision.elements
            ],
            has_image=bool(include_image and (frame.image_b64 or frame.image_path)),
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
        )
        log_event(
            _log,
            "inference.request",
            request_id=req.request_id,
            trace_id=trace_id,
            frame_id=frame.frame_id,
            task_preview=user_goal[:80],
            element_count=len(vision.elements),
            mode=mode,
            prompt_version=PROMPT_VERSION,
        )

        time.sleep(0.02)
        clickable = [e for e in vision.elements if e.clickable]
        goal_l = user_goal.lower()
        read_only = any(
            h in goal_l for h in ("读取", "描述", "看看", "summarize", "describe", "read")
        ) and not any(h in goal_l for h in ("点击", "输入", "click", "type", "打开"))

        # Build a payload that goes through the same validator as HTTP.
        steps = []
        if not read_only and mode != "observation":
            target = next((e for e in vision.elements if e.clickable), None)
            if target is not None and any(h in goal_l for h in ("点击", "click", "打开", "搜索")):
                steps.append(
                    {
                        "action": "click",
                        "target_element_id": target.element_id,
                        "target_point": None,
                        "text": None,
                        "keys": [],
                        "risk": "low",
                        "requires_confirmation": True,
                        "preconditions": ["element.visible == true"],
                        "expected_change": f"interact with {target.element_id}",
                    }
                )

        raw_obj = {
            "kind": "observation" if (read_only or mode == "observation") else "observe_plan",
            "observation": (
                f"Mock screen {frame.width}x{frame.height}: "
                f"{len(vision.elements)} elements, {len(clickable)} clickable. "
                f"User goal: {user_goal}"
            ),
            "confidence": 0.9,
            "notes": f"MockInference prompt={PROMPT_VERSION}",
            "ui_candidates": [
                {
                    "element_id": e.element_id,
                    "type": e.type.value,
                    "text": e.text,
                    "confidence": e.confidence,
                }
                for e in vision.elements[:6]
            ],
            "plan": {
                "goal": user_goal,
                "steps": steps,
                "stop_if": ["target_missing", "window_changed"],
            },
            "needs_user_confirm": True,
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        v = validate_model_output(
            raw_obj,
            frame=frame,
            vision=vision,
            user_goal=user_goal,
            mode="observation" if read_only or mode == "observation" else "observe_plan",
            trace_id=trace_id,
            model_name="mock-qwen3.5-2b",
            latency_ms=latency_ms,
        )
        if not v.ok or v.observation is None:
            # Fallback deterministic (should not fail on mock data)
            observation = ScreenObservation(
                trace_id=trace_id,
                frame_id=frame.frame_id,
                observation=raw_obj["observation"],
                ui_elements=vision.elements[:6],
                notes="mock fallback",
                confidence=0.9,
                model_name="mock-qwen3.5-2b",
                latency_ms=latency_ms,
            )
            plan = None
            if steps:
                plan = ActionPlan(
                    trace_id=trace_id,
                    goal=user_goal,
                    steps=[
                        ActionStep(
                            action=ActionType.CLICK,
                            target_element_id=steps[0]["target_element_id"],
                            risk=RiskLevel.LOW,
                            requires_confirmation=True,
                        )
                    ],
                )
            resp = InferenceResponse(
                request_id=req.request_id,
                trace_id=trace_id,
                ok=True,
                observation=observation,
                plan=plan,
                raw_text=json.dumps(raw_obj, ensure_ascii=False),
                latency_ms=latency_ms,
            )
        else:
            resp = InferenceResponse(
                request_id=req.request_id,
                trace_id=trace_id,
                ok=True,
                observation=v.observation,
                plan=v.plan,
                raw_text=json.dumps(raw_obj, ensure_ascii=False),
                latency_ms=latency_ms,
            )
        log_event(_log, "inference.response", **resp.log_summary())
        return resp
