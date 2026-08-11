"""Phase F agent bench: 3 low-risk e2e tasks + pause-on-missing + state machine.

Usage (conda env dev):
  python benchmarks/phase_f/run_agent_bench.py
  python benchmarks/phase_f/run_agent_bench.py --json-out benchmarks/phase_f/results/phase_f_agent.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"


def _mock_cfg():
    from core.config import load_config
    from core.logging import setup_logging

    setup_logging(level="WARNING", json_logs=False, log_dir=None)
    cfg = load_config()
    cfg.capture.backend = "mock"
    cfg.ui_vision.backend = "mock"
    cfg.inference.backend = "mock"
    cfg.actuator.backend = "mock"
    cfg.actuator.dry_run = True
    cfg.agent.backend = "mock"
    cfg.agent.auto_confirm = True
    cfg.agent.post_action_settle_ms = 0
    return cfg


def run_tasks() -> dict[str, Any]:
    from agent.runtime import TaskAgent
    from core.models import TaskState

    cfg = _mock_cfg()
    goals = [
        ("click_search", "点击搜索按钮"),
        ("type_input", '输入 "hello baodou"'),
        ("observe_only", "描述当前屏幕"),
        ("click_ok", "点击确定"),
    ]
    cases: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for cid, goal in goals:
        agent = TaskAgent(cfg)
        try:
            r = agent.run(goal, execute=True, confirmed=True)
        finally:
            agent.close()
        ok = bool(r.ok and r.task.state == TaskState.COMPLETED)
        # Traceability: every action has target + verify
        traced = True
        for a, v in zip(r.actions, r.verifications, strict=False):
            if not a.success or not v.passed:
                traced = False
            if a.action.value not in ("wait", "none", "reidentify", "key", "hotkey") and not (
                a.resolved_element_id or a.resolved_point
            ):
                traced = False
        cases.append(
            {
                "id": cid,
                "goal": goal,
                "ok": ok and traced,
                "task_state": r.task.state.value,
                "steps_done": r.task.steps_done,
                "plan": r.plan.log_summary() if r.plan else None,
                "previews": [p.log_summary() for p in r.previews],
                "actions": [a.log_summary() for a in r.actions],
                "verifications": [v.log_summary() for v in r.verifications],
                "elapsed_ms": round(r.elapsed_ms, 2),
                "traced": traced,
            }
        )
    return {
        "suite": "low_risk_e2e",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        "passed": sum(1 for c in cases if c["ok"]),
        "total": len(cases),
        "cases": cases,
    }


def run_pause_on_missing() -> dict[str, Any]:
    from actuator.mock import MockActuator
    from agent.mock import MockAgent
    from agent.runtime import TaskAgent
    from capture.mock import MockCapture
    from core.models import (
        ActionPlan,
        ActionStep,
        ActionType,
        RiskLevel,
        TaskState,
        UIVisionResult,
    )
    from inference.mock import MockInference
    from ui_vision.mock import MockUIVision

    cfg = _mock_cfg()
    cfg.agent.max_recovery_attempts = 0
    cfg.agent.pause_on_target_missing = True

    class EmptyAfterPlan(MockUIVision):
        def __init__(self, config):
            super().__init__(config)
            self._n = 0

        def recognize(self, frame, *, trace_id="", image=None, roi=None, goal=None):
            self._n += 1
            if self._n == 1:
                return super().recognize(frame, trace_id=trace_id, image=image, roi=roi, goal=goal)
            return UIVisionResult(
                frame_id=frame.frame_id, trace_id=trace_id, elements=[], source="empty"
            )

    class PlanClick(MockAgent):
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
        vision=EmptyAfterPlan(cfg),
        inference=MockInference(cfg),
        planner=PlanClick(cfg),
        actuator=MockActuator(cfg),
    )
    try:
        r = agent.run("点击搜索", execute=True, confirmed=True)
    finally:
        agent.close()

    paused = r.task.state in {TaskState.PAUSED, TaskState.FAILED}
    no_blind = not (r.ok and r.task.state == TaskState.COMPLETED)
    return {
        "suite": "pause_on_target_missing",
        "ok": paused and no_blind,
        "task_state": r.task.state.value,
        "pause_reason": r.task.pause_reason,
        "error": r.error.to_dict() if r.error else None,
        "actions_count": len(r.actions),
        "no_blind_success": no_blind,
    }


def run_state_graph() -> dict[str, Any]:
    from agent.state import allowed_targets, can_transition
    from core.models import TaskState

    required = [
        (TaskState.IDLE, TaskState.OBSERVING),
        (TaskState.OBSERVING, TaskState.PLANNING),
        (TaskState.PLANNING, TaskState.AWAITING_CONFIRMATION),
        (TaskState.AWAITING_CONFIRMATION, TaskState.EXECUTING),
        (TaskState.EXECUTING, TaskState.VERIFYING),
        (TaskState.VERIFYING, TaskState.COMPLETED),
        (TaskState.EXECUTING, TaskState.PAUSED),
    ]
    ok_all = all(can_transition(a, b) for a, b in required)
    return {
        "suite": "state_machine",
        "ok": ok_all,
        "transitions": {s.value: allowed_targets(s) for s in TaskState},
        "checked": [(a.value, b.value) for a, b in required],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase F agent bench")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(RESULTS / "phase_f_agent.json"),
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "phase": "F",
        "title": "task agent & operation state machine",
        "tasks": run_tasks(),
        "pause_on_missing": run_pause_on_missing(),
        "state_machine": run_state_graph(),
    }
    suites_ok = [
        report["tasks"]["passed"] >= 3,
        report["pause_on_missing"]["ok"],
        report["state_machine"]["ok"],
    ]
    report["ok"] = all(suites_ok)
    report["summary"] = {
        "low_risk_passed": report["tasks"]["passed"],
        "low_risk_total": report["tasks"]["total"],
        "pause_ok": report["pause_on_missing"]["ok"],
        "state_ok": report["state_machine"]["ok"],
    }

    out = Path(args.json_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            report["summary"] | {"ok": report["ok"], "wrote": str(out)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
