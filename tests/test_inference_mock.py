"""Phase E mock inference through shared validator."""

from __future__ import annotations

from capture.mock import MockCapture
from inference.mock import MockInference
from inference.prompts import PROMPT_VERSION
from ui_vision.mock import MockUIVision


def test_mock_observe_plan(config) -> None:
    frame = MockCapture(config).capture(trace_id="tr-e")
    vision = MockUIVision(config).recognize(frame, trace_id="tr-e")
    inf = MockInference(config)
    assert inf.health() is True
    resp = inf.observe(frame, vision, user_goal="点击搜索按钮", trace_id="tr-e")
    assert resp.ok is True
    assert resp.observation is not None
    assert resp.plan is not None
    assert resp.plan.steps
    assert resp.plan.steps[0].target_element_id
    assert "mock" in (resp.observation.model_name or "")


def test_mock_observe_only(config) -> None:
    frame = MockCapture(config).capture(trace_id="tr-e2")
    vision = MockUIVision(config).recognize(frame, trace_id="tr-e2")
    resp = MockInference(config).observe(
        frame, vision, user_goal="描述当前屏幕", trace_id="tr-e2", mode="observation"
    )
    assert resp.ok is True
    assert resp.observation is not None
    # observation mode / read-only → no executable steps
    assert resp.plan is None or resp.plan.steps == []


def test_create_inference_factory(config) -> None:
    from inference import create_inference
    from inference.mock import MockInference

    backend = create_inference(config)
    assert isinstance(backend, MockInference)
    meta = backend.ensure_ready()
    assert meta.get("healthy") is True
    assert meta.get("prompt_version") == PROMPT_VERSION


def test_cli_infer_prompts(capsys) -> None:
    from frontend.cli import main

    assert main(["infer", "prompts"]) == 0
    out = capsys.readouterr().out
    assert "prompt_version" in out
    assert PROMPT_VERSION in out


def test_cli_infer_once_mock(capsys, tmp_path) -> None:
    from frontend.cli import main

    out = tmp_path / "once.json"
    rc = main(
        [
            "infer",
            "once",
            "--backend",
            "mock",
            "--vision-backend",
            "mock",
            "--goal",
            "描述当前屏幕",
            "--json-out",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "observation" in text
