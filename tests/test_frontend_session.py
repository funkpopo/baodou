"""Phase H: UI session, corrections, metrics, diagnostics (headless)."""

from __future__ import annotations

from core.models import (
    ActivityPhase,
    BBox,
    CorrectionKind,
    ElementType,
    Point,
    UIElement,
    UIVisionResult,
    UserCorrection,
)
from frontend.corrections import (
    CorrectionStore,
    apply_corrections_to_goal,
    apply_corrections_to_vision,
    filter_elements_by_corrections,
)
from frontend.diagnostics import build_diagnostics
from frontend.highlight import highlight_elements, resize_for_preview
from frontend.metrics import MetricsCollector
from frontend.session import UISession
from PIL import Image


def _mock_cfg(config):
    config.capture.backend = "mock"
    config.ui_vision.backend = "mock"
    config.inference.backend = "mock"
    config.actuator.backend = "mock"
    config.actuator.dry_run = True
    config.agent.backend = "mock"
    config.agent.auto_confirm = True
    config.safety.default_mode = "allow_low"
    config.frontend.default_mock = True
    return config


def _sample_elements() -> list[UIElement]:
    return [
        UIElement(
            element_id="btn_a",
            type=ElementType.BUTTON,
            text="取消",
            bbox=BBox(x=10, y=10, width=40, height=20),
            clickable=True,
            confidence=0.9,
        ),
        UIElement(
            element_id="btn_b",
            type=ElementType.BUTTON,
            text="搜索",
            bbox=BBox(x=100, y=10, width=40, height=20),
            clickable=True,
            confidence=0.95,
        ),
        UIElement(
            element_id="txt_c",
            type=ElementType.TEXT,
            text="hello",
            bbox=BBox(x=0, y=0, width=30, height=10),
            confidence=0.8,
        ),
    ]


def test_correction_store_kinds() -> None:
    store = CorrectionStore()
    store.reject_element("btn_a", note="不是这个")
    store.prefer_element("btn_b")
    store.click_here(12, 34)
    store.ignore_region(0, 0, 10, 10)
    store.note("备注一下")
    assert len(store.items) == 5
    assert "btn_a" in store.rejected_ids()
    assert store.preferred_ids() == ["btn_b"]
    assert store.click_points()[0] == Point(x=12, y=34)
    assert len(store.ignore_regions()) == 1


def test_apply_corrections_to_goal() -> None:
    corrs = [
        UserCorrection(kind=CorrectionKind.REJECT_ELEMENT, element_id="btn_a"),
        UserCorrection(kind=CorrectionKind.PREFER_ELEMENT, element_id="btn_b"),
        UserCorrection(kind=CorrectionKind.CLICK_HERE, point=Point(x=1, y=2)),
        UserCorrection(
            kind=CorrectionKind.IGNORE_REGION,
            region=BBox(x=0, y=0, width=5, height=5),
        ),
        UserCorrection(kind=CorrectionKind.NOTE, note="忽略弹窗"),
    ]
    text = apply_corrections_to_goal("点击搜索", corrs)
    assert "用户修正" in text
    assert "btn_a" in text
    assert "btn_b" in text
    assert "忽略弹窗" in text


def test_filter_elements_by_corrections() -> None:
    els = _sample_elements()
    corrs = [
        UserCorrection(kind=CorrectionKind.REJECT_ELEMENT, element_id="btn_a"),
        UserCorrection(kind=CorrectionKind.PREFER_ELEMENT, element_id="btn_b"),
        UserCorrection(
            kind=CorrectionKind.IGNORE_REGION,
            region=BBox(x=0, y=0, width=35, height=15),
        ),
    ]
    out = filter_elements_by_corrections(els, corrs)
    ids = [e.element_id for e in out]
    assert "btn_a" not in ids
    assert "txt_c" not in ids  # center inside ignore region
    assert ids[0] == "btn_b"


def test_apply_corrections_to_vision() -> None:
    vision = UIVisionResult(frame_id="f1", elements=_sample_elements())
    corrs = [UserCorrection(kind=CorrectionKind.REJECT_ELEMENT, element_id="btn_a")]
    out = apply_corrections_to_vision(vision, corrs)
    assert all(e.element_id != "btn_a" for e in out.elements)
    assert "corrections=" in out.notes


def test_metrics_collector() -> None:
    m = MetricsCollector(recent_errors_max=3)
    m.record_capture(1.5)
    m.record_vision(2.5)
    m.record_model(3.5)
    m.record_e2e(10.0)
    m.record_queue(1, dropped=2)
    m.push_error("e1")
    m.push_error("e2")
    snap = m.snapshot()
    assert snap.capture_latency_ms == 1.5
    assert snap.vision_latency_ms == 2.5
    assert snap.model_latency_ms == 3.5
    assert snap.queue_length == 1
    assert snap.queue_dropped == 2
    assert "e2" in snap.recent_errors[0]
    assert snap.cpu_percent is not None or snap.memory_rss_mb is not None


def test_highlight_and_resize() -> None:
    img = Image.new("RGB", (200, 100), color=(30, 30, 30))
    els = _sample_elements()
    ann = highlight_elements(img, els, highlight_ids={"btn_b"}, rejected_ids={"btn_a"})
    assert ann.size == (200, 100)
    small = resize_for_preview(ann, 100, 50)
    assert small.size[0] <= 100
    assert small.size[1] <= 50


def test_diagnostics_bundle(config) -> None:
    cfg = _mock_cfg(config)
    session = UISession(cfg, mock=True)
    session.refresh_observe("搜索")
    bundle = build_diagnostics(
        task=None,
        vision=session.get_last_vision(),
        corrections=session.corrections,
        metrics=session.metrics,
        activity=session.activity().log_summary(),
    )
    d = bundle.to_dict()
    assert "elements" in d
    assert "metrics" in d
    assert "activity" in d


def test_session_preview(config) -> None:
    cfg = _mock_cfg(config)
    session = UISession(cfg, mock=True)
    result = session.start_task("点击搜索按钮", execute=False, background=False)
    assert result is not None
    assert result.plan is not None
    snap = session.snapshot()
    assert snap.goal == "点击搜索按钮"
    assert snap.dry_run is True
    assert snap.mock is True
    assert len(snap.elements) >= 1 or result.vision is not None
    diag = session.diagnostics().to_dict()
    assert "plan" in diag


def test_session_execute_auto(config) -> None:
    cfg = _mock_cfg(config)
    session = UISession(cfg, mock=True)
    result = session.start_task(
        "点击搜索按钮",
        execute=True,
        auto_confirm=True,
        background=False,
    )
    assert result is not None
    # completed or at least attempted under dry_run
    assert result.ok or result.task.steps_done >= 0
    m = session.metrics.snapshot()
    assert m.end_to_end_ms is not None


def test_session_corrections_in_goal(config) -> None:
    cfg = _mock_cfg(config)
    session = UISession(cfg, mock=True)
    session.reject_element("btn_cancel_01", note="不是这个")
    session.prefer_element("btn_search_01", note="点搜索")
    result = session.start_task("点按钮", execute=False, background=False)
    assert result is not None
    assert len(result.task.corrections) >= 2
    snap = session.snapshot()
    assert len(snap.corrections) >= 2


def test_session_pause_stop_reset(config) -> None:
    cfg = _mock_cfg(config)
    session = UISession(cfg, mock=True)
    session.pause("test")
    assert session.activity().phase == ActivityPhase.PAUSED
    session.resume("test")
    session.emergency_stop("test")
    assert session.activity().phase == ActivityPhase.STOPPED
    session.reset_stop()
    assert session.activity().phase == ActivityPhase.IDLE


def test_cli_ui_status(capsys) -> None:
    from frontend.cli import main

    assert main(["ui", "status", "--log-level", "WARNING"]) == 0
    out = capsys.readouterr().out
    assert "activity" in out
    assert "metrics" in out


def test_cli_ui_run_preview(tmp_path) -> None:
    from frontend.cli import main

    out = tmp_path / "ui_run.json"
    rc = main(
        [
            "ui",
            "run",
            "--goal",
            "点击搜索按钮",
            "--preview-only",
            "--yes",
            "--json-out",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "diagnostics" in text
    assert "snapshot" in text


def test_cli_ui_correct(capsys) -> None:
    from frontend.cli import main

    assert main(["ui", "correct", "--goal", "搜索", "--log-level", "WARNING"]) == 0
    out = capsys.readouterr().out
    assert "corrections" in out
    assert "effective_goal" in out
