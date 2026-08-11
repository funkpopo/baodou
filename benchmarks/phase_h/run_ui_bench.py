"""Phase H UI / observability bench (headless session + corrections + metrics).

Usage (conda env dev):
  python benchmarks/phase_h/run_ui_bench.py
  python benchmarks/phase_h/run_ui_bench.py --json-out benchmarks/phase_h/results/phase_h_ui.json
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


def _cfg():
    from core.config import load_config
    from core.logging import setup_logging
    from safety.control import reset_safety_control

    setup_logging(level="WARNING", json_logs=False, log_dir=None)
    reset_safety_control()
    cfg = load_config()
    cfg.capture.backend = "mock"
    cfg.ui_vision.backend = "mock"
    cfg.inference.backend = "mock"
    cfg.actuator.backend = "mock"
    cfg.actuator.dry_run = True
    cfg.agent.backend = "mock"
    cfg.agent.auto_confirm = True
    cfg.safety.default_mode = "allow_low"
    cfg.safety.audit_enabled = True
    cfg.safety.audit_dir = str(ART / "audit")
    cfg.frontend.default_mock = True
    return cfg


def _case(cid: str, ok: bool, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "ok": ok, **extra}


def run_suite() -> dict[str, Any]:
    from frontend.corrections import (
        apply_corrections_to_goal,
        apply_corrections_to_vision,
        filter_elements_by_corrections,
    )
    from frontend.highlight import highlight_elements, resize_for_preview
    from frontend.metrics import MetricsCollector
    from frontend.session import UISession
    from PIL import Image

    cfg = _cfg()
    cases: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # 1. Headless session preview
    session = UISession(cfg, mock=True)
    result = session.start_task(
        "点击搜索按钮",
        execute=False,
        auto_confirm=False,
        background=False,
    )
    assert result is not None
    snap = session.snapshot()
    cases.append(
        _case(
            "preview_session",
            bool(
                result.ok
                and result.plan is not None
                and snap.activity.phase.value
                in ("idle", "awaiting_confirm", "recognizing", "inferring")
            ),
            state=result.task.state.value,
            plan_steps=len(result.plan.steps) if result.plan else 0,
            elements=len(snap.elements),
            activity=snap.activity.log_summary(),
        )
    )

    # 2. Execute with auto-confirm (dry_run)
    session2 = UISession(cfg, mock=True)
    result2 = session2.start_task(
        "点击搜索按钮",
        execute=True,
        auto_confirm=True,
        background=False,
    )
    assert result2 is not None
    cases.append(
        _case(
            "execute_auto_confirm",
            bool(result2.ok or result2.task.steps_done > 0),
            ok_flag=result2.ok,
            steps_done=result2.task.steps_done,
            state=result2.task.state.value,
            error=result2.error.message if result2.error else None,
        )
    )

    # 3. User corrections
    session3 = UISession(cfg, mock=True)
    obs = session3.refresh_observe("点击搜索")
    vision = session3.get_last_vision()
    before = len(vision.elements) if vision else 0
    reject_id = vision.elements[0].element_id if vision and vision.elements else "btn_x"
    prefer_id = vision.elements[1].element_id if vision and len(vision.elements) > 1 else reject_id
    session3.reject_element(reject_id, note="不是这个按钮")
    session3.prefer_element(prefer_id, note="点击这个")
    session3.click_here(100, 200, note="点这里")
    session3.ignore_region(0, 0, 50, 50, note="忽略该区域")
    eff = apply_corrections_to_goal("点击搜索按钮", session3.corrections.to_list())
    filtered = (
        apply_corrections_to_vision(vision, session3.corrections.to_list()) if vision else None
    )
    after = len(filtered.elements) if filtered else 0
    cases.append(
        _case(
            "user_corrections",
            bool("用户修正" in eff and len(session3.corrections.items) >= 4 and after <= before),
            before=before,
            after=after,
            effective_has_block="用户修正" in eff,
            correction_kinds=[c.kind.value for c in session3.corrections.items],
            observe_elements=obs.get("vision", {}).get("element_count"),
        )
    )

    # 4. Metrics collector
    mc = MetricsCollector()
    mc.record_capture(12.5)
    mc.record_vision(30.0)
    mc.record_model(120.0)
    mc.record_e2e(200.0)
    mc.record_queue(2, dropped=1)
    mc.push_error("sample error")
    ms = mc.snapshot()
    cases.append(
        _case(
            "metrics_snapshot",
            bool(
                ms.capture_latency_ms == 12.5
                and ms.vision_latency_ms == 30.0
                and ms.model_latency_ms == 120.0
                and ms.queue_length == 2
                and ms.recent_errors
                and ms.cpu_percent is not None
            ),
            summary=ms.log_summary(),
        )
    )

    # 5. Activity / privacy indicators after stop
    session4 = UISession(cfg, mock=True)
    session4.emergency_stop("bench_stop")
    act = session4.activity()
    cases.append(
        _case(
            "emergency_stop_indicator",
            act.phase.value == "stopped" and act.control_state == "emergency_stop",
            activity=act.log_summary(),
        )
    )
    session4.reset_stop()
    cases.append(
        _case(
            "reset_idle",
            session4.activity().phase.value == "idle",
            activity=session4.activity().log_summary(),
        )
    )

    # 6. Highlight / diagnostics
    session5 = UISession(cfg, mock=True)
    session5.refresh_observe("搜索")
    img = session5.get_last_image()
    vision5 = session5.get_last_vision()
    if img is None:
        img = Image.new("RGB", (320, 180), color=(40, 44, 52))
    highlight_ok = False
    try:
        els = vision5.elements if vision5 is not None else []
        hi = {els[0].element_id} if els else set()
        ann = highlight_elements(img, els, highlight_ids=hi)
        small = resize_for_preview(ann, 320, 240)
        ART.mkdir(parents=True, exist_ok=True)
        out_img = ART / "highlight_preview.png"
        small.save(out_img)
        highlight_ok = out_img.exists() and small.size[0] <= 320
    except Exception as exc:  # noqa: BLE001
        highlight_ok = False
        cases.append(_case("highlight_exception", False, error=str(exc)))
    diag = session5.diagnostics()
    cases.append(
        _case(
            "highlight_and_diagnostics",
            highlight_ok and isinstance(diag.to_dict().get("elements"), list),
            highlight_ok=highlight_ok,
            has_image=session5.get_last_image() is not None,
            diag_keys=list(diag.to_dict().keys()),
            element_rows=len(diag.elements),
        )
    )

    # 7. filter helper unit
    if vision is not None:
        from core.models import BBox, CorrectionKind, UserCorrection

        corr = [
            UserCorrection(kind=CorrectionKind.REJECT_ELEMENT, element_id=reject_id),
            UserCorrection(
                kind=CorrectionKind.IGNORE_REGION,
                region=BBox(x=0, y=0, width=1, height=1),
            ),
        ]
        kept = filter_elements_by_corrections(vision.elements, corr)
        cases.append(
            _case(
                "filter_rejected",
                all(e.element_id != reject_id for e in kept),
                kept=len(kept),
                total=len(vision.elements),
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000
    passed = sum(1 for c in cases if c["ok"])
    return {
        "ok": passed == len(cases),
        "passed": passed,
        "total": len(cases),
        "elapsed_ms": round(elapsed, 2),
        "cases": cases,
        "phase": "H",
        "note": "headless UI session; Tk window not required for acceptance",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase H UI bench")
    p.add_argument(
        "--json-out",
        type=str,
        default=str(RESULTS / "phase_h_ui.json"),
    )
    args = p.parse_args(argv)
    report = run_suite()
    out = Path(args.json_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {"ok": report["ok"], "passed": report["passed"], "total": report["total"]}, indent=2
        )
    )
    print(f"wrote {out}", file=sys.stderr)
    for c in report["cases"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['id']}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
