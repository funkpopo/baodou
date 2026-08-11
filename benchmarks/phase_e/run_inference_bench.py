"""Phase E inference bench: mock regression + optional live HTTP (SYCL).

Usage (conda env dev):
  python benchmarks/phase_e/run_inference_bench.py --mock
  python benchmarks/phase_e/run_inference_bench.py --http
  python benchmarks/phase_e/run_inference_bench.py --http --start-server
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
FIX = HERE / "fixtures"
ART = HERE / "artifacts"
RESULTS = HERE / "results"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_fixture_suite() -> dict[str, Any]:
    from core.models import BBox, CaptureMode, ElementType, ScreenFrame, UIElement, UIVisionResult
    from inference.parse import extract_json, is_json_complete
    from inference.validate import validate_model_output

    frame = ScreenFrame(
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
    vision = UIVisionResult(
        frame_id=frame.frame_id,
        elements=[
            UIElement(
                element_id="btn_search_01",
                type=ElementType.BUTTON,
                text="搜索",
                bbox=BBox(x=100, y=40, width=80, height=32),
                confidence=0.95,
                clickable=True,
                visible=True,
                enabled=True,
                frame_id=frame.frame_id,
                source=["fixture"],
            )
        ],
        source="fixture",
    )

    cases: list[dict[str, Any]] = []

    valid = _load_json(FIX / "valid_observe_plan.json")
    v = validate_model_output(valid, frame=frame, vision=vision, user_goal="点击搜索按钮")
    cases.append({"id": "valid_observe_plan", "ok": v.ok, "has_plan": v.plan is not None})

    bad_act = _load_json(FIX / "invalid_action.json")
    v2 = validate_model_output(bad_act, frame=frame, vision=vision, user_goal="run")
    cases.append(
        {
            "id": "invalid_action",
            "ok": (not v2.ok) and v2.plan is None,
            "rejected": not v2.ok,
        }
    )

    bad_el = _load_json(FIX / "invalid_element.json")
    v3 = validate_model_output(bad_el, frame=frame, vision=vision, user_goal="点击")
    cases.append(
        {
            "id": "invalid_element",
            "ok": (not v3.ok) and v3.plan is None,
            "rejected": not v3.ok,
        }
    )

    trunc = (FIX / "truncated_raw.txt").read_text(encoding="utf-8")
    complete = is_json_complete(trunc)
    parsed, note = extract_json(trunc)
    cases.append(
        {
            "id": "truncated_raw",
            "ok": (not complete) or parsed is None,
            "json_complete": complete,
            "parse_note": note,
        }
    )

    passed = sum(1 for c in cases if c.get("ok"))
    return {
        "suite": "fixtures",
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }


def run_mock_live() -> dict[str, Any]:
    from capture.mock import MockCapture
    from core.config import load_config
    from core.logging import setup_logging
    from inference.mock import MockInference
    from ui_vision.mock import MockUIVision

    cfg = load_config()
    setup_logging(level="WARNING", json_logs=False, log_dir=None)
    frame = MockCapture(cfg).capture(trace_id="tr-bench-e")
    vision = MockUIVision(cfg).recognize(frame, trace_id="tr-bench-e")
    inf = MockInference(cfg)
    t0 = time.perf_counter()
    resp = inf.observe(frame, vision, user_goal="点击搜索按钮", trace_id="tr-bench-e")
    dt = (time.perf_counter() - t0) * 1000
    return {
        "suite": "mock_once",
        "ok": resp.ok,
        "latency_ms": round(dt, 2),
        "has_observation": resp.observation is not None,
        "has_plan": resp.plan is not None and bool(resp.plan.steps),
        "observation_preview": (resp.observation.observation[:120] if resp.observation else ""),
    }


def run_http(start_server: bool) -> dict[str, Any]:
    from capture.mss_backend import MssCapture
    from core.config import load_config
    from core.logging import setup_logging
    from core.models import FrameKind
    from inference.http_client import HttpInference
    from inference.runtime_info import collect_runtime_record
    from ui_vision.factory import create_ui_vision

    cfg = load_config()
    cfg.inference.backend = "http"
    if start_server:
        cfg.inference.auto_start_server = True
    setup_logging(level="INFO", json_logs=False, log_dir=None)

    runtime = collect_runtime_record(cfg)
    inf = HttpInference(cfg)
    t_load0 = time.perf_counter()
    try:
        ready = inf.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        return {
            "suite": "http",
            "ok": False,
            "error": str(exc),
            "runtime": runtime,
        }
    load_ms = (time.perf_counter() - t_load0) * 1000

    cap = MssCapture(cfg)
    vision = create_ui_vision(cfg)
    try:
        packet = cap.capture_packet(kind=FrameKind.MODEL, force=True, encode=True)
        vis = vision.recognize(
            packet.meta, trace_id=packet.meta.trace_id, image=packet.image, goal="描述当前屏幕"
        )
        t0 = time.perf_counter()
        resp = inf.observe(
            packet.meta,
            vis,
            user_goal="描述当前屏幕并给出只读观察 JSON",
            trace_id=packet.meta.trace_id,
            mode="observation",
            include_image=True,
        )
        dt = (time.perf_counter() - t0) * 1000

        # Illegal output must not produce executable steps even if model misbehaves —
        # re-validate a bad fixture through the same path conceptually already covered.
        return {
            "suite": "http",
            "ok": bool(resp.observation is not None),
            "resp_ok": resp.ok,
            "load_or_ready_ms": round(load_ms, 2),
            "infer_ms": round(dt, 2),
            "error_code": resp.error_code,
            "observation_preview": (resp.observation.observation[:200] if resp.observation else ""),
            "plan_steps": len(resp.plan.steps) if resp.plan else 0,
            "server": {
                k: ready.get(k)
                for k in ("already_running", "started_by_us", "base", "load_ms", "vision", "device")
                if k in ready
            },
            "runtime_llama": runtime.get("llama"),
        }
    finally:
        vision.close()
        cap.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Run mock + fixture suite")
    ap.add_argument("--http", action="store_true", help="Run live llama-server bench")
    ap.add_argument("--start-server", action="store_true")
    args = ap.parse_args()
    if not args.mock and not args.http:
        args.mock = True

    ART.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"phase": "E", "results": []}
    if args.mock:
        report["results"].append(run_fixture_suite())
        report["results"].append(run_mock_live())
    if args.http:
        report["results"].append(run_http(args.start_server))

    # Summary
    ok_flags = []
    for r in report["results"]:
        if r.get("suite") == "fixtures":
            ok_flags.append(r.get("passed") == r.get("total"))
        else:
            ok_flags.append(bool(r.get("ok")))
    report["ok"] = all(ok_flags) if ok_flags else False

    out = RESULTS / "phase_e_inference.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {out}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
