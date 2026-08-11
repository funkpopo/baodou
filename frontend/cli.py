"""CLI entry: config, mock demo, capture (C), UI vision (D), inference (E), agent (F), safety (G), UI (H)."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from core.cancel import get_global_token, install_signal_handlers
from core.config import PROJECT_ROOT, config_to_safe_dict, load_config
from core.logging import get_logger, setup_logging
from core.pipeline import MockPipeline

_log = get_logger("frontend.cli")


def _print_json(payload: object) -> None:
    """Print JSON safely on Windows GBK consoles (UIA names may contain special chars)."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="baodou",
        description="baodou — local AI desktop assistant",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package/protocol versions")

    sub.add_parser("config-show", help="Print resolved configuration as JSON")

    demo = sub.add_parser("demo", help="Run mock end-to-end pipeline (Phase B acceptance)")
    demo.add_argument(
        "--goal",
        type=str,
        default="点击搜索按钮",
        help="User goal text",
    )
    demo.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write PipelineResult JSON",
    )
    demo.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="demo_log_level",
        help="Override log level for this demo run",
    )

    health = sub.add_parser("health", help="Check local config and optional llama-server")
    health.add_argument(
        "--http",
        action="store_true",
        help="Also probe inference HTTP /health",
    )

    # --- Phase C capture ---
    cap = sub.add_parser("capture", help="Screen capture (Phase C)")
    cap_sub = cap.add_subparsers(dest="capture_cmd", required=True)

    mon = cap_sub.add_parser("monitors", help="List monitors / virtual desktop / cursor")
    mon.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="cap_log_level",
    )

    once = cap_sub.add_parser("once", help="Single screenshot")
    once.add_argument(
        "--mode",
        choices=["primary", "all", "window", "region"],
        default=None,
    )
    once.add_argument(
        "--kind", choices=["preview", "vision", "model", "verify", "raw"], default="vision"
    )
    once.add_argument("--out", type=str, default=None, help="Output image path")
    once.add_argument("--window-title", type=str, default=None)
    once.add_argument("--region", type=str, default=None, help="x,y,w,h physical pixels")
    once.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="cap_log_level",
    )

    stream = cap_sub.add_parser("stream", help="Run bounded multi-stream pipeline briefly")
    stream.add_argument("--seconds", type=float, default=2.0)
    stream.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="cap_log_level",
    )

    # --- Phase D UI vision ---
    vis = sub.add_parser("vision", help="UI recognition (Phase D)")
    vis_sub = vis.add_subparsers(dest="vision_cmd", required=True)

    vonce = vis_sub.add_parser("once", help="Capture + recognize UI elements once")
    vonce.add_argument(
        "--backend",
        choices=["composite", "uia", "ocr", "rules", "mock"],
        default=None,
        help="Override ui_vision.backend",
    )
    vonce.add_argument(
        "--mode",
        choices=["primary", "all", "window", "region"],
        default=None,
    )
    vonce.add_argument("--window-title", type=str, default=None)
    vonce.add_argument("--region", type=str, default=None, help="x,y,w,h physical pixels (ROI)")
    vonce.add_argument("--goal", type=str, default=None, help="Filter compact context by goal")
    vonce.add_argument(
        "--annotate",
        type=str,
        default=None,
        help="Save numbered-box annotation image path",
    )
    vonce.add_argument("--json-out", type=str, default=None, help="Write full vision JSON")
    vonce.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="vis_log_level",
    )

    vctx = vis_sub.add_parser("context", help="Print compact element list for the model")
    vctx.add_argument(
        "--backend", choices=["composite", "uia", "ocr", "rules", "mock"], default=None
    )
    vctx.add_argument("--goal", type=str, default=None)
    vctx.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="vis_log_level",
    )

    # --- Phase E inference ---
    inf = sub.add_parser("infer", help="llama.cpp / Qwen inference (Phase E)")
    inf_sub = inf.add_subparsers(dest="infer_cmd", required=True)

    inf_info = inf_sub.add_parser("info", help="Record llama binary / config runtime info")
    inf_info.add_argument("--json-out", type=str, default=None, help="Write runtime record JSON")
    inf_info.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="inf_log_level",
    )

    inf_srv = inf_sub.add_parser("server", help="Manage llama-server lifecycle")
    inf_srv.add_argument(
        "action",
        choices=["status", "start", "stop", "warmup", "recover"],
        help="Lifecycle action",
    )
    inf_srv.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="inf_log_level",
    )

    inf_once = inf_sub.add_parser(
        "once", help="Capture + UI vision + model observe/plan (mock or http)"
    )
    inf_once.add_argument(
        "--backend",
        choices=["mock", "http"],
        default=None,
        help="Override inference.backend",
    )
    inf_once.add_argument(
        "--vision-backend",
        choices=["composite", "uia", "ocr", "rules", "mock"],
        default=None,
    )
    inf_once.add_argument("--goal", type=str, default="描述当前屏幕")
    inf_once.add_argument(
        "--mode",
        choices=["observe_plan", "observation"],
        default="observe_plan",
    )
    inf_once.add_argument("--no-image", action="store_true", help="Text+UI only (no screenshot)")
    inf_once.add_argument("--stream", action="store_true", help="Use streaming gate")
    inf_once.add_argument("--start-server", action="store_true", help="Auto-start llama-server")
    inf_once.add_argument("--json-out", type=str, default=None)
    inf_once.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="inf_log_level",
    )

    inf_reg = inf_sub.add_parser("prompts", help="Show prompt registry / versions")
    inf_reg.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="inf_log_level",
    )

    # --- Phase F agent ---
    ag = sub.add_parser("agent", help="Task agent / state machine (Phase F)")
    ag_sub = ag.add_subparsers(dest="agent_cmd", required=True)

    ag_run = ag_sub.add_parser("run", help="Run task: observe → plan → confirm → act → verify")
    ag_run.add_argument("--goal", type=str, required=True, help="Natural language user goal")
    ag_run.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm low-risk steps (still blocks high risk)",
    )
    ag_run.add_argument(
        "--preview-only",
        action="store_true",
        help="Plan + preview only; do not execute",
    )
    ag_run.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Force dry-run (no real OS input); default from config",
    )
    ag_run.add_argument(
        "--live",
        action="store_true",
        help="Allow real input when dry_run=false (dangerous; still needs --yes)",
    )
    ag_run.add_argument(
        "--mock",
        action="store_true",
        help="Force mock capture/vision/inference/actuator (safe offline)",
    )
    ag_run.add_argument(
        "--actuator",
        choices=["mock", "win"],
        default=None,
        help="Override actuator.backend",
    )
    ag_run.add_argument("--max-steps", type=int, default=None)
    ag_run.add_argument("--json-out", type=str, default=None)
    ag_run.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ag_log_level",
    )

    ag_prev = ag_sub.add_parser("preview", help="Show action previews without executing")
    ag_prev.add_argument("--goal", type=str, required=True)
    ag_prev.add_argument("--mock", action="store_true")
    ag_prev.add_argument("--json-out", type=str, default=None)
    ag_prev.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ag_log_level",
    )

    ag_states = ag_sub.add_parser("states", help="Print task state machine graph")
    ag_states.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ag_log_level",
    )

    # --- Phase G safety ---
    saf = sub.add_parser("safety", help="Safety / permission / privacy (Phase G)")
    saf_sub = saf.add_subparsers(dest="safety_cmd", required=True)

    saf_status = saf_sub.add_parser("status", help="Show control plane + safety config summary")
    saf_status.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="saf_log_level",
    )

    saf_pause = saf_sub.add_parser("pause", help="Global pause")
    saf_pause.add_argument("--reason", type=str, default="cli_pause")
    saf_sub.add_parser("resume", help="Resume from pause")
    saf_stop = saf_sub.add_parser("stop", help="Emergency stop (cancels global token)")
    saf_stop.add_argument("--reason", type=str, default="cli_emergency_stop")
    saf_sub.add_parser("reset", help="Clear emergency stop + cancel token")

    saf_check = saf_sub.add_parser("check", help="Evaluate one synthetic step against policy")
    saf_check.add_argument("--goal", type=str, required=True)
    saf_check.add_argument(
        "--action",
        type=str,
        default="click",
        help="Action type (click|type|hotkey|...)",
    )
    saf_check.add_argument("--text", type=str, default="")
    saf_check.add_argument("--element-id", type=str, default="btn_ok_01")
    saf_check.add_argument("--risk", choices=["low", "medium", "high"], default="low")
    saf_check.add_argument(
        "--screen-text",
        type=str,
        default="",
        help="Untrusted screen/OCR text to include in threat scan",
    )
    saf_check.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="saf_log_level",
    )

    saf_redact = saf_sub.add_parser("redact", help="Redact PII in sample text")
    saf_redact.add_argument("--text", type=str, required=True)

    saf_sub.add_parser("threats", help="Print threat model summary")

    saf_audit = saf_sub.add_parser("audit", help="Audit log ops")
    saf_audit_sub = saf_audit.add_subparsers(dest="audit_cmd", required=True)
    saf_audit_sub.add_parser("path", help="Show audit directory / today's file")
    aud_clean = saf_audit_sub.add_parser("cleanup", help="Cleanup audit files")
    aud_clean.add_argument("--days", type=int, default=None, help="Remove files older than N days")
    aud_clean.add_argument("--wipe", action="store_true", help="Wipe all audit files")
    saf_audit_sub.add_parser("disable", help="Disable audit file persistence (runtime)")

    # --- Phase H UI / observability ---
    ui = sub.add_parser("ui", help="Main window / session observability (Phase H)")
    ui_sub = ui.add_subparsers(dest="ui_cmd", required=True)

    ui_open = ui_sub.add_parser("open", help="Open Tk main window")
    ui_open.add_argument(
        "--live",
        action="store_true",
        help="Use real capture/vision backends (still dry_run by default)",
    )
    ui_open.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ui_log_level",
    )

    ui_run = ui_sub.add_parser(
        "run", help="Headless session: observe/plan/execute via UISession (no Tk)"
    )
    ui_run.add_argument("--goal", type=str, required=True)
    ui_run.add_argument("--yes", action="store_true", help="Auto-confirm low-risk steps")
    ui_run.add_argument("--preview-only", action="store_true")
    ui_run.add_argument("--mock", action="store_true", default=True)
    ui_run.add_argument(
        "--reject",
        type=str,
        default=None,
        help="Reject element_id before run (user correction)",
    )
    ui_run.add_argument(
        "--prefer",
        type=str,
        default=None,
        help="Prefer element_id before run (user correction)",
    )
    ui_run.add_argument("--json-out", type=str, default=None)
    ui_run.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ui_log_level",
    )

    ui_status = ui_sub.add_parser("status", help="Headless session status + metrics sample")
    ui_status.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ui_log_level",
    )

    ui_corr = ui_sub.add_parser("correct", help="Demo user corrections on mock vision")
    ui_corr.add_argument("--goal", type=str, default="点击搜索按钮")
    ui_corr.add_argument("--reject", type=str, default=None)
    ui_corr.add_argument("--prefer", type=str, default=None)
    ui_corr.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="ui_log_level",
    )

    return p


def cmd_version() -> int:
    from core import __version__
    from core.models import PROTOCOL_VERSION

    print(json.dumps({"baodou": __version__, "protocol": PROTOCOL_VERSION}, ensure_ascii=False))
    return 0


def cmd_config_show(config_path: str | None) -> int:
    cfg = load_config(config_path)
    print(json.dumps(config_to_safe_dict(cfg), ensure_ascii=False, indent=2))
    return 0


def cmd_demo(
    config_path: str | None, goal: str, json_out: str | None, log_level: str | None
) -> int:
    cfg = load_config(config_path)
    level = log_level or cfg.app.log_level
    setup_logging(
        level=level,
        json_logs=cfg.app.log_json,
        log_dir=cfg.app.log_dir,
        project_root=cfg.project_root,
    )
    install_signal_handlers(get_global_token(), graceful_sec=cfg.app.graceful_shutdown_sec)

    _log.info(
        "demo_start",
        extra={"event": "demo_start", "goal": goal, "backend_capture": cfg.capture.backend},
    )
    pipe = MockPipeline(cfg)
    result = pipe.run(goal)
    summary = result.to_dict()
    # Compact console summary (logs already have full chain).
    print(
        json.dumps(
            {
                "ok": result.ok,
                "trace_id": result.trace_id,
                "task_state": result.task.state.value,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "event_kinds": [e.kind.value for e in result.events],
                "error": result.error.to_dict() if result.error else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if json_out:
        out = Path(json_out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {out}", file=sys.stderr)
    # Accept demo as success if chain completed or paused for confirmation with dry-run act.
    if result.ok:
        return 0
    if result.action and result.verification and result.verification.passed:
        return 0
    return 1


def cmd_health(config_path: str | None, probe_http: bool) -> int:
    cfg = load_config(config_path)
    setup_logging(level="INFO", json_logs=False, log_dir=None)
    checks: dict[str, object] = {
        "config_ok": True,
        "schema_version": cfg.schema_version,
        "model_gguf_exists": cfg.paths.model_gguf.exists(),
        "mmproj_exists": cfg.paths.mmproj_gguf.exists(),
        "llama_dir_exists": cfg.paths.llama_dir.exists(),
        "inference_backend": cfg.inference.backend,
        "capture_backend": cfg.capture.backend,
        "ui_vision_backend": cfg.ui_vision.backend,
        "ui_vision_sources": cfg.ui_vision.sources,
    }
    try:
        from capture.geometry import list_monitors, virtual_desktop

        mons = list_monitors()
        checks["monitors"] = len(mons)
        checks["virtual_desktop"] = virtual_desktop().model_dump()
        checks["primary"] = next(
            (m.model_dump() for m in mons if m.is_primary),
            mons[0].model_dump() if mons else None,
        )
    except Exception as exc:  # noqa: BLE001
        checks["monitors_error"] = str(exc)
    if probe_http or cfg.inference.backend == "http":
        from inference.http_client import HttpInference

        http = HttpInference(cfg)
        checks["llama_server_health"] = http.health()
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))
    return 0 if checks.get("config_ok") else 1


def _setup_from_args(config_path: str | None, log_level: str | None):
    cfg = load_config(config_path)
    level = log_level or cfg.app.log_level
    setup_logging(
        level=level,
        json_logs=cfg.app.log_json,
        log_dir=cfg.app.log_dir,
        project_root=cfg.project_root,
    )
    install_signal_handlers(get_global_token(), graceful_sec=cfg.app.graceful_shutdown_sec)
    return cfg


def cmd_capture_monitors(config_path: str | None, log_level: str | None) -> int:
    _setup_from_args(config_path, log_level)
    from capture.geometry import cursor_pos_physical, list_monitors, virtual_desktop

    payload = {
        "monitors": [m.model_dump(mode="json") for m in list_monitors()],
        "virtual_desktop": virtual_desktop().model_dump(mode="json"),
        "cursor": cursor_pos_physical().model_dump(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_capture_once(
    config_path: str | None,
    *,
    mode: str | None,
    kind: str,
    out: str | None,
    window_title: str | None,
    region: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    cfg.capture.backend = "mss"
    if mode:
        cfg.capture.mode = mode  # type: ignore[assignment]
    if window_title:
        cfg.capture.mode = "window"
        cfg.capture.window_title = window_title
    if region:
        parts = [int(x.strip()) for x in region.split(",")]
        if len(parts) != 4:
            print("region must be x,y,w,h", file=sys.stderr)
            return 2
        cfg.capture.mode = "region"
        cfg.capture.region = {"x": parts[0], "y": parts[1], "width": parts[2], "height": parts[3]}

    from capture.mss_backend import MssCapture
    from core.models import FrameKind

    kind_enum = FrameKind(kind)
    cap = MssCapture(cfg)
    try:
        packet = cap.capture_packet(kind=kind_enum, force=True, encode=True)
        out_path = (
            Path(out)
            if out
            else (PROJECT_ROOT / "benchmarks" / "phase_c" / "artifacts" / f"once_{kind}.png")
        )
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        packet.save(out_path)
        center = packet.meta.image_to_screen(packet.meta.width / 2, packet.meta.height / 2)
        print(
            json.dumps(
                {
                    **packet.meta.log_summary(),
                    "saved": str(out_path),
                    "center_screen": center.model_dump(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    finally:
        cap.close()
    return 0


def cmd_capture_stream(config_path: str | None, seconds: float, log_level: str | None) -> int:
    cfg = _setup_from_args(config_path, log_level)
    cfg.capture.backend = "mss"
    from capture.pipeline import CapturePipeline
    from core.models import FrameKind

    pipe = CapturePipeline(cfg)
    consumed = {
        k.value: 0 for k in (FrameKind.PREVIEW, FrameKind.VISION, FrameKind.MODEL, FrameKind.VERIFY)
    }
    try:
        pipe.start()
        pipe.request_model_frame(user_request=True)
        end = time.monotonic() + max(0.2, seconds)
        while time.monotonic() < end and not get_global_token().is_cancelled:
            for kind in (FrameKind.PREVIEW, FrameKind.VISION, FrameKind.MODEL, FrameKind.VERIFY):
                pkt = pipe.latest(kind, timeout=0.01)
                if pkt is not None:
                    consumed[kind.value] += 1
                    pkt.release()
            time.sleep(0.01)
        stats = pipe.stats_dict()
    finally:
        pipe.close()

    bounded = all(
        stats["streams"][k]["high_watermark"] <= cfg.capture.queue_size for k in stats["streams"]
    )
    print(
        json.dumps(
            {
                "ok": bounded,
                "seconds": seconds,
                "consumed": consumed,
                "stats": stats,
                "queues_bounded": bounded,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if bounded else 1


def _parse_region(region: str | None) -> dict[str, int] | None:
    if not region:
        return None
    parts = [int(x.strip()) for x in region.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be x,y,w,h")
    return {"x": parts[0], "y": parts[1], "width": parts[2], "height": parts[3]}


def cmd_vision_once(
    config_path: str | None,
    *,
    backend: str | None,
    mode: str | None,
    window_title: str | None,
    region: str | None,
    goal: str | None,
    annotate: str | None,
    json_out: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    if backend:
        cfg.ui_vision.backend = backend  # type: ignore[assignment]
        if backend != "composite":
            cfg.ui_vision.sources = [backend]
    cfg.capture.backend = "mss"
    if mode:
        cfg.capture.mode = mode  # type: ignore[assignment]
    if window_title:
        cfg.capture.mode = "window"
        cfg.capture.window_title = window_title
    roi = None
    try:
        region_dict = _parse_region(region)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if region_dict and mode == "region":
        cfg.capture.mode = "region"
        cfg.capture.region = region_dict
    elif region_dict:
        from core.models import BBox

        roi = BBox(
            x=region_dict["x"],
            y=region_dict["y"],
            width=region_dict["width"],
            height=region_dict["height"],
        )

    from capture.mss_backend import MssCapture
    from core.models import FrameKind
    from ui_vision.context import annotate_from_frame, serialize_for_model, serialize_text_summary
    from ui_vision.factory import create_ui_vision

    cap = MssCapture(cfg)
    vision = create_ui_vision(cfg)
    try:
        packet = cap.capture_packet(kind=FrameKind.VISION, force=True, encode=True)
        result = vision.recognize(
            packet.meta,
            trace_id=packet.meta.trace_id,
            image=packet.image,
            roi=roi,
            goal=goal,
        )
        compact = serialize_for_model(
            result, goal=goal, max_elements=cfg.ui_vision.context_max_elements
        )
        # Multi-res sanity: every center must map back into image bounds roughly
        coord_ok = True
        samples = []
        for el in result.elements[:12]:
            c = el.center
            img_pt = packet.meta.screen_to_image(c.x, c.y)
            inside = 0 <= img_pt.x < packet.meta.width and 0 <= img_pt.y < packet.meta.height
            # Allow slight tolerance for chrome outside scaled ROI
            if (
                el.visible
                and (el.clickable or el.editable)
                and not inside
                and not (
                    -2 <= img_pt.x < packet.meta.width + 2
                    and -2 <= img_pt.y < packet.meta.height + 2
                )
            ):
                coord_ok = False
            samples.append(
                {
                    "element_id": el.element_id,
                    "type": el.type.value,
                    "text": el.text[:40],
                    "center": c.model_dump(),
                    "image_xy": img_pt.model_dump(),
                    "dpi_scale": el.dpi_scale,
                    "source": el.source,
                }
            )

        ann_path = None
        if annotate or cfg.ui_vision.annotate_boxes:
            out_ann = (
                Path(annotate)
                if annotate
                else (
                    PROJECT_ROOT
                    / "benchmarks"
                    / "phase_d"
                    / "artifacts"
                    / f"annotate_{packet.meta.frame_id}.png"
                )
            )
            if not out_ann.is_absolute():
                out_ann = PROJECT_ROOT / out_ann
            if packet.image is not None:
                ann = annotate_from_frame(
                    packet.image,
                    result,
                    packet.meta.screen_to_image,
                    goal=goal,
                    max_draw=cfg.ui_vision.context_max_elements,
                )
                out_ann.parent.mkdir(parents=True, exist_ok=True)
                ann.save(out_ann)
                ann_path = str(out_ann)

        payload = {
            "ok": coord_ok and len(result.elements) >= 0,
            "coord_ok": coord_ok,
            "vision": result.log_summary(),
            "frame": packet.meta.log_summary(),
            "compact_count": len(compact),
            "compact": compact[:24],
            "samples": samples,
            "text_summary": serialize_text_summary(
                result, goal=goal, max_elements=min(16, cfg.ui_vision.context_max_elements)
            ),
            "annotate": ann_path,
        }
        _print_json(payload)
        if json_out:
            out = Path(json_out)
            if not out.is_absolute():
                out = PROJECT_ROOT / out
            out.parent.mkdir(parents=True, exist_ok=True)
            full = {
                **payload,
                "elements": [e.model_dump(mode="json") for e in result.elements],
                "compact_all": compact,
            }
            out.write_text(
                json.dumps(full, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print(f"wrote {out}", file=sys.stderr)
    finally:
        vision.close()
        cap.close()
    return 0


def cmd_vision_context(
    config_path: str | None,
    *,
    backend: str | None,
    goal: str | None,
    log_level: str | None,
) -> int:
    return cmd_vision_once(
        config_path,
        backend=backend or "mock",
        mode="primary",
        window_title=None,
        region=None,
        goal=goal,
        annotate=None,
        json_out=None,
        log_level=log_level,
    )


def cmd_infer_info(config_path: str | None, json_out: str | None, log_level: str | None) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from inference.runtime_info import collect_runtime_record

    rec = collect_runtime_record(cfg)
    _print_json(rec)
    if json_out:
        out = Path(json_out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0 if rec.get("llama", {}).get("exists") else 1


def cmd_infer_prompts(config_path: str | None, log_level: str | None) -> int:
    _setup_from_args(config_path, log_level)
    from inference.prompts import prompt_registry_meta

    _print_json(prompt_registry_meta())
    return 0


def cmd_infer_server(config_path: str | None, action: str, log_level: str | None) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from inference.server import LlamaServerManager

    mgr = LlamaServerManager(cfg)
    try:
        if action == "status":
            _print_json(mgr.status())
            return 0 if mgr.health() else 1
        if action == "start":
            cfg.inference.auto_start_server = True
            info = mgr.start(warmup=cfg.inference.warmup_on_start)
            _print_json(info)
            return 0
        if action == "warmup":
            if not mgr.health():
                print("server not healthy; start first", file=sys.stderr)
                return 1
            _print_json(mgr.warmup())
            return 0
        if action == "recover":
            _print_json(mgr.recover())
            return 0
        if action == "stop":
            # Only stops process we manage in this process; external servers need manual kill.
            mgr.stop(force=True)
            _print_json(
                {"stopped": True, "note": "managed process only; external server untouched"}
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 2


def cmd_infer_once(
    config_path: str | None,
    *,
    backend: str | None,
    vision_backend: str | None,
    goal: str,
    mode: str,
    no_image: bool,
    stream: bool,
    start_server: bool,
    json_out: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    if backend:
        cfg.inference.backend = backend  # type: ignore[assignment]
    if vision_backend:
        cfg.ui_vision.backend = vision_backend  # type: ignore[assignment]
        if vision_backend != "composite":
            cfg.ui_vision.sources = [vision_backend]
    if start_server:
        cfg.inference.auto_start_server = True
        cfg.inference.backend = "http"

    from capture.factory import create_capture
    from core.models import FrameKind
    from inference.http_client import create_inference
    from ui_vision.factory import create_ui_vision

    # Prefer real capture for http; mock capture still works for offline.
    if cfg.inference.backend == "http" and cfg.capture.backend == "mock":
        cfg.capture.backend = "mss"

    cap = create_capture(cfg)
    vision = create_ui_vision(cfg)
    inf = create_inference(cfg)
    try:
        if cfg.inference.backend == "http":
            ready = inf.ensure_ready()
            print(
                json.dumps(
                    {
                        "server": {
                            k: ready.get(k)
                            for k in (
                                "already_running",
                                "started_by_us",
                                "base",
                                "load_ms",
                                "vision",
                            )
                            if k in ready
                        }
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )

        packet_meta = None
        image = None
        if hasattr(cap, "capture_packet"):
            packet = cap.capture_packet(kind=FrameKind.MODEL, force=True, encode=True)  # type: ignore[attr-defined]
            # Model path needs base64 on ScreenFrame (encode alone only fills image_bytes).
            if not no_image and hasattr(packet, "attach_b64"):
                with contextlib.suppress(Exception):
                    packet.attach_b64()
            frame = packet.meta
            image = packet.image
            packet_meta = frame
        else:
            frame = cap.capture()
            packet_meta = frame

        if no_image:
            frame.image_b64 = None
            frame.image_path = None

        vis = vision.recognize(
            frame,
            trace_id=frame.trace_id,
            image=image,
            goal=goal,
        )

        if stream and cfg.inference.backend == "http":
            final = None
            for ev in inf.stream_observe(
                frame,
                vis,
                user_goal=goal,
                trace_id=frame.trace_id or "",
                mode=mode,
                include_image=not no_image,
            ):
                if ev.get("type") == "delta":
                    continue
                final = ev
            if not final or not final.get("response"):
                _print_json({"ok": False, "stream": final})
                return 1
            resp = final["response"]
            ready_for_action = bool(final.get("ready_for_action"))
        else:
            resp = inf.observe(
                frame,
                vis,
                user_goal=goal,
                trace_id=frame.trace_id or "",
                mode=mode,
                include_image=not no_image,
            )
            ready_for_action = bool(resp.ok and resp.observation is not None)
            # Plans are validated but still require Phase F safety before actuation.
            if resp.plan and resp.plan.steps:
                ready_for_action = bool(resp.ok)

        payload = {
            "ok": resp.ok,
            "ready_for_action_candidate": ready_for_action,
            "note": "validated plan is NOT executed here; Phase F safety/actuator required",
            "trace_id": resp.trace_id,
            "latency_ms": resp.latency_ms,
            "error_code": resp.error_code,
            "error_message": resp.error_message,
            "frame": packet_meta.log_summary() if packet_meta else None,
            "vision": vis.log_summary(),
            "observation": resp.observation.log_summary() if resp.observation else None,
            "plan": resp.plan.log_summary() if resp.plan else None,
            "observation_text": (resp.observation.observation if resp.observation else "")[:500],
            "raw_preview": (resp.raw_text or "")[:600],
        }
        _print_json(payload)
        if json_out:
            out = Path(json_out)
            if not out.is_absolute():
                out = PROJECT_ROOT / out
            out.parent.mkdir(parents=True, exist_ok=True)
            full = {
                **payload,
                "observation_full": resp.observation.model_dump(mode="json")
                if resp.observation
                else None,
                "plan_full": resp.plan.model_dump(mode="json") if resp.plan else None,
                "raw_text": resp.raw_text,
            }
            out.write_text(
                json.dumps(full, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print(f"wrote {out}", file=sys.stderr)
        # Phase E: structured observation (even degraded / plan-rejected) is a successful handle.
        # Illegal plans never execute here; exit 0 means "safe structured result", not "plan approved".
        return 0 if resp.observation is not None else 1
    finally:
        vision.close()
        cap.close()
        inf.close()


def _apply_agent_mock(cfg) -> None:
    cfg.capture.backend = "mock"
    cfg.ui_vision.backend = "mock"
    cfg.inference.backend = "mock"
    cfg.actuator.backend = "mock"
    cfg.agent.backend = "mock"


def cmd_agent_run(
    config_path: str | None,
    *,
    goal: str,
    yes: bool,
    preview_only: bool,
    dry_run: bool | None,
    live: bool,
    mock: bool,
    actuator: str | None,
    max_steps: int | None,
    json_out: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    # Safe default: offline mock chain unless --live explicitly requested.
    if mock or not live:
        _apply_agent_mock(cfg)
    else:
        # Real capture + vision; actuator still dry_run unless config disables it.
        if cfg.capture.backend == "mock":
            cfg.capture.backend = "mss"
        if cfg.ui_vision.backend == "mock":
            cfg.ui_vision.backend = "composite"
    if actuator:
        cfg.actuator.backend = actuator  # type: ignore[assignment]
    if dry_run is True:
        cfg.actuator.dry_run = True
    if live and dry_run is not True:
        # --live alone does NOT disable dry_run; require explicit env/config change.
        # Only turn off dry_run if user set actuator.dry_run false in config already.
        pass
    if yes:
        cfg.agent.auto_confirm = True

    from agent.factory import create_task_agent
    from core.models import TaskState

    agent = create_task_agent(cfg)
    try:
        if preview_only:
            result = agent.preview(goal)
        else:
            result = agent.run(
                goal,
                execute=True,
                confirmed=bool(yes),
                max_steps=max_steps,
            )
        summary = {
            "ok": result.ok,
            "trace_id": result.trace_id,
            "task_state": result.task.state.value,
            "elapsed_ms": round(result.elapsed_ms, 2),
            "goal": goal,
            "dry_run": cfg.actuator.dry_run,
            "auto_confirm": cfg.agent.auto_confirm,
            "plan": result.plan.log_summary() if result.plan else None,
            "previews": [p.log_summary() for p in result.previews],
            "steps_done": result.task.steps_done,
            "steps_skipped": result.task.steps_skipped,
            "actions": [a.log_summary() for a in result.actions],
            "verifications": [v.log_summary() for v in result.verifications],
            "pause_reason": result.task.pause_reason or None,
            "error": result.error.to_dict() if result.error else None,
            "event_kinds": [e.kind.value for e in result.events],
        }
        _print_json(summary)
        if json_out:
            out = Path(json_out)
            if not out.is_absolute():
                out = PROJECT_ROOT / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"wrote {out}", file=sys.stderr)
        if result.ok:
            return 0
        # Paused for confirmation is a controlled outcome (exit 2) when not --yes
        if result.task.state == TaskState.PAUSED:
            return 2
        return 1
    finally:
        agent.close()


def cmd_agent_preview(
    config_path: str | None,
    *,
    goal: str,
    mock: bool,
    json_out: str | None,
    log_level: str | None,
) -> int:
    return cmd_agent_run(
        config_path,
        goal=goal,
        yes=False,
        preview_only=True,
        dry_run=True,
        live=False,
        mock=mock or True,
        actuator="mock",
        max_steps=None,
        json_out=json_out,
        log_level=log_level,
    )


def cmd_agent_states(config_path: str | None, log_level: str | None) -> int:
    _setup_from_args(config_path, log_level)
    from agent.state import allowed_targets
    from core.models import TaskState

    graph = {s.value: allowed_targets(s) for s in TaskState}
    _print_json({"states": [s.value for s in TaskState], "transitions": graph})
    return 0


def cmd_safety_status(config_path: str | None, log_level: str | None) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from safety.control import get_safety_control

    ctrl = get_safety_control()
    ctrl.emergency_stop_enabled = cfg.safety.emergency_stop_enabled
    ctrl.pause_on_focus_loss = cfg.safety.pause_on_focus_loss
    _print_json(
        {
            "control": ctrl.status(),
            "safety": {
                "default_mode": cfg.safety.default_mode,
                "block_high_risk": cfg.safety.block_high_risk,
                "require_confirmation_below": cfg.safety.require_confirmation_below,
                "audit_enabled": cfg.safety.audit_enabled,
                "audit_dir": cfg.safety.audit_dir,
                "redact_pii": cfg.safety.redact_pii,
                "max_actions_per_minute": cfg.safety.max_actions_per_minute,
                "max_consecutive_actions": cfg.safety.max_consecutive_actions,
                "max_task_duration_sec": cfg.safety.max_task_duration_sec,
                "max_mouse_move_px": cfg.safety.max_mouse_move_px,
                "action_whitelist": cfg.safety.action_whitelist,
                "app_denylist": cfg.safety.app_denylist,
                "window_title_denylist": cfg.safety.window_title_denylist,
            },
            "actuator_dry_run": cfg.actuator.dry_run,
        }
    )
    return 0


def cmd_safety_control(config_path: str | None, action: str, reason: str = "") -> int:
    cfg = _setup_from_args(config_path, None)
    from safety.control import get_safety_control

    ctrl = get_safety_control()
    ctrl.emergency_stop_enabled = cfg.safety.emergency_stop_enabled
    if action == "pause":
        ctrl.request_pause(reason or "cli_pause")
    elif action == "resume":
        try:
            ctrl.request_resume(reason or "cli_resume")
        except Exception as exc:  # noqa: BLE001
            _print_json({"ok": False, "error": str(exc), "control": ctrl.status()})
            return 1
    elif action == "stop":
        ctrl.request_stop(reason or "cli_emergency_stop", token=get_global_token())
    elif action == "reset":
        ctrl.reset_stop(reason or "cli_reset")
    else:
        _print_json({"ok": False, "error": f"unknown action {action}"})
        return 1
    _print_json({"ok": True, "action": action, "control": ctrl.status()})
    return 0


def cmd_safety_check(
    config_path: str | None,
    *,
    goal: str,
    action: str,
    text: str,
    element_id: str,
    risk: str,
    screen_text: str,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from core.models import ActionPlan, ActionStep, ActionType, RiskLevel
    from safety.policy import SafetyPolicy
    from safety.risk import classify_step
    from safety.threats import scan_plan

    try:
        act = ActionType(action)
    except ValueError:
        _print_json({"ok": False, "error": f"invalid action: {action}"})
        return 1
    step = ActionStep(
        action=act,
        target_element_id=element_id or None,
        text=text or None,
        risk=RiskLevel(risk),
        requires_confirmation=True,
        expected_change="cli safety check",
    )
    plan = ActionPlan(goal=goal, steps=[step], risk_max=RiskLevel(risk))
    screen_texts = [screen_text] if screen_text else []
    threats = scan_plan(plan, cfg=cfg.safety, screen_texts=screen_texts)
    cat, elev, rules = classify_step(
        step, plan=plan, sensitive_keywords=cfg.safety.sensitive_keywords
    )
    decision = SafetyPolicy(cfg).evaluate(
        step, plan, screen_texts=screen_texts, threat_report=threats
    )
    _print_json(
        {
            "goal": goal,
            "classification": {
                "category": cat.value,
                "risk": elev.value,
                "rules": rules,
            },
            "threats": threats.log_summary(),
            "decision": decision.log_summary(),
            "mode": cfg.safety.default_mode,
        }
    )
    return 0 if decision.allowed else 1


def cmd_safety_redact(text: str) -> int:
    from safety.redact import redact_text

    out = redact_text(text, enabled=True)
    _print_json({"input": text, "redacted": out, "changed": out != text})
    return 0


def cmd_safety_threats() -> int:
    from safety.threats import THREAT_MODEL

    _print_json({"threat_model": THREAT_MODEL})
    return 0


def cmd_safety_audit(
    config_path: str | None, audit_cmd: str, *, days: int | None, wipe: bool
) -> int:
    cfg = _setup_from_args(config_path, None)
    from safety.audit import AuditLog

    audit = AuditLog(cfg)
    if audit_cmd == "path":
        _print_json(
            {
                "enabled": audit.enabled(),
                "path": str(audit.path) if audit.path else None,
                "dir": cfg.safety.audit_dir,
            }
        )
        return 0
    if audit_cmd == "cleanup":
        result = audit.cleanup(older_than_days=days, wipe_all=wipe)
        _print_json({"ok": True, **result})
        return 0
    if audit_cmd == "disable":
        audit.disable_persistence()
        _print_json({"ok": True, "audit_enabled": False})
        return 0
    _print_json({"ok": False, "error": f"unknown audit cmd {audit_cmd}"})
    return 1


def cmd_ui_open(config_path: str | None, *, live: bool, log_level: str | None) -> int:
    from frontend.app import run_app

    return run_app(config_path, mock=not live, log_level=log_level)


def cmd_ui_run(
    config_path: str | None,
    *,
    goal: str,
    yes: bool,
    preview_only: bool,
    mock: bool,
    reject: str | None,
    prefer: str | None,
    json_out: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from core.models import TaskState
    from safety.control import reset_safety_control

    from frontend.session import UISession

    reset_safety_control()
    # Headless runs default to allow_low + auto when --yes so mock demos complete
    if yes:
        cfg.safety.default_mode = "allow_low"
        cfg.agent.auto_confirm = True
    session = UISession(cfg, mock=mock if mock is not False else True)
    if reject:
        session.reject_element(reject, note="cli reject")
    if prefer:
        session.prefer_element(prefer, note="cli prefer")
    result = session.start_task(
        goal,
        execute=not preview_only,
        auto_confirm=bool(yes),
        background=False,
    )
    assert result is not None
    snap = session.snapshot()
    diag = session.diagnostics().to_dict()
    payload = {
        "ok": result.ok,
        "trace_id": result.trace_id,
        "task_state": result.task.state.value,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "goal": goal,
        "activity": snap.activity.log_summary(),
        "metrics": snap.metrics.log_summary(),
        "corrections": snap.corrections,
        "plan": result.plan.log_summary() if result.plan else None,
        "previews": [p.log_summary() for p in result.previews],
        "observation": (result.observation.observation[:200] if result.observation else ""),
        "elements": snap.elements[:16],
        "error": result.error.to_dict() if result.error else None,
        "diagnostics_keys": list(diag.keys()),
    }
    _print_json(payload)
    if json_out:
        out = Path(json_out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        full = {
            **payload,
            "snapshot": snap.to_dict(),
            "diagnostics": diag,
            "result": result.to_dict(),
        }
        out.write_text(
            json.dumps(full, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {out}", file=sys.stderr)
    if result.ok:
        return 0
    if result.task.state == TaskState.PAUSED:
        return 2
    return 1


def cmd_ui_status(config_path: str | None, log_level: str | None) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from safety.control import get_safety_control

    from frontend.session import UISession

    session = UISession(cfg, mock=True)
    snap = session.snapshot()
    _print_json(
        {
            "frontend": {
                "mode": cfg.frontend.mode,
                "window_title": cfg.frontend.window_title,
                "show_diagnostics": cfg.frontend.show_diagnostics,
                "activity_indicators": cfg.frontend.activity_indicators,
                "default_mock": cfg.frontend.default_mock,
            },
            "activity": snap.activity.log_summary(),
            "metrics": snap.metrics.log_summary(),
            "control": get_safety_control().status(),
            "dry_run": cfg.actuator.dry_run,
        }
    )
    return 0


def cmd_ui_correct(
    config_path: str | None,
    *,
    goal: str,
    reject: str | None,
    prefer: str | None,
    log_level: str | None,
) -> int:
    cfg = _setup_from_args(config_path, log_level)
    from frontend.corrections import apply_corrections_to_goal, apply_corrections_to_vision
    from frontend.session import UISession

    session = UISession(cfg, mock=True)
    # Seed vision via refresh
    session.refresh_observe(goal)
    vision = session.get_last_vision()
    before = len(vision.elements) if vision else 0
    if reject:
        session.reject_element(reject)
    elif vision and vision.elements:
        # Default: reject first non-preferred element for demo
        session.reject_element(vision.elements[0].element_id, note="不是这个")
    if prefer:
        session.prefer_element(prefer)
    elif vision and len(vision.elements) > 1:
        session.prefer_element(vision.elements[min(1, len(vision.elements) - 1)].element_id)
    effective = apply_corrections_to_goal(goal, session.corrections.to_list())
    after_vision = (
        apply_corrections_to_vision(vision, session.corrections.to_list()) if vision else None
    )
    _print_json(
        {
            "goal": goal,
            "effective_goal": effective,
            "corrections": session.corrections.log_summary(),
            "elements_before": before,
            "elements_after": len(after_vision.elements) if after_vision else 0,
            "element_ids_after": [
                e.element_id for e in (after_vision.elements if after_vision else [])
            ][:16],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        return cmd_version()
    if args.command == "config-show":
        return cmd_config_show(args.config)
    if args.command == "demo":
        level = getattr(args, "demo_log_level", None) or args.log_level
        return cmd_demo(args.config, args.goal, args.json_out, level)
    if args.command == "health":
        return cmd_health(args.config, args.http)
    if args.command == "capture":
        level = getattr(args, "cap_log_level", None) or args.log_level
        if args.capture_cmd == "monitors":
            return cmd_capture_monitors(args.config, level)
        if args.capture_cmd == "once":
            return cmd_capture_once(
                args.config,
                mode=args.mode,
                kind=args.kind,
                out=args.out,
                window_title=args.window_title,
                region=args.region,
                log_level=level,
            )
        if args.capture_cmd == "stream":
            return cmd_capture_stream(args.config, args.seconds, level)
    if args.command == "vision":
        level = getattr(args, "vis_log_level", None) or args.log_level
        if args.vision_cmd == "once":
            return cmd_vision_once(
                args.config,
                backend=args.backend,
                mode=args.mode,
                window_title=args.window_title,
                region=args.region,
                goal=args.goal,
                annotate=args.annotate,
                json_out=args.json_out,
                log_level=level,
            )
        if args.vision_cmd == "context":
            return cmd_vision_context(
                args.config,
                backend=args.backend,
                goal=args.goal,
                log_level=level,
            )
    if args.command == "infer":
        level = getattr(args, "inf_log_level", None) or args.log_level
        if args.infer_cmd == "info":
            return cmd_infer_info(args.config, args.json_out, level)
        if args.infer_cmd == "prompts":
            return cmd_infer_prompts(args.config, level)
        if args.infer_cmd == "server":
            return cmd_infer_server(args.config, args.action, level)
        if args.infer_cmd == "once":
            return cmd_infer_once(
                args.config,
                backend=args.backend,
                vision_backend=args.vision_backend,
                goal=args.goal,
                mode=args.mode,
                no_image=args.no_image,
                stream=args.stream,
                start_server=args.start_server,
                json_out=args.json_out,
                log_level=level,
            )
    if args.command == "agent":
        level = getattr(args, "ag_log_level", None) or args.log_level
        if args.agent_cmd == "states":
            return cmd_agent_states(args.config, level)
        if args.agent_cmd == "preview":
            return cmd_agent_preview(
                args.config,
                goal=args.goal,
                mock=bool(getattr(args, "mock", True)),
                json_out=args.json_out,
                log_level=level,
            )
        if args.agent_cmd == "run":
            dry = True if args.dry_run else None
            return cmd_agent_run(
                args.config,
                goal=args.goal,
                yes=bool(args.yes),
                preview_only=bool(args.preview_only),
                dry_run=dry,
                live=bool(args.live),
                mock=bool(args.mock),
                actuator=args.actuator,
                max_steps=args.max_steps,
                json_out=args.json_out,
                log_level=level,
            )
    if args.command == "safety":
        level = getattr(args, "saf_log_level", None) or args.log_level
        if args.safety_cmd == "status":
            return cmd_safety_status(args.config, level)
        if args.safety_cmd == "pause":
            return cmd_safety_control(args.config, "pause", getattr(args, "reason", ""))
        if args.safety_cmd == "resume":
            return cmd_safety_control(args.config, "resume")
        if args.safety_cmd == "stop":
            return cmd_safety_control(args.config, "stop", getattr(args, "reason", ""))
        if args.safety_cmd == "reset":
            return cmd_safety_control(args.config, "reset")
        if args.safety_cmd == "check":
            return cmd_safety_check(
                args.config,
                goal=args.goal,
                action=args.action,
                text=args.text or "",
                element_id=args.element_id,
                risk=args.risk,
                screen_text=args.screen_text or "",
                log_level=level,
            )
        if args.safety_cmd == "redact":
            return cmd_safety_redact(args.text)
        if args.safety_cmd == "threats":
            return cmd_safety_threats()
        if args.safety_cmd == "audit":
            return cmd_safety_audit(
                args.config,
                args.audit_cmd,
                days=getattr(args, "days", None),
                wipe=bool(getattr(args, "wipe", False)),
            )
    if args.command == "ui":
        level = getattr(args, "ui_log_level", None) or args.log_level
        if args.ui_cmd == "open":
            return cmd_ui_open(args.config, live=bool(args.live), log_level=level)
        if args.ui_cmd == "run":
            return cmd_ui_run(
                args.config,
                goal=args.goal,
                yes=bool(args.yes),
                preview_only=bool(args.preview_only),
                mock=bool(getattr(args, "mock", True)),
                reject=getattr(args, "reject", None),
                prefer=getattr(args, "prefer", None),
                json_out=args.json_out,
                log_level=level,
            )
        if args.ui_cmd == "status":
            return cmd_ui_status(args.config, level)
        if args.ui_cmd == "correct":
            return cmd_ui_correct(
                args.config,
                goal=args.goal,
                reject=getattr(args, "reject", None),
                prefer=getattr(args, "prefer", None),
                log_level=level,
            )
    parser.error(f"unknown command {args.command}")
    return 2


def demo_main() -> None:
    raise SystemExit(main(["demo"]))


if __name__ == "__main__":
    raise SystemExit(main())
