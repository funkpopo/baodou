"""CLI entry: config, mock demo, capture pipeline (Phase C)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from core.cancel import get_global_token, install_signal_handlers
from core.config import PROJECT_ROOT, config_to_safe_dict, load_config
from core.logging import get_logger, setup_logging
from core.pipeline import MockPipeline

_log = get_logger("frontend.cli")


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
    parser.error(f"unknown command {args.command}")
    return 2


def demo_main() -> None:
    raise SystemExit(main(["demo"]))


if __name__ == "__main__":
    raise SystemExit(main())
