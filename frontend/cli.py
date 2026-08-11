"""CLI entry: config load, mock pipeline demo, graceful cancel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.cancel import get_global_token, install_signal_handlers
from core.config import PROJECT_ROOT, config_to_safe_dict, load_config
from core.logging import get_logger, setup_logging
from core.pipeline import MockPipeline

_log = get_logger("frontend.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="baodou",
        description="baodou — local AI desktop assistant (Phase B skeleton)",
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
    }
    if probe_http or cfg.inference.backend == "http":
        from inference.http_client import HttpInference

        http = HttpInference(cfg)
        checks["llama_server_health"] = http.health()
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks.get("config_ok") else 1


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
    parser.error(f"unknown command {args.command}")
    return 2


def demo_main() -> None:
    raise SystemExit(main(["demo"]))


if __name__ == "__main__":
    raise SystemExit(main())
