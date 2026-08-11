"""Structured logging with trace_id correlation."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_TRACE = threading.local()
_CONFIGURED = False
_LOCK = threading.Lock()


def new_trace_id() -> str:
    return f"tr-{uuid4().hex[:16]}"


def set_trace_id(trace_id: str | None) -> None:
    _TRACE.trace_id = trace_id


def get_trace_id() -> str | None:
    return getattr(_TRACE, "trace_id", None)


@contextmanager
def trace_scope(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace_id for the current thread for the duration of the block."""
    tid = trace_id or new_trace_id()
    prev = get_trace_id()
    set_trace_id(tid)
    try:
        yield tid
    finally:
        set_trace_id(prev)


class TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "-"  # type: ignore[attr-defined]
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        # Extra structured fields (avoid reserved LogRecord attrs).
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "trace_id",
            "asctime",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(
    *,
    level: str = "INFO",
    json_logs: bool = True,
    log_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> None:
    """Idempotent root logging setup for the process."""
    global _CONFIGURED
    with _LOCK:
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(getattr(logging, level.upper(), logging.INFO))

        trace_filter = TraceFilter()
        formatter: logging.Formatter = JsonFormatter() if json_logs else TextFormatter()

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        stream.addFilter(trace_filter)
        root.addHandler(stream)

        if log_dir:
            base = Path(log_dir)
            if not base.is_absolute() and project_root is not None:
                base = project_root / base
            base.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                base / "baodou.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(trace_filter)
            root.addHandler(file_handler)

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        # Sensible default for imports/tests before explicit setup.
        setup_logging(level="INFO", json_logs=False, log_dir=None)
    return logging.getLogger(name)


# Keys that conflict with logging.LogRecord / Formatter internals.
_RESERVED_LOG_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


def _safe_extra(fields: dict[str, Any]) -> dict[str, Any]:
    """Rename reserved LogRecord keys so logger.extra accepts them."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if key == "event":
            continue
        if key in _RESERVED_LOG_KEYS:
            out[f"field_{key}"] = value
        else:
            out[key] = value
    return out


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured event line (kind + fields)."""
    extra = _safe_extra(fields)
    extra["event"] = event
    logger.log(level, event, extra=extra)
