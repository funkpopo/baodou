"""Global cancellation token and graceful shutdown."""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable

from core.errors import CancelledError
from core.logging import get_logger

_log = get_logger("core.cancel")

_OnCancel = Callable[[], None]


class CancellationToken:
    """Thread-safe cancel flag with optional wait/callback."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._event = threading.Event()
        self._callbacks: list[_OnCancel] = []
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            callbacks = list(self._callbacks)
        _log.warning(
            "cancel_requested", extra={"event": "cancel", "reason": reason, "token": self.name}
        )
        for cb in callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001 — shutdown path must not raise
                _log.exception("cancel_callback_failed", extra={"event": "cancel_callback_failed"})

    def reset(self) -> None:
        with self._lock:
            self._event.clear()
            self._reason = None

    def check(self) -> None:
        """Raise CancelledError if cancelled."""
        if self.is_cancelled:
            raise CancelledError(self._reason or "操作已取消")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or timeout. Returns True if cancelled."""
        return self._event.wait(timeout)

    def on_cancel(self, callback: _OnCancel) -> None:
        with self._lock:
            self._callbacks.append(callback)
            already = self._event.is_set()
        if already:
            callback()

    def sleep(self, seconds: float, *, slice_sec: float = 0.05) -> None:
        """Interruptible sleep; raises CancelledError if cancelled mid-wait."""
        end = time.monotonic() + seconds
        while True:
            self.check()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            self._event.wait(min(slice_sec, remaining))


# Process-wide token used by pipeline and CLI.
_GLOBAL = CancellationToken(name="global")
_HANDLERS_INSTALLED = False
_HANDLER_LOCK = threading.Lock()


def get_global_token() -> CancellationToken:
    return _GLOBAL


def install_signal_handlers(
    token: CancellationToken | None = None,
    *,
    graceful_sec: float = 5.0,
) -> None:
    """Install SIGINT/SIGTERM handlers that cancel then force-exit on second signal.

    On Windows, SIGTERM may be limited; SIGINT (Ctrl+C) is the primary path.
    """
    global _HANDLERS_INSTALLED
    tok = token or _GLOBAL

    def _handler(signum: int, _frame: object | None) -> None:
        sig_name = (
            signal.Signals(signum).name
            if signum in signal.Signals._value2member_map_
            else str(signum)
        )
        if tok.is_cancelled:
            _log.error(
                "force_exit",
                extra={"event": "force_exit", "signal": sig_name, "graceful_sec": graceful_sec},
            )
            raise SystemExit(130)
        tok.cancel(reason=f"signal:{sig_name}")
        _log.info(
            "graceful_shutdown_started",
            extra={"event": "graceful_shutdown", "signal": sig_name, "graceful_sec": graceful_sec},
        )

    with _HANDLER_LOCK:
        if _HANDLERS_INSTALLED:
            return
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            # May fail if not main thread.
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, _handler)
        _HANDLERS_INSTALLED = True
