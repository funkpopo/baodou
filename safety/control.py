"""Global pause, emergency stop, and focus-loss pause (Phase G).

Hotkeys are exposed as API hooks; actual OS global hooks are optional and
best-effort on Windows. CLI / UI call request_pause / request_stop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.cancel import CancellationToken, get_global_token
from core.errors import ErrorCode, SafetyError
from core.logging import get_logger, log_event

_log = get_logger("safety.control")


class ControlState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyControl:
    """Process-wide safety control plane (thread-safe)."""

    state: ControlState = ControlState.RUNNING
    reason: str = ""
    pause_on_focus_loss: bool = False
    emergency_stop_enabled: bool = True
    owner_pid: int | None = None
    last_focus_hwnd: int | None = None
    changed_at: float = field(default_factory=time.monotonic)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "reason": self.reason,
                "pause_on_focus_loss": self.pause_on_focus_loss,
                "emergency_stop_enabled": self.emergency_stop_enabled,
                "changed_at_mono": self.changed_at,
            }

    def is_running(self) -> bool:
        with self._lock:
            return self.state == ControlState.RUNNING

    def is_stopped(self) -> bool:
        with self._lock:
            return self.state == ControlState.EMERGENCY_STOP

    def is_paused(self) -> bool:
        with self._lock:
            return self.state == ControlState.PAUSED

    def request_pause(self, reason: str = "user_pause") -> None:
        with self._lock:
            if self.state == ControlState.EMERGENCY_STOP:
                return
            self.state = ControlState.PAUSED
            self.reason = reason
            self.changed_at = time.monotonic()
        log_event(_log, "safety.pause", reason=reason)

    def request_resume(self, reason: str = "user_resume") -> None:
        with self._lock:
            if self.state == ControlState.EMERGENCY_STOP:
                raise SafetyError(
                    "紧急停止后不可直接恢复，请 reset_stop()",
                    code=ErrorCode.PERMISSION_DENIED,
                    details={"state": self.state.value},
                )
            self.state = ControlState.RUNNING
            self.reason = reason
            self.changed_at = time.monotonic()
        log_event(_log, "safety.resume", reason=reason)

    def request_stop(self, reason: str = "emergency_stop", *, token: CancellationToken | None = None) -> None:
        if not self.emergency_stop_enabled:
            log_event(_log, "safety.stop_ignored", reason="emergency_stop_disabled")
            return
        with self._lock:
            self.state = ControlState.EMERGENCY_STOP
            self.reason = reason
            self.changed_at = time.monotonic()
        tok = token or get_global_token()
        tok.cancel(reason=reason)
        log_event(_log, "safety.emergency_stop", reason=reason)

    def reset_stop(self, reason: str = "reset") -> None:
        """Clear emergency stop and cancel token for a new session."""
        with self._lock:
            self.state = ControlState.RUNNING
            self.reason = reason
            self.changed_at = time.monotonic()
        tok = get_global_token()
        tok.reset()
        log_event(_log, "safety.reset_stop", reason=reason)

    def check(self) -> None:
        """Raise if paused or emergency-stopped. Call before each action."""
        with self._lock:
            state = self.state
            reason = self.reason
        if state == ControlState.EMERGENCY_STOP:
            raise SafetyError(
                f"紧急停止: {reason or 'emergency_stop'}",
                code=ErrorCode.CANCELLED,
                details={"control": state.value, "reason": reason},
            )
        if state == ControlState.PAUSED:
            raise SafetyError(
                f"已暂停: {reason or 'paused'}",
                code=ErrorCode.CONFIRMATION_REQUIRED,
                details={"control": state.value, "reason": reason},
            )

    def poll_focus_loss(self) -> bool:
        """
        Best-effort: if pause_on_focus_loss and foreground window leaves our
        process, auto-pause. Returns True if a pause was triggered.
        """
        if not self.pause_on_focus_loss:
            return False
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = int(user32.GetForegroundWindow())
            if hwnd == 0:
                return False
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            import os

            my_pid = os.getpid()
            with self._lock:
                prev = self.last_focus_hwnd
                self.last_focus_hwnd = hwnd
                # First sample: just record
                if prev is None:
                    self.owner_pid = my_pid
                    return False
                fg_pid = int(pid.value)
            # If focus moved to another process, pause
            if fg_pid not in (0, my_pid) and self.is_running():
                self.request_pause(reason="focus_loss")
                return True
        except Exception:  # noqa: BLE001 — never crash control plane
            return False
        return False


_GLOBAL_CONTROL: SafetyControl | None = None
_CONTROL_LOCK = threading.Lock()


def get_safety_control() -> SafetyControl:
    global _GLOBAL_CONTROL
    with _CONTROL_LOCK:
        if _GLOBAL_CONTROL is None:
            _GLOBAL_CONTROL = SafetyControl()
        return _GLOBAL_CONTROL


def reset_safety_control(**kwargs: Any) -> SafetyControl:
    """Replace global control (tests)."""
    global _GLOBAL_CONTROL
    with _CONTROL_LOCK:
        _GLOBAL_CONTROL = SafetyControl(**kwargs)
        return _GLOBAL_CONTROL
