"""Windows mouse/keyboard injection via SendInput (Phase F).

Default path is still dry-run: no OS events unless ``actuator.dry_run=false``.
Uses only stdlib ``ctypes`` (no pyautogui dependency).
"""

from __future__ import annotations

import sys
import time
from typing import Any

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import ActuatorError, ErrorCode
from core.logging import get_logger, log_event
from core.models import (
    ActionResult,
    ActionStep,
    ActionType,
    Point,
    UIElement,
    UIVisionResult,
    VerificationResult,
)

from actuator.base import ActuatorBackend
from actuator.rate_limit import ActionRateLimiter
from actuator.relocate import RelocateResult, relocate_target
from actuator.verify import verify_step

_log = get_logger("actuator.win")

# Virtual-key map for common keys / hotkeys
_VK: dict[str, int] = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "pause": 0x13,
    "caps": 0x14,
    "escape": 0x1B,
    "esc": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "win": 0x5B,
    "cmd": 0x5B,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


def _require_windows() -> Any:
    if sys.platform != "win32":
        raise ActuatorError(
            "WinActuator 仅支持 Windows",
            code=ErrorCode.NOT_IMPLEMENTED,
        )
    import ctypes
    from ctypes import wintypes

    return ctypes, wintypes


class WinActuator(ActuatorBackend):
    """Real input injector with dry-run gate + relocate + rate limit."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._history: list[ActionResult] = []
        self._limiter = ActionRateLimiter(config.actuator.max_actions_per_minute)
        self._prior_elements: dict[str, UIElement] = {}
        self._screen_w = 0
        self._screen_h = 0
        self._refresh_screen_size()

    def _refresh_screen_size(self) -> None:
        try:
            ctypes, _ = _require_windows()
            user32 = ctypes.windll.user32
            self._screen_w = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
            self._screen_h = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
            if self._screen_w <= 0 or self._screen_h <= 0:
                self._screen_w = int(user32.GetSystemMetrics(0))
                self._screen_h = int(user32.GetSystemMetrics(1))
        except Exception:  # noqa: BLE001
            self._screen_w = 1920
            self._screen_h = 1080

    # ------------------------------------------------------------------ public
    def execute(
        self,
        step: ActionStep,
        vision: UIVisionResult,
        *,
        trace_id: str = "",
        coordinate_confirmed: bool = False,
        prior_element: UIElement | None = None,
    ) -> ActionResult:
        get_global_token().check()
        t0 = time.perf_counter()
        self._limiter.check()

        if step.action == ActionType.NONE:
            return self._finish(
                step,
                True,
                "no-op",
                t0,
                trace_id=trace_id,
            )

        if step.action == ActionType.WAIT:
            ms = step.wait_ms or min(step.timeout_ms, 5000)
            if not self.config.actuator.dry_run:
                time.sleep(ms / 1000.0)
            return self._finish(
                step,
                True,
                f"wait {ms}ms",
                t0,
                trace_id=trace_id,
                dry_hint=True,
            )

        if step.action == ActionType.REIDENTIFY:
            return self._finish(
                step,
                True,
                "reidentify marker (agent handles capture)",
                t0,
                trace_id=trace_id,
            )

        prior = prior_element
        if prior is None and step.target_element_id:
            prior = self._prior_elements.get(step.target_element_id)

        try:
            located = relocate_target(
                step,
                vision,
                self.config,
                prior_element=prior,
                coordinate_confirmed=coordinate_confirmed,
            )
        except ActuatorError:
            raise

        if located.element is not None:
            self._prior_elements[located.element.element_id] = located.element
            if step.target_element_id and step.target_element_id != located.element.element_id:
                self._prior_elements[step.target_element_id] = located.element

        point = located.point
        if point is not None and self.config.actuator.bound_check:
            self._check_bounds(point)

        dry = self.config.actuator.dry_run
        try:
            if dry:
                msg = self._dry_message(step, located)
            else:
                msg = self._inject(step, located)
                self._limiter.record()
        except ActuatorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ActuatorError(
                f"输入注入失败: {exc}",
                code=ErrorCode.ACTION_FAILED,
                cause=exc,
            ) from exc

        return self._finish(
            step,
            True,
            msg,
            t0,
            trace_id=trace_id,
            point=point,
            element_id=located.element_id,
            relocated=located.relocated,
        )

    def verify(
        self,
        step: ActionStep,
        action: ActionResult,
        *,
        trace_id: str = "",
        vision_before: UIVisionResult | None = None,
        vision_after: UIVisionResult | None = None,
        change_score: float | None = None,
        frame_id_before: str | None = None,
        frame_id_after: str | None = None,
    ) -> VerificationResult:
        get_global_token().check()
        return verify_step(
            step,
            action,
            config=self.config,
            vision_before=vision_before,
            vision_after=vision_after,
            change_score=change_score,
            frame_id_before=frame_id_before,
            frame_id_after=frame_id_after,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------ inject
    def _inject(self, step: ActionStep, located: RelocateResult) -> str:
        ctypes, wintypes = _require_windows()
        action = step.action
        point = located.point

        if action == ActionType.MOVE:
            assert point is not None
            self._mouse_move(point.x, point.y)
            return f"move to ({point.x},{point.y})"

        if action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
            assert point is not None
            self._mouse_move(point.x, point.y)
            time.sleep(self.config.actuator.click_delay_ms / 1000.0)
            if action == ActionType.CLICK:
                self._mouse_click(left=True)
                return f"click ({point.x},{point.y})"
            if action == ActionType.DOUBLE_CLICK:
                self._mouse_click(left=True)
                time.sleep(0.05)
                self._mouse_click(left=True)
                return f"double_click ({point.x},{point.y})"
            self._mouse_click(left=False)
            return f"right_click ({point.x},{point.y})"

        if action == ActionType.DRAG:
            assert point is not None
            end = step.end_point
            if end is None:
                raise ActuatorError("drag 需要 end_point", code=ErrorCode.TARGET_INVALID)
            self._mouse_move(point.x, point.y)
            self._mouse_down(left=True)
            time.sleep(0.02)
            self._mouse_move(end.x, end.y)
            time.sleep(0.02)
            self._mouse_up(left=True)
            return f"drag ({point.x},{point.y})→({end.x},{end.y})"

        if action == ActionType.SCROLL:
            if point is not None:
                self._mouse_move(point.x, point.y)
            self._mouse_wheel(step.scroll_dy, horizontal=False)
            if step.scroll_dx:
                self._mouse_wheel(step.scroll_dx, horizontal=True)
            return f"scroll dx={step.scroll_dx} dy={step.scroll_dy}"

        if action == ActionType.TYPE:
            if point is not None:
                self._mouse_move(point.x, point.y)
                self._mouse_click(left=True)
                time.sleep(0.05)
            text = step.text or ""
            self._type_text(text)
            return f"type {len(text)} chars"

        if action in (ActionType.KEY, ActionType.HOTKEY):
            self._hotkey(step.keys)
            return f"keys {'+'.join(step.keys)}"

        raise ActuatorError(
            f"不支持的动作: {action.value}",
            code=ErrorCode.NOT_IMPLEMENTED,
        )

    def _dry_message(self, step: ActionStep, located: RelocateResult) -> str:
        pt = located.point
        pt_s = f"({pt.x},{pt.y})" if pt else "n/a"
        return (
            f"[dry_run] {step.action.value} target={located.element_id or step.target_element_id} "
            f"point={pt_s} method={located.method} text={step.text!r} keys={step.keys}"
        )

    # ------------------------------------------------------------------ low-level win32
    def _check_bounds(self, point: Point) -> None:
        # Virtual desktop can have negative origins; only reject absurd values.
        if self._screen_w <= 0:
            self._refresh_screen_size()
        # Soft check: allow multi-monitor negative coords; reject huge outliers.
        if abs(point.x) > 100_000 or abs(point.y) > 100_000:
            raise ActuatorError(
                f"坐标异常: ({point.x},{point.y})",
                code=ErrorCode.COORDINATE_OUT_OF_BOUNDS,
            )

    def _to_absolute(self, x: int, y: int) -> tuple[int, int]:
        """Convert physical pixel to SendInput absolute 0..65535 over virtual screen."""
        ctypes, _ = _require_windows()
        user32 = ctypes.windll.user32
        left = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
        top = int(user32.GetSystemMetrics(77))
        width = int(user32.GetSystemMetrics(78)) or 1
        height = int(user32.GetSystemMetrics(79)) or 1
        ax = int(round((x - left) * 65535 / max(width - 1, 1)))
        ay = int(round((y - top) * 65535 / max(height - 1, 1)))
        return max(0, min(65535, ax)), max(0, min(65535, ay))

    def _send_input(self, *inputs: Any) -> None:
        ctypes, wintypes = _require_windows()

        # Build INPUT structures dynamically to avoid layout issues across Python builds.
        ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]

        arr = (INPUT * len(inputs))()
        for i, spec in enumerate(inputs):
            kind = spec["type"]
            arr[i].type = kind
            if kind == 0:  # mouse
                mi = spec["mi"]
                arr[i].union.mi = MOUSEINPUT(
                    mi.get("dx", 0),
                    mi.get("dy", 0),
                    mi.get("data", 0),
                    mi.get("flags", 0),
                    0,
                    0,
                )
            elif kind == 1:  # keyboard
                ki = spec["ki"]
                arr[i].union.ki = KEYBDINPUT(
                    ki.get("vk", 0),
                    ki.get("scan", 0),
                    ki.get("flags", 0),
                    0,
                    0,
                )
        sent = ctypes.windll.user32.SendInput(len(inputs), ctypes.byref(arr), ctypes.sizeof(INPUT))
        if sent != len(inputs):
            raise ActuatorError(
                f"SendInput 部分失败: {sent}/{len(inputs)}",
                code=ErrorCode.ACTION_FAILED,
            )

    def _mouse_move(self, x: int, y: int) -> None:
        ax, ay = self._to_absolute(x, y)
        # MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        flags = 0x0001 | 0x8000 | 0x4000
        self._send_input({"type": 0, "mi": {"dx": ax, "dy": ay, "flags": flags}})
        dur = self.config.actuator.move_duration_ms / 1000.0
        if dur > 0:
            time.sleep(min(dur, 0.2))

    def _mouse_down(self, *, left: bool = True) -> None:
        flag = 0x0002 if left else 0x0008  # LEFTDOWN / RIGHTDOWN
        self._send_input({"type": 0, "mi": {"flags": flag}})

    def _mouse_up(self, *, left: bool = True) -> None:
        flag = 0x0004 if left else 0x0010  # LEFTUP / RIGHTUP
        self._send_input({"type": 0, "mi": {"flags": flag}})

    def _mouse_click(self, *, left: bool = True) -> None:
        self._mouse_down(left=left)
        time.sleep(0.015)
        self._mouse_up(left=left)

    def _mouse_wheel(self, delta_notches: int, *, horizontal: bool = False) -> None:
        if delta_notches == 0:
            return
        # WHEEL_DELTA = 120
        data = int(delta_notches) * 120
        flag = 0x0800 if not horizontal else 0x1000  # WHEEL / HWHEEL
        # data is unsigned in mouseData; negative via two's complement
        data_u = data & 0xFFFFFFFF
        self._send_input({"type": 0, "mi": {"data": data_u, "flags": flag}})

    def _key_event(self, vk: int, *, up: bool = False) -> None:
        flags = 0x0002 if up else 0  # KEYEVENTF_KEYUP
        self._send_input({"type": 1, "ki": {"vk": vk, "flags": flags}})

    def _type_text(self, text: str) -> None:
        ctypes, _ = _require_windows()
        interval = self.config.actuator.type_interval_ms / 1000.0
        for ch in text:
            get_global_token().check()
            scan = ctypes.windll.user32.VkKeyScanW(ord(ch))
            if scan == -1:
                # Unicode fallback via KEYEVENTF_UNICODE
                self._send_input({"type": 1, "ki": {"vk": 0, "scan": ord(ch), "flags": 0x0004}})
                self._send_input(
                    {
                        "type": 1,
                        "ki": {"vk": 0, "scan": ord(ch), "flags": 0x0004 | 0x0002},
                    }
                )
            else:
                vk = scan & 0xFF
                shift = bool(scan & 0x100)
                if shift:
                    self._key_event(0x10, up=False)
                self._key_event(vk, up=False)
                self._key_event(vk, up=True)
                if shift:
                    self._key_event(0x10, up=True)
            if interval > 0:
                time.sleep(interval)

    def _hotkey(self, keys: list[str]) -> None:
        if not keys:
            raise ActuatorError("hotkey/key 需要 keys", code=ErrorCode.TARGET_INVALID)
        vks: list[int] = []
        for k in keys:
            name = k.strip().lower()
            if name in _VK:
                vks.append(_VK[name])
            elif len(name) == 1:
                ch = name.upper()
                vks.append(ord(ch))
            else:
                raise ActuatorError(f"未知按键: {k}", code=ErrorCode.TARGET_INVALID)
        for vk in vks:
            self._key_event(vk, up=False)
        for vk in reversed(vks):
            self._key_event(vk, up=True)

    # ------------------------------------------------------------------ helpers
    def _finish(
        self,
        step: ActionStep,
        success: bool,
        message: str,
        t0: float,
        *,
        trace_id: str,
        point: Point | None = None,
        element_id: str | None = None,
        relocated: bool = False,
        dry_hint: bool = False,
    ) -> ActionResult:
        result = ActionResult(
            trace_id=trace_id,
            step_id=step.step_id,
            action=step.action,
            success=success,
            dry_run=self.config.actuator.dry_run or dry_hint,
            message=message,
            latency_ms=(time.perf_counter() - t0) * 1000,
            resolved_point=point,
            resolved_element_id=element_id or step.target_element_id,
            relocated=relocated,
        )
        self._history.append(result)
        log_event(_log, "actuator.execute", **result.log_summary())
        return result
