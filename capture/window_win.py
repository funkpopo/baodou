"""Windows window enumeration / bounds (ctypes, no extra deps)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from core.errors import CaptureError, ErrorCode
from core.models import BBox, WindowInfo

user32 = ctypes.windll.user32 if sys.platform == "win32" else None


def _get_window_text(hwnd: int) -> str:
    if user32 is None:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_class_name(hwnd: int) -> str:
    if user32 is None:
        return ""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _get_rect(hwnd: int) -> BBox:
    if user32 is None:
        return BBox(x=0, y=0, width=0, height=0)
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise CaptureError(f"GetWindowRect 失败 hwnd={hwnd}", code=ErrorCode.CAPTURE_FAILED)
    return BBox(
        x=int(rect.left),
        y=int(rect.top),
        width=int(rect.right - rect.left),
        height=int(rect.bottom - rect.top),
    )


def list_top_level_windows(*, visible_only: bool = True) -> list[WindowInfo]:
    if user32 is None:
        return []
    results: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):  # type: ignore[no-untyped-def]
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_text(int(hwnd))
        if not title:
            return True
        try:
            box = _get_rect(int(hwnd))
        except CaptureError:
            return True
        if box.width <= 0 or box.height <= 0:
            return True
        results.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                class_name=_get_class_name(int(hwnd)),
                left=box.x,
                top=box.y,
                width=box.width,
                height=box.height,
                visible=bool(user32.IsWindowVisible(hwnd)),
            )
        )
        return True

    user32.EnumWindows(_enum, 0)
    return results


def find_window(
    *,
    title_substr: str | None = None,
    hwnd: int | None = None,
) -> WindowInfo:
    if user32 is None:
        raise CaptureError("窗口采集仅支持 Windows", code=ErrorCode.CAPTURE_FAILED)
    if hwnd is not None:
        if not user32.IsWindow(hwnd):
            raise CaptureError(f"无效 hwnd={hwnd}", code=ErrorCode.TARGET_INVALID)
        box = _get_rect(int(hwnd))
        return WindowInfo(
            hwnd=int(hwnd),
            title=_get_window_text(int(hwnd)),
            class_name=_get_class_name(int(hwnd)),
            left=box.x,
            top=box.y,
            width=box.width,
            height=box.height,
            visible=bool(user32.IsWindowVisible(hwnd)),
        )
    if not title_substr:
        raise CaptureError("需要 window_title 或 window_hwnd", code=ErrorCode.CAPTURE_FAILED)
    needle = title_substr.lower()
    matches = [w for w in list_top_level_windows() if needle in w.title.lower()]
    if not matches:
        raise CaptureError(
            f"未找到标题包含 {title_substr!r} 的窗口",
            code=ErrorCode.TARGET_INVALID,
        )
    # Prefer largest visible area (likely main window)
    matches.sort(key=lambda w: w.width * w.height, reverse=True)
    return matches[0]


def find_privacy_windows(title_keywords: list[str]) -> list[WindowInfo]:
    if not title_keywords:
        return []
    keys = [k.lower() for k in title_keywords if k]
    hits: list[WindowInfo] = []
    for w in list_top_level_windows(visible_only=True):
        t = w.title.lower()
        if any(k in t for k in keys):
            hits.append(w)
    return hits
