"""Multi-monitor geometry, DPI scale, and coordinate conversion (Windows-first)."""

from __future__ import annotations

import contextlib
import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from core.errors import CaptureError, ErrorCode
from core.models import BBox, DisplayOrientation, MonitorInfo, Point

# Windows display orientation constants (DEVMODE)
_DMDO_DEFAULT = 0
_DMDO_90 = 1
_DMDO_180 = 2
_MDDO_270 = 3

_DPI_AWARE_SET = False


def ensure_dpi_awareness() -> None:
    """Best-effort per-monitor DPI awareness so physical pixels match mouse coords."""
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET or sys.platform != "win32":
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE_V2 = 2 (Windows 10 1703+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        with contextlib.suppress(Exception):
            ctypes.windll.user32.SetProcessDPIAware()
    _DPI_AWARE_SET = True


def _orientation_from_dm(value: int) -> DisplayOrientation:
    mapping = {
        _DMDO_DEFAULT: DisplayOrientation.ROT0,
        _DMDO_90: DisplayOrientation.ROT90,
        _DMDO_180: DisplayOrientation.ROT180,
        _MDDO_270: DisplayOrientation.ROT270,
    }
    return mapping.get(value, DisplayOrientation.ROT0)


def _monitor_dpi(hmonitor: int) -> tuple[int, int]:
    if sys.platform != "win32":
        return 96, 96
    try:
        dpi_x = wintypes.UINT()
        dpi_y = wintypes.UINT()
        # MDT_EFFECTIVE_DPI = 0
        hr = ctypes.windll.shcore.GetDpiForMonitor(
            hmonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)
        )
        if hr == 0:
            return int(dpi_x.value), int(dpi_y.value)
    except Exception:  # noqa: BLE001
        pass
    try:
        dc = ctypes.windll.user32.GetDC(0)
        x = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
        y = ctypes.windll.gdi32.GetDeviceCaps(dc, 90)  # LOGPIXELSY
        ctypes.windll.user32.ReleaseDC(0, dc)
        if x > 0 and y > 0:
            return int(x), int(y)
    except Exception:  # noqa: BLE001
        pass
    return 96, 96


def _display_orientation(device_name: str) -> DisplayOrientation:
    if sys.platform != "win32" or not device_name:
        return DisplayOrientation.ROT0
    try:

        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ("dmDeviceName", wintypes.WCHAR * 32),
                ("dmSpecVersion", wintypes.WORD),
                ("dmDriverVersion", wintypes.WORD),
                ("dmSize", wintypes.WORD),
                ("dmDriverExtra", wintypes.WORD),
                ("dmFields", wintypes.DWORD),
                ("dmOrientation", wintypes.SHORT),
                ("dmPaperSize", wintypes.SHORT),
                ("dmPaperLength", wintypes.SHORT),
                ("dmPaperWidth", wintypes.SHORT),
                ("dmScale", wintypes.SHORT),
                ("dmCopies", wintypes.SHORT),
                ("dmDefaultSource", wintypes.SHORT),
                ("dmPrintQuality", wintypes.SHORT),
                ("dmColor", wintypes.SHORT),
                ("dmDuplex", wintypes.SHORT),
                ("dmYResolution", wintypes.SHORT),
                ("dmTTOption", wintypes.SHORT),
                ("dmCollate", wintypes.SHORT),
                ("dmFormName", wintypes.WCHAR * 32),
                ("dmLogPixels", wintypes.WORD),
                ("dmBitsPerPel", wintypes.DWORD),
                ("dmPelsWidth", wintypes.DWORD),
                ("dmPelsHeight", wintypes.DWORD),
                ("dmDisplayFlags", wintypes.DWORD),
                ("dmDisplayFrequency", wintypes.DWORD),
                ("dmICMMethod", wintypes.DWORD),
                ("dmICMIntent", wintypes.DWORD),
                ("dmMediaType", wintypes.DWORD),
                ("dmDitherType", wintypes.DWORD),
                ("dmReserved1", wintypes.DWORD),
                ("dmReserved2", wintypes.DWORD),
                ("dmPanningWidth", wintypes.DWORD),
                ("dmPanningHeight", wintypes.DWORD),
                ("dmPosition_x", ctypes.c_long),
                ("dmPosition_y", ctypes.c_long),
                ("dmDisplayOrientation", wintypes.DWORD),
                ("dmDisplayFixedOutput", wintypes.DWORD),
            ]

        devmode = DEVMODE()
        devmode.dmSize = ctypes.sizeof(DEVMODE)
        # ENUM_CURRENT_SETTINGS = -1
        if ctypes.windll.user32.EnumDisplaySettingsW(device_name, -1, ctypes.byref(devmode)):
            return _orientation_from_dm(int(devmode.dmDisplayOrientation))
    except Exception:  # noqa: BLE001
        pass
    return DisplayOrientation.ROT0


@dataclass(frozen=True)
class CaptureRegion:
    """Rectangle to grab, in virtual-desktop physical pixels."""

    left: int
    top: int
    width: int
    height: int
    mon_index: int = 1
    dpi_scale: float = 1.0
    dpi_x: int = 96
    dpi_y: int = 96
    orientation: DisplayOrientation = DisplayOrientation.ROT0
    is_primary: bool = False
    name: str = ""

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    def as_bbox(self) -> BBox:
        return BBox(x=self.left, y=self.top, width=self.width, height=self.height)

    def to_monitor_info(self) -> MonitorInfo:
        return MonitorInfo(
            index=self.mon_index,
            left=self.left,
            top=self.top,
            width=self.width,
            height=self.height,
            is_primary=self.is_primary,
            name=self.name,
            dpi_x=self.dpi_x,
            dpi_y=self.dpi_y,
            dpi_scale=self.dpi_scale,
            orientation=self.orientation,
        )


def list_monitors() -> list[MonitorInfo]:
    """Return physical monitors (mss index 1..N). Index 0 virtual-all is separate."""
    ensure_dpi_awareness()
    try:
        import mss
    except ImportError as exc:
        raise CaptureError("mss 未安装", cause=exc) from exc

    out: list[MonitorInfo] = []
    with mss.mss() as sct:
        # sct.monitors[0] is virtual union; [1..] are physical
        for idx, mon in enumerate(sct.monitors):
            if idx == 0:
                continue
            dpi_x, dpi_y = 96, 96
            orientation = DisplayOrientation.ROT0
            name = str(mon.get("name") or f"Monitor-{idx}")
            if sys.platform == "win32":
                # Map via EnumDisplayMonitors + rect match
                hmon = _find_hmonitor(mon["left"], mon["top"], mon["width"], mon["height"])
                if hmon:
                    dpi_x, dpi_y = _monitor_dpi(hmon)
                # device name for orientation often like \\.\DISPLAY1
                orientation = _orientation_from_device_index(idx)
            scale = dpi_x / 96.0 if dpi_x else 1.0
            out.append(
                MonitorInfo(
                    index=idx,
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                    is_primary=bool(mon.get("is_primary"))
                    or (idx == 1 and not any(m.get("is_primary") for m in sct.monitors[1:])),
                    name=name,
                    dpi_x=dpi_x,
                    dpi_y=dpi_y,
                    dpi_scale=scale,
                    orientation=orientation,
                )
            )
    # Fix primary flag if mss provided it
    if out and not any(m.is_primary for m in out):
        # Prefer monitor covering (0,0) or index 1
        for m in out:
            if m.left <= 0 <= m.left + m.width and m.top <= 0 <= m.top + m.height:
                m.is_primary = True
                break
        else:
            out[0].is_primary = True
    return out


def virtual_desktop() -> MonitorInfo:
    ensure_dpi_awareness()
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[0]
        return MonitorInfo(
            index=0,
            left=int(mon["left"]),
            top=int(mon["top"]),
            width=int(mon["width"]),
            height=int(mon["height"]),
            is_primary=False,
            name="virtual-all",
            dpi_x=96,
            dpi_y=96,
            dpi_scale=1.0,
        )


def _find_hmonitor(left: int, top: int, width: int, height: int) -> int | None:
    if sys.platform != "win32":
        return None
    found: list[int] = []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    def _callback(hmon, _hdc, lprect, _lparam):  # type: ignore[no-untyped-def]
        r = lprect.contents
        if (
            int(r.left) == left
            and int(r.top) == top
            and int(r.right - r.left) == width
            and int(r.bottom - r.top) == height
        ):
            found.append(int(hmon))
        return True

    try:
        ctypes.windll.user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_callback), 0)
    except Exception:  # noqa: BLE001
        return None
    return found[0] if found else None


def _orientation_from_device_index(idx: int) -> DisplayOrientation:
    # Best-effort: \\.\DISPLAY{n}
    return _display_orientation(f"\\\\.\\DISPLAY{idx}")


def resolve_capture_region(
    *,
    mode: str,
    monitor_index: int = 1,
    region: dict[str, int] | None = None,
    window_bbox: BBox | None = None,
) -> CaptureRegion:
    """Resolve mode → physical capture rectangle."""
    ensure_dpi_awareness()
    monitors = list_monitors()
    if not monitors and mode != "region":
        raise CaptureError("未检测到显示器", code=ErrorCode.MONITOR_NOT_FOUND)

    if mode == "all":
        vd = virtual_desktop()
        return CaptureRegion(
            left=vd.left,
            top=vd.top,
            width=vd.width,
            height=vd.height,
            mon_index=0,
            name="virtual-all",
        )

    if mode == "primary":
        primary = next((m for m in monitors if m.is_primary), monitors[0])
        return CaptureRegion(
            left=primary.left,
            top=primary.top,
            width=primary.width,
            height=primary.height,
            mon_index=primary.index,
            dpi_scale=primary.dpi_scale,
            dpi_x=primary.dpi_x,
            dpi_y=primary.dpi_y,
            orientation=primary.orientation,
            is_primary=True,
            name=primary.name,
        )

    if mode == "region":
        if not region:
            raise CaptureError("region 模式需要 capture.region", code=ErrorCode.CAPTURE_FAILED)
        try:
            left = int(region["x"])
            top = int(region["y"])
            width = int(region["width"])
            height = int(region["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"非法 region: {region}", cause=exc) from exc
        if width <= 0 or height <= 0:
            raise CaptureError("region 宽高必须 > 0")
        # Inherit DPI from monitor containing the region center
        cx, cy = left + width // 2, top + height // 2
        host = _monitor_at(monitors, cx, cy) or (monitors[0] if monitors else None)
        return CaptureRegion(
            left=left,
            top=top,
            width=width,
            height=height,
            mon_index=host.index if host else -1,
            dpi_scale=host.dpi_scale if host else 1.0,
            dpi_x=host.dpi_x if host else 96,
            dpi_y=host.dpi_y if host else 96,
            orientation=host.orientation if host else DisplayOrientation.ROT0,
            name=host.name if host else "region",
        )

    if mode == "window":
        if window_bbox is None or window_bbox.width <= 0 or window_bbox.height <= 0:
            raise CaptureError("window 模式需要有效窗口矩形", code=ErrorCode.CAPTURE_FAILED)
        host = _monitor_at(
            monitors,
            window_bbox.x + window_bbox.width // 2,
            window_bbox.y + window_bbox.height // 2,
        )
        return CaptureRegion(
            left=window_bbox.x,
            top=window_bbox.y,
            width=window_bbox.width,
            height=window_bbox.height,
            mon_index=host.index if host else -1,
            dpi_scale=host.dpi_scale if host else 1.0,
            dpi_x=host.dpi_x if host else 96,
            dpi_y=host.dpi_y if host else 96,
            orientation=host.orientation if host else DisplayOrientation.ROT0,
            name=host.name if host else "window",
        )

    # Explicit monitor by index (fallback for primary-like)
    mon = next((m for m in monitors if m.index == monitor_index), None)
    if mon is None:
        if monitor_index == 0:
            return resolve_capture_region(mode="all", monitor_index=0)
        raise CaptureError(
            f"显示器 index={monitor_index} 不存在",
            code=ErrorCode.MONITOR_NOT_FOUND,
            details={"available": [m.index for m in monitors]},
        )
    return CaptureRegion(
        left=mon.left,
        top=mon.top,
        width=mon.width,
        height=mon.height,
        mon_index=mon.index,
        dpi_scale=mon.dpi_scale,
        dpi_x=mon.dpi_x,
        dpi_y=mon.dpi_y,
        orientation=mon.orientation,
        is_primary=mon.is_primary,
        name=mon.name,
    )


def _monitor_at(monitors: list[MonitorInfo], x: int, y: int) -> MonitorInfo | None:
    for m in monitors:
        if m.bbox.contains(x, y):
            return m
    return None


def physical_to_logical(pt: Point, dpi_scale: float) -> Point:
    if dpi_scale <= 0:
        return pt
    return Point(x=int(round(pt.x / dpi_scale)), y=int(round(pt.y / dpi_scale)))


def logical_to_physical(pt: Point, dpi_scale: float) -> Point:
    return Point(x=int(round(pt.x * dpi_scale)), y=int(round(pt.y * dpi_scale)))


def image_point_to_screen(
    *,
    origin_x: int,
    origin_y: int,
    scale_x: float,
    scale_y: float,
    ix: float,
    iy: float,
) -> Point:
    return Point(
        x=int(round(origin_x + ix * scale_x)),
        y=int(round(origin_y + iy * scale_y)),
    )


def cursor_pos_physical() -> Point:
    """Current mouse position in virtual-desktop physical pixels (Windows)."""
    ensure_dpi_awareness()
    if sys.platform != "win32":
        return Point(x=0, y=0)

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return Point(x=int(pt.x), y=int(pt.y))
