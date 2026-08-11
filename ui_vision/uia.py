"""Windows UI Automation tree → UIElement (physical screen pixels).

Uses comtypes + UIAutomationClient when available. Coordinates come from
UIA BoundingRectangle in physical pixels when the process is Per-Monitor DPI aware.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from core.config import AppConfig
from core.logging import get_logger
from core.models import BBox, ElementType, ScreenFrame, UIElement
from PIL import Image

from ui_vision.base import UIRecognizer
from ui_vision.coords import attach_dpi_fields, clamp_to_frame_region, physical_to_logical_bbox
from ui_vision.ids import content_hash, make_element_id

_log = get_logger("ui_vision.uia")

# UIA Control Type IDs
_UIA_BUTTON = 50000
_UIA_CALENDAR = 50001
_UIA_CHECKBOX = 50002
_UIA_COMBOBOX = 50003
_UIA_EDIT = 50004
_UIA_HYPERLINK = 50005
_UIA_IMAGE = 50006
_UIA_LISTITEM = 50007
_UIA_LIST = 50008
_UIA_MENU = 50009
_UIA_MENUBAR = 50010
_UIA_MENUITEM = 50011
_UIA_PROGRESSBAR = 50012
_UIA_RADIOBUTTON = 50013
_UIA_SCROLLBAR = 50014
_UIA_SLIDER = 50015
_UIA_SPINNER = 50016
_UIA_STATUSBAR = 50017
_UIA_TAB = 50018
_UIA_TABITEM = 50019
_UIA_TEXT = 50020
_UIA_TOOLBAR = 50021
_UIA_TOOLTIP = 50022
_UIA_TREE = 50023
_UIA_TREEITEM = 50024
_UIA_CUSTOM = 50025
_UIA_GROUP = 50026
_UIA_THUMB = 50027
_UIA_DATAITEM = 50028
_UIA_DATAGRID = 50029
_UIA_DOCUMENT = 50030
_UIA_SPLITBUTTON = 50031
_UIA_WINDOW = 50032
_UIA_PANE = 50033
_UIA_HEADER = 50034
_UIA_HEADERITEM = 50035
_UIA_TABLE = 50036
_UIA_TITLEBAR = 50037
_UIA_SEPARATOR = 50038

_CONTROL_MAP: dict[int, ElementType] = {
    _UIA_BUTTON: ElementType.BUTTON,
    _UIA_SPLITBUTTON: ElementType.BUTTON,
    _UIA_CHECKBOX: ElementType.CHECKBOX,
    _UIA_RADIOBUTTON: ElementType.RADIO,
    _UIA_EDIT: ElementType.INPUT,
    _UIA_COMBOBOX: ElementType.INPUT,
    _UIA_HYPERLINK: ElementType.LINK,
    _UIA_MENU: ElementType.MENU,
    _UIA_MENUBAR: ElementType.MENU,
    _UIA_MENUITEM: ElementType.MENU_ITEM,
    _UIA_TAB: ElementType.TAB,
    _UIA_TABITEM: ElementType.TAB,
    _UIA_LIST: ElementType.LIST,
    _UIA_LISTITEM: ElementType.LIST,
    _UIA_TREE: ElementType.LIST,
    _UIA_TREEITEM: ElementType.LIST,
    _UIA_TABLE: ElementType.TABLE,
    _UIA_DATAGRID: ElementType.TABLE,
    _UIA_DATAITEM: ElementType.TABLE,
    _UIA_IMAGE: ElementType.IMAGE,
    _UIA_TEXT: ElementType.TEXT,
    _UIA_DOCUMENT: ElementType.TEXT,
    _UIA_WINDOW: ElementType.WINDOW,
    _UIA_PANE: ElementType.OTHER,
    _UIA_GROUP: ElementType.OTHER,
    _UIA_TOOLBAR: ElementType.OTHER,
    _UIA_STATUSBAR: ElementType.OTHER,
    _UIA_TITLEBAR: ElementType.OTHER,
    _UIA_CUSTOM: ElementType.OTHER,
}

# UIA State patterns (IsEnabled, IsOffscreen, IsKeyboardFocusable often via patterns)
_CLICKABLE_TYPES = {
    ElementType.BUTTON,
    ElementType.LINK,
    ElementType.CHECKBOX,
    ElementType.RADIO,
    ElementType.MENU_ITEM,
    ElementType.TAB,
    ElementType.LIST,
}
_EDITABLE_TYPES = {ElementType.INPUT}


def _control_to_element_type(control_type: int) -> ElementType:
    return _CONTROL_MAP.get(int(control_type), ElementType.OTHER)


def _role_name(control_type: int) -> str:
    mapping = {
        _UIA_BUTTON: "button",
        _UIA_EDIT: "textbox",
        _UIA_CHECKBOX: "checkbox",
        _UIA_RADIOBUTTON: "radio",
        _UIA_HYPERLINK: "link",
        _UIA_MENUITEM: "menuitem",
        _UIA_MENU: "menu",
        _UIA_TABITEM: "tab",
        _UIA_TAB: "tablist",
        _UIA_WINDOW: "window",
        _UIA_LIST: "list",
        _UIA_LISTITEM: "listitem",
        _UIA_IMAGE: "image",
        _UIA_TEXT: "text",
        _UIA_COMBOBOX: "combobox",
        _UIA_TABLE: "table",
        _UIA_DATAGRID: "grid",
    }
    return mapping.get(int(control_type), f"uia:{control_type}")


class UiaRecognizer(UIRecognizer):
    name = "uia"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._automation: Any = None
        self._walker: Any = None
        self._init_error: str | None = None
        self._ensure_engine()

    def _ensure_engine(self) -> bool:
        if self._automation is not None:
            return True
        if sys.platform != "win32":
            self._init_error = "UIA only supported on Windows"
            return False
        try:
            from capture.geometry import ensure_dpi_awareness

            ensure_dpi_awareness()
        except Exception:  # noqa: BLE001
            pass
        try:
            import comtypes.client

            # Generate / load UIAutomationClient typelib wrappers once.
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen.UIAutomationClient import CUIAutomation, IUIAutomation

            self._automation = comtypes.client.CreateObject(CUIAutomation, interface=IUIAutomation)
            self._walker = self._automation.ControlViewWalker
            self._init_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            _log.warning("uia_init_failed", extra={"event": "uia_init_failed", "error": str(exc)})
            return False

    def recognize(
        self,
        frame: ScreenFrame,
        image: Image.Image | None = None,
        *,
        roi: BBox | None = None,
        trace_id: str = "",
    ) -> list[UIElement]:
        del image  # UIA is tree-based; image unused.
        if not self._ensure_engine():
            return []

        t0 = time.perf_counter()
        max_elements = max(8, self.config.ui_vision.max_elements * 3)
        max_depth = int(getattr(self.config.ui_vision, "uia_max_depth", 12) or 12)
        timeout_ms = self.config.ui_vision.timeout_ms
        elements: list[UIElement] = []
        used_ids: list[str] = []

        try:
            root = self._root_element(frame)
            if root is None:
                return []
            self._walk(
                root,
                frame=frame,
                roi=roi,
                elements=elements,
                used_ids=used_ids,
                depth=0,
                max_depth=max_depth,
                max_elements=max_elements,
                t0=t0,
                timeout_ms=timeout_ms,
                parent_id=None,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "uia_recognize_failed",
                extra={"event": "uia_recognize_failed", "error": str(exc), "trace_id": trace_id},
            )
            return elements

        return elements

    def _root_element(self, frame: ScreenFrame) -> Any:
        assert self._automation is not None
        hwnd = None
        if frame.window and frame.window.hwnd:
            hwnd = int(frame.window.hwnd)
        if hwnd:
            try:
                # UIA_HwndProperty or ElementFromHandle
                return self._automation.ElementFromHandle(hwnd)
            except Exception:  # noqa: BLE001
                pass
        # Fallback: element from point at frame center, then walk up to window;
        # or whole desktop root.
        try:
            cx = frame.origin_x + (frame.physical_width or int(frame.width * frame.scale_x)) // 2
            cy = frame.origin_y + (frame.physical_height or int(frame.height * frame.scale_y)) // 2
            pt_el = self._automation.ElementFromPoint(self._point(cx, cy))
            if pt_el is not None:
                return pt_el
        except Exception:  # noqa: BLE001
            pass
        try:
            return self._automation.GetRootElement()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _point(x: int, y: int) -> Any:
        # tagPOINT
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        return POINT(int(x), int(y))

    def _walk(
        self,
        element: Any,
        *,
        frame: ScreenFrame,
        roi: BBox | None,
        elements: list[UIElement],
        used_ids: list[str],
        depth: int,
        max_depth: int,
        max_elements: int,
        t0: float,
        timeout_ms: int,
        parent_id: str | None,
    ) -> None:
        if len(elements) >= max_elements:
            return
        if (time.perf_counter() - t0) * 1000 > timeout_ms:
            return
        if depth > max_depth:
            return

        ui_el = self._to_ui_element(element, frame=frame, depth=depth, parent_id=parent_id)
        keep_children = True
        if ui_el is not None:
            if roi is not None:
                cx, cy = ui_el.bbox.center()
                if not roi.contains(cx, cy) and ui_el.type not in (
                    ElementType.WINDOW,
                    ElementType.DIALOG,
                ):
                    # Still walk children that might sit inside ROI.
                    keep_children = True
                else:
                    elements.append(ui_el)
                    used_ids.append(ui_el.element_id)
            else:
                elements.append(ui_el)
                used_ids.append(ui_el.element_id)
            parent_for_child = ui_el.element_id
        else:
            parent_for_child = parent_id

        if not keep_children or depth >= max_depth:
            return

        try:
            child = self._walker.GetFirstChildElement(element)
        except Exception:  # noqa: BLE001
            return
        while child is not None:
            self._walk(
                child,
                frame=frame,
                roi=roi,
                elements=elements,
                used_ids=used_ids,
                depth=depth + 1,
                max_depth=max_depth,
                max_elements=max_elements,
                t0=t0,
                timeout_ms=timeout_ms,
                parent_id=parent_for_child,
            )
            if len(elements) >= max_elements:
                break
            if (time.perf_counter() - t0) * 1000 > timeout_ms:
                break
            try:
                child = self._walker.GetNextSiblingElement(child)
            except Exception:  # noqa: BLE001
                break

    def _to_ui_element(
        self,
        element: Any,
        *,
        frame: ScreenFrame,
        depth: int,
        parent_id: str | None,
    ) -> UIElement | None:
        try:
            # CurrentPropertyValue ids:
            # Name 30005, ControlType 30003, BoundingRectangle 30001,
            # IsEnabled 30010, IsOffscreen 30022, ClassName 30012,
            # AutomationId 30011, IsKeyboardFocusable 30009
            name = str(self._prop(element, 30005) or "")
            control_type = int(self._prop(element, 30003) or 0)
            rect = self._prop(element, 30001)
            is_enabled = bool(
                self._prop(element, 30010) if self._prop(element, 30010) is not None else True
            )
            is_offscreen = bool(self._prop(element, 30022) or False)
            class_name = str(self._prop(element, 30012) or "")
            automation_id = str(self._prop(element, 30011) or "")
            focusable = bool(self._prop(element, 30009) or False)
        except Exception:  # noqa: BLE001
            return None

        box = self._parse_rect(rect)
        if box is None or box.width <= 1 or box.height <= 1:
            return None
        # Skip enormous pure desktop panes with no name at depth 0 sometimes useful as window
        et = _control_to_element_type(control_type)
        # Skip nameless noise panes that are huge
        if (
            et == ElementType.OTHER
            and not name
            and depth > 2
            and box.width * box.height > 400 * 300
        ):
            return None

        box = clamp_to_frame_region(box, frame)
        if box.width <= 0 or box.height <= 0:
            return None

        # Filter elements completely outside capture region
        pw = frame.physical_width or int(round(frame.width * frame.scale_x))
        ph = frame.physical_height or int(round(frame.height * frame.scale_y))
        region = BBox(x=frame.origin_x, y=frame.origin_y, width=pw, height=ph)
        cx, cy = box.center()
        if not region.contains(cx, cy) and et not in (ElementType.WINDOW, ElementType.DIALOG):
            return None

        role = _role_name(control_type)
        text = name.strip()
        clickable = et in _CLICKABLE_TYPES or (focusable and et == ElementType.OTHER and bool(text))
        editable = et in _EDITABLE_TYPES
        # Value pattern for edit current value — best effort
        if editable:
            try:
                val = self._prop(element, 30045)  # Value may vary; ignore failures
                if val and not text:
                    text = str(val)
            except Exception:  # noqa: BLE001
                pass

        native = automation_id or (f"cls:{class_name}" if class_name else None)
        ch = content_hash(
            type=et, text=text, role=role, bbox=box, enabled=is_enabled, native_id=native
        )
        eid = make_element_id(type=et, text=text, bbox=box, role=role, native_id=native)
        conf = 0.92 if text else 0.8
        if is_offscreen:
            conf *= 0.5
        el = UIElement(
            element_id=eid,
            type=et,
            role=role,
            text=text,
            name=name,
            bbox=box,
            bbox_logical=physical_to_logical_bbox(box, frame.dpi_scale or 1.0),
            confidence=min(0.99, conf),
            visible=not is_offscreen,
            enabled=is_enabled,
            clickable=clickable and is_enabled and not is_offscreen,
            editable=editable and is_enabled,
            source=["uia"],
            frame_id=frame.frame_id,
            parent_id=parent_id,
            depth=depth,
            content_hash=ch,
            native_id=native,
            extra={"class_name": class_name, "control_type": control_type},
        )
        return attach_dpi_fields(el, frame)

    @staticmethod
    def _prop(element: Any, pid: int) -> Any:
        # Prefer typed Current* attributes (faster / more reliable via comtypes).
        attr_map = {
            30005: "CurrentName",
            30003: "CurrentControlType",
            30001: "CurrentBoundingRectangle",
            30010: "CurrentIsEnabled",
            30022: "CurrentIsOffscreen",
            30012: "CurrentClassName",
            30011: "CurrentAutomationId",
            30009: "CurrentIsKeyboardFocusable",
        }
        attr = attr_map.get(pid)
        if attr and hasattr(element, attr):
            try:
                return getattr(element, attr)
            except Exception:  # noqa: BLE001
                pass
        try:
            return element.GetCurrentPropertyValue(pid)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _parse_rect(rect: Any) -> BBox | None:
        if rect is None:
            return None
        try:
            # comtypes may return tuple (left, top, right, bottom) or structure
            if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                left, top, right, bottom = (
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                )
            else:
                left = float(getattr(rect, "left", getattr(rect, "Left", 0)))
                top = float(getattr(rect, "top", getattr(rect, "Top", 0)))
                right = float(getattr(rect, "right", getattr(rect, "Right", 0)))
                bottom = float(getattr(rect, "bottom", getattr(rect, "Bottom", 0)))
                # Sometimes width/height form
                if right == 0 and bottom == 0 and hasattr(rect, "width"):
                    right = left + float(rect.width)
                    bottom = top + float(rect.height)
            w = int(round(right - left))
            h = int(round(bottom - top))
            if w <= 0 or h <= 0:
                # Maybe already left,top,width,height
                if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                    w2, h2 = int(rect[2]), int(rect[3])
                    if w2 > 0 and h2 > 0 and w2 < 20000:
                        return BBox(x=int(rect[0]), y=int(rect[1]), width=w2, height=h2)
                return None
            return BBox(x=int(round(left)), y=int(round(top)), width=w, height=h)
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        self._automation = None
        self._walker = None
