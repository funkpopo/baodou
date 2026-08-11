"""Mock UI vision: fixed, stable element set for pipeline demos and tests."""

from __future__ import annotations

import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import BBox, ElementType, ScreenFrame, UIElement, UIVisionResult
from PIL import Image

from ui_vision.base import UIRecognizer, UIVisionBackend
from ui_vision.coords import attach_dpi_fields, physical_to_logical_bbox
from ui_vision.ids import content_hash, make_element_id

_log = get_logger("ui_vision.mock")


def _mock_elements(frame: ScreenFrame) -> list[UIElement]:
    """Synthetic tree scaled to frame physical region (multi-res safe)."""
    # Prefer physical size so different resolutions stay consistent in screen space.
    pw = frame.physical_width if frame.physical_width is not None else frame.width
    ph = frame.physical_height if frame.physical_height is not None else frame.height
    ox, oy = frame.origin_x, frame.origin_y
    scale = frame.dpi_scale or 1.0

    specs: list[tuple[str, ElementType, str, str, BBox, bool, bool, float]] = [
        (
            "win_main_01",
            ElementType.WINDOW,
            "window",
            "Mock Desktop",
            BBox(x=ox, y=oy, width=pw, height=ph),
            False,
            False,
            0.99,
        ),
        (
            "btn_search_01",
            ElementType.BUTTON,
            "button",
            "搜索",
            BBox(x=ox + pw - 200, y=oy + 40, width=96, height=36),
            True,
            False,
            0.97,
        ),
        (
            "inp_query_01",
            ElementType.INPUT,
            "textbox",
            "",
            BBox(x=ox + pw // 2 - 160, y=oy + 40, width=280, height=36),
            True,
            True,
            0.95,
        ),
        (
            "btn_ok_01",
            ElementType.BUTTON,
            "button",
            "确定",
            BBox(x=ox + pw // 2 - 40, y=oy + ph - 80, width=80, height=32),
            True,
            False,
            0.92,
        ),
    ]
    out: list[UIElement] = []
    for eid, et, role, text, box, clickable, editable, conf in specs:
        ch = content_hash(type=et, text=text, role=role, bbox=box)
        # Keep classic demo ids stable; also verify make_element_id path works.
        _ = make_element_id(type=et, text=text, bbox=box, role=role)
        el = UIElement(
            element_id=eid,
            type=et,
            role=role,
            text=text,
            name=text,
            bbox=box,
            bbox_logical=physical_to_logical_bbox(box, scale),
            confidence=conf,
            visible=True,
            enabled=True,
            clickable=clickable,
            editable=editable,
            source=["mock"],
            frame_id=frame.frame_id,
            content_hash=ch,
            parent_id="win_main_01" if eid != "win_main_01" else None,
            depth=0 if eid == "win_main_01" else 1,
        )
        out.append(attach_dpi_fields(el, frame))
    return out


class MockRecognizer(UIRecognizer):
    name = "mock"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def recognize(
        self,
        frame: ScreenFrame,
        image: Image.Image | None = None,
        *,
        roi: BBox | None = None,
        trace_id: str = "",
    ) -> list[UIElement]:
        del image, roi, trace_id
        return _mock_elements(frame)


class MockUIVision(UIVisionBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def recognize(
        self,
        frame: ScreenFrame,
        *,
        trace_id: str = "",
        image: Image.Image | None = None,
        roi: BBox | None = None,
        goal: str | None = None,
    ) -> UIVisionResult:
        del image, roi, goal
        get_global_token().check()
        t0 = time.perf_counter()
        time.sleep(0.005)

        elements = _mock_elements(frame)
        thr = self.config.ui_vision.confidence_threshold
        elements = [e for e in elements if e.confidence >= thr]
        elements = elements[: self.config.ui_vision.max_elements]

        latency = (time.perf_counter() - t0) * 1000
        result = UIVisionResult(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            elements=elements,
            latency_ms=latency,
            source="mock",
            sources_used=["mock"],
            notes="synthetic UI tree for Phase B/D",
            dpi_scale=frame.dpi_scale or 1.0,
        )
        log_event(_log, "vision.result", **result.log_summary())
        return result
