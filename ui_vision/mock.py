"""Mock UI vision: fixed, stable element set for pipeline demos and tests."""

from __future__ import annotations

import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.logging import get_logger, log_event
from core.models import BBox, ElementType, ScreenFrame, UIElement, UIVisionResult

from ui_vision.base import UIVisionBackend

_log = get_logger("ui_vision.mock")


class MockUIVision(UIVisionBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def recognize(self, frame: ScreenFrame, *, trace_id: str = "") -> UIVisionResult:
        get_global_token().check()
        t0 = time.perf_counter()
        time.sleep(0.005)

        elements = [
            UIElement(
                element_id="win_main_01",
                type=ElementType.WINDOW,
                role="window",
                text="Mock Desktop",
                bbox=BBox(x=0, y=0, width=frame.width, height=frame.height),
                confidence=0.99,
                visible=True,
                enabled=True,
                clickable=False,
                source=["mock"],
                frame_id=frame.frame_id,
            ),
            UIElement(
                element_id="btn_search_01",
                type=ElementType.BUTTON,
                role="button",
                text="搜索",
                bbox=BBox(x=frame.width - 200, y=40, width=96, height=36),
                confidence=0.97,
                visible=True,
                enabled=True,
                clickable=True,
                source=["mock"],
                frame_id=frame.frame_id,
            ),
            UIElement(
                element_id="inp_query_01",
                type=ElementType.INPUT,
                role="textbox",
                text="",
                bbox=BBox(x=frame.width // 2 - 160, y=40, width=280, height=36),
                confidence=0.95,
                visible=True,
                enabled=True,
                clickable=True,
                editable=True,
                source=["mock"],
                frame_id=frame.frame_id,
            ),
            UIElement(
                element_id="btn_ok_01",
                type=ElementType.BUTTON,
                role="button",
                text="确定",
                bbox=BBox(x=frame.width // 2 - 40, y=frame.height - 80, width=80, height=32),
                confidence=0.92,
                visible=True,
                enabled=True,
                clickable=True,
                source=["mock"],
                frame_id=frame.frame_id,
            ),
        ]

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
            notes="synthetic UI tree for Phase B",
        )
        log_event(_log, "vision.result", **result.log_summary())
        return result
