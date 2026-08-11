"""OCR recognizer for text not covered by UI Automation.

Primary: optional ``pytesseract`` if installed.
Fallback: empty list (pipeline still works with UIA + rules).
Coordinates produced in image space, then converted to physical pixels.
"""

from __future__ import annotations

import time
from typing import Any

from core.config import AppConfig
from core.logging import get_logger
from core.models import BBox, ElementType, ScreenFrame, UIElement
from PIL import Image

from ui_vision.base import UIRecognizer
from ui_vision.coords import attach_dpi_fields, image_bbox_to_screen, physical_to_logical_bbox
from ui_vision.ids import content_hash, make_element_id

_log = get_logger("ui_vision.ocr")


class OcrRecognizer(UIRecognizer):
    name = "ocr"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._available: bool | None = None
        self._engine: str = "none"

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract  # noqa: F401

            self._available = True
            self._engine = "pytesseract"
        except Exception:  # noqa: BLE001
            self._available = False
            self._engine = "none"
        return self._available

    def recognize(
        self,
        frame: ScreenFrame,
        image: Image.Image | None = None,
        *,
        roi: BBox | None = None,
        trace_id: str = "",
    ) -> list[UIElement]:
        if image is None:
            return []
        if not self.config.ui_vision.ocr_enabled:
            return []
        if not self.available():
            return []

        t0 = time.perf_counter()
        work = image
        offset_x = 0
        offset_y = 0
        if roi is not None:
            # ROI is physical — convert to image crop.
            from ui_vision.coords import screen_bbox_to_image

            ib = screen_bbox_to_image(frame, roi)
            x0 = max(0, ib.x)
            y0 = max(0, ib.y)
            x1 = min(image.width, ib.x + ib.width)
            y1 = min(image.height, ib.y + ib.height)
            if x1 <= x0 or y1 <= y0:
                return []
            work = image.crop((x0, y0, x1, y1))
            offset_x, offset_y = x0, y0

        try:
            boxes = self._tesseract_boxes(work)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ocr_failed",
                extra={"event": "ocr_failed", "error": str(exc), "trace_id": trace_id},
            )
            return []

        min_conf = float(getattr(self.config.ui_vision, "ocr_min_confidence", 0.4) or 0.4)
        elements: list[UIElement] = []
        used: list[str] = []
        for item in boxes:
            text = str(item.get("text") or "").strip()
            if not text or len(text) < 1:
                continue
            conf = float(item.get("conf", 0.5))
            if conf < min_conf:
                continue
            img_box = BBox(
                x=offset_x + int(item["x"]),
                y=offset_y + int(item["y"]),
                width=max(1, int(item["w"])),
                height=max(1, int(item["h"])),
            )
            phys = image_bbox_to_screen(frame, img_box)
            if phys.width <= 0 or phys.height <= 0:
                continue
            et = _guess_type(text)
            role = "text" if et == ElementType.TEXT else et.value
            ch = content_hash(type=et, text=text, role=role, bbox=phys)
            eid = make_element_id(type=et, text=text, bbox=phys, role=role, existing=used)
            used.append(eid)
            el = UIElement(
                element_id=eid,
                type=et,
                role=role,
                text=text,
                name=text,
                bbox=phys,
                bbox_logical=physical_to_logical_bbox(phys, frame.dpi_scale or 1.0),
                confidence=min(0.95, conf),
                visible=True,
                enabled=True,
                clickable=et in {ElementType.BUTTON, ElementType.LINK},
                editable=False,
                source=["ocr"],
                frame_id=frame.frame_id,
                content_hash=ch,
                needs_review=conf < 0.6,
                extra={"ocr_engine": self._engine},
            )
            elements.append(attach_dpi_fields(el, frame))

        _ = (time.perf_counter() - t0) * 1000
        return elements[: self.config.ui_vision.max_elements]

    def _tesseract_boxes(self, image: Image.Image) -> list[dict[str, Any]]:
        import pytesseract

        lang = getattr(self.config.ui_vision, "ocr_lang", "chi_sim+eng") or "chi_sim+eng"
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        out: list[dict[str, Any]] = []
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf_raw = float(data["conf"][i])
            except (ValueError, TypeError):
                conf_raw = -1
            if conf_raw < 0:
                continue
            conf = conf_raw / 100.0
            out.append(
                {
                    "text": text,
                    "conf": conf,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "w": int(data["width"][i]),
                    "h": int(data["height"][i]),
                }
            )
        return out


def _guess_type(text: str) -> ElementType:
    t = text.strip().lower()
    button_hints = {
        "ok",
        "cancel",
        "确定",
        "取消",
        "搜索",
        "search",
        "submit",
        "保存",
        "save",
        "关闭",
        "close",
        "yes",
        "no",
        "是",
        "否",
        "登录",
        "login",
        "下一步",
        "next",
    }
    if t in button_hints or (len(t) <= 8 and t.endswith(("…", "..."))):
        return ElementType.BUTTON
    if t.startswith(("http://", "https://", "www.")):
        return ElementType.LINK
    return ElementType.TEXT
