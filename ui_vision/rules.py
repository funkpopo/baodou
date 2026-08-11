"""Lightweight visual rules for common controls (no heavy CV model).

Uses Pillow + numpy to find high-contrast rectangular regions that often
correspond to buttons / inputs. Complements UIA on self-drawn UIs.
Image-space boxes are converted to physical screen pixels via the frame.
"""

from __future__ import annotations

import time

import numpy as np
from core.config import AppConfig
from core.models import BBox, ElementType, ScreenFrame, UIElement
from PIL import Image, ImageFilter

from ui_vision.base import UIRecognizer
from ui_vision.coords import attach_dpi_fields, image_bbox_to_screen, physical_to_logical_bbox
from ui_vision.ids import content_hash, make_element_id


class RulesRecognizer(UIRecognizer):
    name = "rules"

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
        del trace_id
        if image is None:
            return []
        if not getattr(self.config.ui_vision, "rules_enabled", True):
            return []

        t0 = time.perf_counter()
        work = image.convert("RGB")
        ox = oy = 0
        if roi is not None:
            from ui_vision.coords import screen_bbox_to_image

            ib = screen_bbox_to_image(frame, roi)
            x0, y0 = max(0, ib.x), max(0, ib.y)
            x1, y1 = min(work.width, ib.x + ib.width), min(work.height, ib.y + ib.height)
            if x1 <= x0 or y1 <= y0:
                return []
            work = work.crop((x0, y0, x1, y1))
            ox, oy = x0, y0

        # Downscale for speed on large frames
        max_side = 640
        scale = 1.0
        if max(work.width, work.height) > max_side:
            scale = max_side / float(max(work.width, work.height))
            nw, nh = max(1, int(work.width * scale)), max(1, int(work.height * scale))
            small = work.resize((nw, nh), Image.Resampling.BILINEAR)
        else:
            small = work

        gray = small.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges, dtype=np.float32)
        # Binary-ish edge map
        thr = max(20.0, float(arr.mean() + arr.std()))
        mask = arr > thr

        boxes = _connected_rectangles(mask, min_area=80, max_area=mask.size // 3)
        # Map back to full image coords
        inv = 1.0 / scale if scale > 0 else 1.0
        elements: list[UIElement] = []
        used: list[str] = []
        timeout_ms = self.config.ui_vision.timeout_ms

        for bx, by, bw, bh in boxes:
            if (time.perf_counter() - t0) * 1000 > timeout_ms:
                break
            img_box = BBox(
                x=ox + int(round(bx * inv)),
                y=oy + int(round(by * inv)),
                width=max(2, int(round(bw * inv))),
                height=max(2, int(round(bh * inv))),
            )
            # Aspect heuristics
            ar = img_box.width / max(1, img_box.height)
            et = ElementType.OTHER
            clickable = False
            editable = False
            conf = 0.55
            if 1.5 <= ar <= 8.0 and 18 <= img_box.height <= 64 and img_box.width <= 420:
                et = ElementType.BUTTON
                clickable = True
                conf = 0.62
            elif 3.0 <= ar <= 20.0 and 18 <= img_box.height <= 48 and img_box.width >= 80:
                et = ElementType.INPUT
                editable = True
                clickable = True
                conf = 0.58
            elif 0.7 <= ar <= 1.4 and 12 <= img_box.width <= 48:
                et = ElementType.ICON
                clickable = True
                conf = 0.5
            else:
                continue  # skip unstructured blobs

            phys = image_bbox_to_screen(frame, img_box)
            if phys.width < 4 or phys.height < 4:
                continue
            role = et.value
            ch = content_hash(type=et, text="", role=role, bbox=phys)
            eid = make_element_id(type=et, text="", bbox=phys, role=role, existing=used)
            used.append(eid)
            el = UIElement(
                element_id=eid,
                type=et,
                role=role,
                text="",
                bbox=phys,
                bbox_logical=physical_to_logical_bbox(phys, frame.dpi_scale or 1.0),
                confidence=conf,
                visible=True,
                enabled=True,
                clickable=clickable,
                editable=editable,
                source=["rules"],
                frame_id=frame.frame_id,
                content_hash=ch,
                needs_review=True,  # visual heuristic — model/user may confirm
                extra={"rule": "edge_rect"},
            )
            elements.append(attach_dpi_fields(el, frame))

        return elements[: self.config.ui_vision.max_elements]


def _connected_rectangles(
    mask: np.ndarray,
    *,
    min_area: int,
    max_area: int,
) -> list[tuple[int, int, int, int]]:
    """Very small connected-component bounding boxes on a boolean mask."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    # Sample stride to keep runtime bounded
    step = 2 if max(h, w) > 400 else 1
    for y in range(0, h, step):
        row = mask[y]
        for x in range(0, w, step):
            if not row[x] or visited[y, x]:
                continue
            # flood fill bounds
            stack = [(x, y)]
            visited[y, x] = True
            minx = maxx = x
            miny = maxy = y
            count = 0
            while stack and count < max_area * 2:
                cx, cy = stack.pop()
                count += 1
                minx, maxx = min(minx, cx), max(maxx, cx)
                miny, maxy = min(miny, cy), max(maxy, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            bw = maxx - minx + 1
            bh = maxy - miny + 1
            area = bw * bh
            if area < min_area or area > max_area:
                continue
            # reject very thin lines
            if bw < 8 or bh < 8:
                continue
            boxes.append((minx, miny, bw, bh))
            if len(boxes) >= 80:
                return boxes
    return boxes
