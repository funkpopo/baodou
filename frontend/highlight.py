"""Draw UI element / target highlights for the main window preview (Phase H)."""

from __future__ import annotations

from typing import Any

from core.models import ActionPreview, BBox, UIElement, UIVisionResult
from PIL import Image, ImageDraw, ImageFont


def _font() -> Any:
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def _to_image_box(
    bbox: BBox,
    screen_to_image: Any | None,
) -> BBox | None:
    if screen_to_image is None:
        return bbox
    try:
        p1 = screen_to_image(bbox.x, bbox.y)
        p2 = screen_to_image(bbox.x + bbox.width, bbox.y + bbox.height)

        def _xy(pt: Any) -> tuple[int, int]:
            if hasattr(pt, "x") and hasattr(pt, "y"):
                return int(pt.x), int(pt.y)
            if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                return int(pt[0]), int(pt[1])
            raise TypeError(type(pt))

        x1, y1 = _xy(p1)
        x2, y2 = _xy(p2)
        return BBox(x=min(x1, x2), y=min(y1, y2), width=abs(x2 - x1), height=abs(y2 - y1))
    except Exception:  # noqa: BLE001
        return None


def highlight_elements(
    image: Image.Image,
    elements: list[UIElement],
    *,
    screen_to_image: Any | None = None,
    highlight_ids: set[str] | None = None,
    max_draw: int = 48,
    rejected_ids: set[str] | None = None,
) -> Image.Image:
    """Draw element boxes; emphasize highlight_ids in gold, rejected in dim red."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    hi = highlight_ids or set()
    rej = rejected_ids or set()
    for i, el in enumerate(elements[:max_draw]):
        box = _to_image_box(el.bbox, screen_to_image)
        if box is None or box.width <= 0 or box.height <= 0:
            continue
        if el.element_id in rej:
            color = (120, 40, 40)
            width = 1
        elif el.element_id in hi:
            color = (255, 200, 0)
            width = 3
        elif el.clickable or el.editable:
            color = (64, 160, 255)
            width = 2
        else:
            color = (100, 100, 100)
            width = 1
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.width, box.y + box.height
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = el.element_id
        if el.text:
            label = f"{el.element_id}:{el.text[:16]}"
        ty = max(0, y1 - 12)
        if font:
            draw.text((x1 + 1, ty), label, fill=color, font=font)
        else:
            draw.text((x1 + 1, ty), label, fill=color)
        _ = i
    return canvas


def highlight_preview_target(
    image: Image.Image,
    preview: ActionPreview,
    vision: UIVisionResult | None = None,
    *,
    screen_to_image: Any | None = None,
) -> Image.Image:
    """Emphasize the upcoming action target (bbox + crosshair at center)."""
    elements = list(vision.elements) if vision else []
    hi: set[str] = set()
    if preview.target_element_id:
        hi.add(preview.target_element_id)
    canvas = highlight_elements(
        image,
        elements,
        screen_to_image=screen_to_image,
        highlight_ids=hi,
    )
    draw = ImageDraw.Draw(canvas)
    box: BBox | None = preview.target_bbox
    pt = preview.target_point
    if box is not None:
        ib = _to_image_box(box, screen_to_image)
        if ib is not None:
            draw.rectangle(
                [ib.x, ib.y, ib.x + ib.width, ib.y + ib.height],
                outline=(255, 64, 64),
                width=4,
            )
            cx = ib.x + ib.width // 2
            cy = ib.y + ib.height // 2
            r = 8
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 64, 64), width=2)
            draw.line([cx - 12, cy, cx + 12, cy], fill=(255, 64, 64), width=2)
            draw.line([cx, cy - 12, cx, cy + 12], fill=(255, 64, 64), width=2)
    elif pt is not None:
        if screen_to_image is not None:
            try:
                p = screen_to_image(pt.x, pt.y)
                cx, cy = (int(p.x), int(p.y)) if hasattr(p, "x") else (int(p[0]), int(p[1]))
            except Exception:  # noqa: BLE001
                cx, cy = pt.x, pt.y
        else:
            cx, cy = pt.x, pt.y
        r = 10
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 64, 64), width=3)
        draw.line([cx - 14, cy, cx + 14, cy], fill=(255, 64, 64), width=2)
        draw.line([cx, cy - 14, cx, cy + 14], fill=(255, 64, 64), width=2)
    return canvas


def resize_for_preview(
    image: Image.Image,
    max_width: int = 720,
    max_height: int = 480,
) -> Image.Image:
    """Fit image into preview pane while preserving aspect ratio."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 0.999:
        return image.convert("RGB")
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
