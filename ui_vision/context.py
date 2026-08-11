"""Compact UI context for Qwen: filter, serialize, annotate numbered boxes."""

from __future__ import annotations

import re
from typing import Any

from core.models import BBox, ElementType, UIElement, UIVisionResult
from PIL import Image, ImageDraw, ImageFont

# Fields kept in compact JSON for the model (no raw pixels).
_COMPACT_KEYS = (
    "element_id",
    "type",
    "text",
    "role",
    "bbox",
    "center",
    "confidence",
    "clickable",
    "editable",
    "enabled",
    "visible",
    "needs_review",
    "source",
)


def _clean_text(text: str, *, max_len: int = 80) -> str:
    # Strip bidi/control marks that break some Windows consoles; keep content readable.
    cleaned = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text or "")
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned).strip()
    return cleaned[:max_len]


def element_to_compact(el: UIElement) -> dict[str, Any]:
    c = el.center
    return {
        "element_id": el.element_id,
        "type": el.type.value if isinstance(el.type, ElementType) else str(el.type),
        "text": _clean_text(el.text or el.name or ""),
        "role": el.role,
        "bbox": {"x": el.bbox.x, "y": el.bbox.y, "width": el.bbox.width, "height": el.bbox.height},
        "center": {"x": c.x, "y": c.y},
        "confidence": round(el.confidence, 3),
        "clickable": el.clickable,
        "editable": el.editable,
        "enabled": el.enabled,
        "visible": el.visible,
        "needs_review": el.needs_review or el.conflict,
        "source": list(el.source),
    }


def filter_elements_for_goal(
    elements: list[UIElement],
    goal: str | None,
    *,
    max_elements: int = 32,
    include_review: bool = True,
) -> list[UIElement]:
    """Rank / filter elements relevant to a user goal to keep prompt compact."""
    if not elements:
        return []
    if not goal or not goal.strip():
        # Default: interactive first, then labeled text, cap count.
        ranked = sorted(
            elements,
            key=lambda e: (
                0 if (e.clickable or e.editable) else 1,
                0 if (e.text or e.name) else 1,
                -e.confidence,
                e.depth,
            ),
        )
        out = ranked[:max_elements]
        if include_review:
            for e in elements:
                if (e.needs_review or e.conflict) and e not in out:
                    out.append(e)
                    if len(out) >= max_elements + 4:
                        break
        return out[: max_elements + 4]

    tokens = _goal_tokens(goal)
    scored: list[tuple[float, UIElement]] = []
    for e in elements:
        score = 0.0
        blob = f"{e.text} {e.name} {e.role} {e.type.value}".lower()
        for tok in tokens:
            if tok in blob:
                score += 3.0
            elif len(tok) >= 2 and tok in blob.replace(" ", ""):
                score += 1.5
        if e.clickable or e.editable:
            score += 1.0
        if e.type in {
            ElementType.BUTTON,
            ElementType.INPUT,
            ElementType.LINK,
            ElementType.MENU_ITEM,
        }:
            score += 0.8
        score += e.confidence
        if e.needs_review or e.conflict:
            score += 0.3  # keep ambiguous candidates visible to the model
        # Prefer on-screen actionable near typical chrome
        scored.append((score, e))

    scored.sort(key=lambda t: t[0], reverse=True)
    # Always keep a small set of top interactive even if score low
    picked: list[UIElement] = []
    seen: set[str] = set()
    for _, e in scored:
        if e.element_id in seen:
            continue
        seen.add(e.element_id)
        picked.append(e)
        if len(picked) >= max_elements:
            break
    return picked


def serialize_for_model(
    result: UIVisionResult,
    *,
    goal: str | None = None,
    max_elements: int = 32,
) -> list[dict[str, Any]]:
    els = filter_elements_for_goal(result.elements, goal, max_elements=max_elements)
    return [element_to_compact(e) for e in els]


def serialize_text_summary(
    result: UIVisionResult,
    *,
    goal: str | None = None,
    max_elements: int = 32,
) -> str:
    """Human/model-readable compact lines; prefer element_id over raw coords."""
    rows = serialize_for_model(result, goal=goal, max_elements=max_elements)
    lines = [f"frame_id={result.frame_id} elements={len(rows)} dpi_scale={result.dpi_scale:.2f}"]
    for i, r in enumerate(rows, start=1):
        flag = " REVIEW" if r.get("needs_review") else ""
        lines.append(
            f"[{i}] id={r['element_id']} type={r['type']} text={r['text']!r} "
            f"clickable={r['clickable']} conf={r['confidence']}{flag}"
        )
    return "\n".join(lines)


def annotate_image(
    image: Image.Image,
    elements: list[UIElement],
    *,
    frame_origin_to_image: Any | None = None,
    max_draw: int = 32,
    goal: str | None = None,
) -> Image.Image:
    """Draw numbered boxes; index matches compact list order for the model.

    ``frame_origin_to_image`` if provided should be a callable (UIElement) -> BBox
    in image pixels; otherwise assumes element.extra['image_bbox'] or skips
    conversion (caller should pass image-space boxes via extra).
    """
    els = filter_elements_for_goal(elements, goal, max_elements=max_draw)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    colors = [
        (255, 64, 64),
        (64, 160, 255),
        (64, 200, 64),
        (255, 180, 0),
        (200, 64, 200),
        (0, 200, 200),
    ]
    for i, el in enumerate(els):
        box = _image_box_for_element(el, frame_origin_to_image)
        if box is None:
            continue
        color = colors[i % len(colors)]
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.width, box.y + box.height
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{i + 1}:{el.element_id}"
        ty = max(0, y1 - 12)
        if font:
            draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 12], fill=color)
            draw.text((x1 + 1, ty), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1 + 1, ty), label, fill=color)
    return canvas


def annotate_from_frame(
    image: Image.Image,
    result: UIVisionResult,
    frame_screen_to_image,
    *,
    goal: str | None = None,
    max_draw: int = 32,
) -> Image.Image:
    """Annotate using a ``frame.screen_to_image``-style converter."""

    def _xy(pt: Any) -> tuple[int, int]:
        if hasattr(pt, "x") and hasattr(pt, "y"):
            return int(pt.x), int(pt.y)
        if isinstance(pt, (tuple, list)) and len(pt) >= 2:
            return int(pt[0]), int(pt[1])
        raise TypeError(f"unsupported point type: {type(pt)!r}")

    def _conv(el: UIElement) -> BBox:
        x, y = _xy(frame_screen_to_image(el.bbox.x, el.bbox.y))
        x2, y2 = _xy(frame_screen_to_image(el.bbox.x + el.bbox.width, el.bbox.y + el.bbox.height))
        return BBox(x=min(x, x2), y=min(y, y2), width=abs(x2 - x), height=abs(y2 - y))

    return annotate_image(
        image, result.elements, frame_origin_to_image=_conv, max_draw=max_draw, goal=goal
    )


def _image_box_for_element(el: UIElement, converter: Any | None) -> BBox | None:
    if converter is not None:
        try:
            return converter(el)
        except Exception:  # noqa: BLE001
            return None
    raw = el.extra.get("image_bbox") if el.extra else None
    if isinstance(raw, dict):
        return BBox(**raw)
    return None


def _goal_tokens(goal: str) -> list[str]:
    # Split CJK-ish and latin tokens
    parts = re.findall(r"[\u4e00-\u9fff]{1,8}|[a-zA-Z0-9_]{2,}", goal.lower())
    # Also single CJK chars for short goals like "搜索"
    extras = re.findall(r"[\u4e00-\u9fff]", goal)
    tokens = list(dict.fromkeys([*parts, *extras]))
    stop = {"的", "了", "在", "是", "我", "请", "一下", "to", "the", "a", "an", "and"}
    return [t for t in tokens if t not in stop]
