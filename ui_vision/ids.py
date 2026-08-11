"""Stable short-term element IDs and content hashes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from core.models import BBox, ElementType, UIElement

# Quantize physical pixels so tiny jitter does not churn IDs.
_Q = 8

_TYPE_PREFIX: dict[ElementType, str] = {
    ElementType.WINDOW: "win",
    ElementType.DIALOG: "dlg",
    ElementType.BUTTON: "btn",
    ElementType.INPUT: "inp",
    ElementType.CHECKBOX: "chk",
    ElementType.RADIO: "rad",
    ElementType.TAB: "tab",
    ElementType.MENU: "mnu",
    ElementType.MENU_ITEM: "mi",
    ElementType.LINK: "lnk",
    ElementType.LIST: "lst",
    ElementType.TABLE: "tbl",
    ElementType.ICON: "ico",
    ElementType.TEXT: "txt",
    ElementType.IMAGE: "img",
    ElementType.OTHER: "el",
}


def _norm_text(text: str, *, max_len: int = 48) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t[:max_len]


def _quantize_bbox(bbox: BBox, q: int = _Q) -> tuple[int, int, int, int]:
    return (
        (bbox.x // q) * q,
        (bbox.y // q) * q,
        max(q, (bbox.width // q) * q),
        max(q, (bbox.height // q) * q),
    )


def content_hash(
    *,
    type: ElementType | str,
    text: str = "",
    role: str = "",
    bbox: BBox,
    enabled: bool = True,
    native_id: str | None = None,
) -> str:
    """Fingerprint for short-term identity / staleness across frames."""
    t = type.value if isinstance(type, ElementType) else str(type)
    qx, qy, qw, qh = _quantize_bbox(bbox)
    raw = "|".join(
        [
            t,
            _norm_text(text),
            (role or "").strip().lower(),
            f"{qx},{qy},{qw},{qh}",
            "1" if enabled else "0",
            native_id or "",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def make_element_id(
    *,
    type: ElementType | str,
    text: str = "",
    bbox: BBox,
    role: str = "",
    native_id: str | None = None,
    existing: Iterable[str] | None = None,
) -> str:
    """Stable short id like ``btn_a3f2c101``. Collision-safe within a result set."""
    et = type if isinstance(type, ElementType) else ElementType(str(type))
    prefix = _TYPE_PREFIX.get(et, "el")
    h = content_hash(type=et, text=text, role=role, bbox=bbox, native_id=native_id)
    base = f"{prefix}_{h[:8]}"
    used = set(existing or [])
    if base not in used:
        return base
    # Rare collision: extend hash suffix.
    n = 1
    while True:
        cand = f"{base}{n:02d}"
        if cand not in used:
            return cand
        n += 1


def assign_ids(elements: list[UIElement]) -> list[UIElement]:
    """Fill missing element_id / content_hash; keep existing stable ids when set."""
    used: list[str] = []
    out: list[UIElement] = []
    for el in elements:
        ch = el.content_hash or content_hash(
            type=el.type,
            text=el.text or el.name,
            role=el.role,
            bbox=el.bbox,
            enabled=el.enabled,
            native_id=el.native_id,
        )
        eid = el.element_id
        if not eid or eid.startswith("tmp_"):
            eid = make_element_id(
                type=el.type,
                text=el.text or el.name,
                bbox=el.bbox,
                role=el.role,
                native_id=el.native_id,
                existing=used,
            )
        used.append(eid)
        out.append(el.model_copy(update={"element_id": eid, "content_hash": ch}))
    return out


def element_stale(
    previous: UIElement,
    current_by_id: dict[str, UIElement],
    *,
    frame_id: str,
    iou_min: float = 0.35,
) -> bool:
    """True if element is missing, wrong frame, or geometry/text drifted too far."""
    if previous.frame_id and previous.frame_id != frame_id:
        # Frame changed — look for match by id or hash.
        cur = current_by_id.get(previous.element_id)
        if cur is None:
            # Try content hash match.
            for c in current_by_id.values():
                if previous.matches_hash(c, iou_min=iou_min):
                    return False
            return True
        return not previous.matches_hash(cur, iou_min=iou_min)
    cur = current_by_id.get(previous.element_id)
    if cur is None:
        return True
    return not previous.matches_hash(cur, iou_min=iou_min)
