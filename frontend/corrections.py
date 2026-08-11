"""User target corrections as structured task context (Phase H)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.models import (
    BBox,
    CorrectionKind,
    Point,
    UIElement,
    UIVisionResult,
    UserCorrection,
    bbox_iou,
)


@dataclass
class CorrectionStore:
    """Mutable store of user corrections for the active task session."""

    items: list[UserCorrection] = field(default_factory=list)

    def clear(self) -> None:
        self.items.clear()

    def add(self, correction: UserCorrection) -> UserCorrection:
        self.items.append(correction)
        return correction

    def reject_element(self, element_id: str, *, note: str = "") -> UserCorrection:
        return self.add(
            UserCorrection(
                kind=CorrectionKind.REJECT_ELEMENT,
                element_id=element_id,
                note=note or f"不是这个: {element_id}",
            )
        )

    def prefer_element(self, element_id: str, *, note: str = "") -> UserCorrection:
        return self.add(
            UserCorrection(
                kind=CorrectionKind.PREFER_ELEMENT,
                element_id=element_id,
                note=note or f"点击这个: {element_id}",
            )
        )

    def click_here(self, x: int, y: int, *, note: str = "") -> UserCorrection:
        return self.add(
            UserCorrection(
                kind=CorrectionKind.CLICK_HERE,
                point=Point(x=int(x), y=int(y)),
                note=note or f"点击这里: ({x},{y})",
            )
        )

    def ignore_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        note: str = "",
    ) -> UserCorrection:
        return self.add(
            UserCorrection(
                kind=CorrectionKind.IGNORE_REGION,
                region=BBox(x=int(x), y=int(y), width=int(width), height=int(height)),
                note=note or f"忽略区域: ({x},{y},{width},{height})",
            )
        )

    def note(self, text: str) -> UserCorrection:
        return self.add(UserCorrection(kind=CorrectionKind.NOTE, note=text))

    def rejected_ids(self) -> set[str]:
        return {
            c.element_id
            for c in self.items
            if c.kind == CorrectionKind.REJECT_ELEMENT and c.element_id
        }

    def preferred_ids(self) -> list[str]:
        return [
            c.element_id
            for c in self.items
            if c.kind == CorrectionKind.PREFER_ELEMENT and c.element_id
        ]

    def ignore_regions(self) -> list[BBox]:
        return [c.region for c in self.items if c.kind == CorrectionKind.IGNORE_REGION and c.region]

    def click_points(self) -> list[Point]:
        return [c.point for c in self.items if c.kind == CorrectionKind.CLICK_HERE and c.point]

    def to_list(self) -> list[UserCorrection]:
        return list(self.items)

    def log_summary(self) -> list[dict[str, Any]]:
        return [c.log_summary() for c in self.items]


def apply_corrections_to_goal(user_goal: str, corrections: list[UserCorrection]) -> str:
    """Append structured correction hints to the natural-language goal.

    The result is still *context for planning*, never raw OS input.
    """
    if not corrections:
        return user_goal
    parts: list[str] = [user_goal.rstrip()]
    parts.append("\n[用户修正]")
    for c in corrections:
        if c.kind == CorrectionKind.REJECT_ELEMENT:
            parts.append(f"- 不要使用元素 {c.element_id}" + (f"（{c.note}）" if c.note else ""))
        elif c.kind == CorrectionKind.PREFER_ELEMENT:
            parts.append(f"- 优先使用元素 {c.element_id}" + (f"（{c.note}）" if c.note else ""))
        elif c.kind == CorrectionKind.CLICK_HERE and c.point:
            parts.append(f"- 用户指定点击点 ({c.point.x},{c.point.y})")
        elif c.kind == CorrectionKind.IGNORE_REGION and c.region:
            r = c.region
            parts.append(f"- 忽略区域 x={r.x} y={r.y} w={r.width} h={r.height}")
        elif c.kind == CorrectionKind.NOTE and c.note:
            parts.append(f"- 备注: {c.note[:120]}")
    return "\n".join(parts)


def filter_elements_by_corrections(
    elements: list[UIElement],
    corrections: list[UserCorrection],
    *,
    iou_ignore: float = 0.3,
) -> list[UIElement]:
    """Drop rejected elements and those overlapping ignore regions."""
    if not corrections:
        return list(elements)
    rejected = {
        c.element_id
        for c in corrections
        if c.kind == CorrectionKind.REJECT_ELEMENT and c.element_id
    }
    regions = [c.region for c in corrections if c.kind == CorrectionKind.IGNORE_REGION and c.region]
    out: list[UIElement] = []
    for el in elements:
        if el.element_id in rejected:
            continue
        skip = False
        for reg in regions:
            if bbox_iou(el.bbox, reg) >= iou_ignore:
                skip = True
                break
            # Also skip if center is inside ignore region
            cx, cy = el.bbox.center()
            if reg.x <= cx <= reg.x + reg.width and reg.y <= cy <= reg.y + reg.height:
                skip = True
                break
        if not skip:
            out.append(el)
    # Preferred first
    preferred = [
        c.element_id
        for c in corrections
        if c.kind == CorrectionKind.PREFER_ELEMENT and c.element_id
    ]
    if preferred:
        pref_set = set(preferred)
        head = [e for e in out if e.element_id in pref_set]
        tail = [e for e in out if e.element_id not in pref_set]
        # Stable order by preference list
        head_sorted = sorted(
            head,
            key=lambda e: preferred.index(e.element_id) if e.element_id in preferred else 999,
        )
        return head_sorted + tail
    return out


def apply_corrections_to_vision(
    vision: UIVisionResult,
    corrections: list[UserCorrection],
) -> UIVisionResult:
    """Return a shallow-copied vision result with filtered/reordered elements."""
    if not corrections:
        return vision
    filtered = filter_elements_by_corrections(vision.elements, corrections)
    data = vision.model_dump()
    data["elements"] = [e.model_dump() for e in filtered]
    note = vision.notes or ""
    extra = f"corrections={len(corrections)} kept={len(filtered)}/{len(vision.elements)}"
    data["notes"] = f"{note}; {extra}".strip("; ")
    return UIVisionResult.model_validate(data)
