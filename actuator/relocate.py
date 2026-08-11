"""Re-locate action targets on a fresh vision result (Phase F).

Prefer element_id; fall back to content_hash / IoU+type+text match.
Bare coordinates only when explicitly allowed and confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import AppConfig
from core.errors import ActuatorError, ErrorCode
from core.models import (
    ActionStep,
    ActionType,
    Point,
    UIElement,
    UIVisionResult,
    bbox_iou,
)


@dataclass
class RelocateResult:
    element: UIElement | None
    point: Point | None
    element_id: str | None
    relocated: bool
    method: str  # id | hash | iou | coordinate | none | wait
    message: str = ""


_POINTLESS = {
    ActionType.NONE,
    ActionType.WAIT,
    ActionType.REIDENTIFY,
    ActionType.KEY,
    ActionType.HOTKEY,
}


def _find_by_id(vision: UIVisionResult, element_id: str) -> UIElement | None:
    return vision.by_id(element_id)


def _find_fuzzy(
    vision: UIVisionResult,
    ref: UIElement | None,
    *,
    element_id: str | None,
    iou_min: float,
) -> UIElement | None:
    """Match by content_hash, then type+text+IoU against prior element snapshot."""
    if ref is not None and ref.content_hash:
        for el in vision.elements:
            if el.content_hash and el.content_hash == ref.content_hash:
                return el
    if ref is not None:
        best: UIElement | None = None
        best_iou = 0.0
        for el in vision.elements:
            if el.type != ref.type:
                continue
            if (el.text or "").strip() != (ref.text or "").strip():
                continue
            iou = bbox_iou(el.bbox, ref.bbox)
            if iou >= iou_min and iou > best_iou:
                best = el
                best_iou = iou
        if best is not None:
            return best
    # Last resort: same id prefix / substring not used — keep strict.
    if element_id:
        # Try text match if id looks like mock stable id disappeared after re-capture.
        pass
    return None


def relocate_target(
    step: ActionStep,
    vision: UIVisionResult,
    config: AppConfig,
    *,
    prior_element: UIElement | None = None,
    coordinate_confirmed: bool = False,
) -> RelocateResult:
    """Resolve the physical point / element for a step on the *current* vision frame."""
    if step.action in _POINTLESS:
        return RelocateResult(
            element=None,
            point=None,
            element_id=None,
            relocated=False,
            method="none",
            message="no target required",
        )

    iou_min = config.actuator.relocate_iou_min
    el: UIElement | None = None
    method = "none"
    relocated = False

    if step.target_element_id:
        el = _find_by_id(vision, step.target_element_id)
        if el is not None:
            method = "id"
        else:
            el = _find_fuzzy(
                vision,
                prior_element,
                element_id=step.target_element_id,
                iou_min=iou_min,
            )
            if el is not None:
                method = (
                    "hash"
                    if prior_element and el.content_hash == prior_element.content_hash
                    else "iou"
                )
                relocated = True

    if el is None and step.target_point is not None:
        # Coordinate path — only if allowed.
        prefer_id = config.agent.prefer_element_id
        needs_confirm = config.agent.coordinate_requires_confirm
        allowed = step.allow_coordinate_fallback or not prefer_id
        if not allowed:
            raise ActuatorError(
                f"禁止裸坐标：缺少 element_id 解析结果 ({step.target_element_id})",
                code=ErrorCode.TARGET_INVALID,
                details={"element_id": step.target_element_id},
            )
        if needs_confirm and not coordinate_confirmed and not step.allow_coordinate_fallback:
            raise ActuatorError(
                "裸坐标需要二次确认",
                code=ErrorCode.CONFIRMATION_REQUIRED,
                details={"point": step.target_point.model_dump()},
            )
        return RelocateResult(
            element=None,
            point=step.target_point,
            element_id=None,
            relocated=False,
            method="coordinate",
            message="using confirmed coordinates",
        )

    if el is None:
        raise ActuatorError(
            f"目标元素不存在或已失效: {step.target_element_id}",
            code=ErrorCode.TARGET_STALE if step.target_element_id else ErrorCode.TARGET_INVALID,
            details={"element_id": step.target_element_id, "frame_id": vision.frame_id},
        )

    if config.actuator.relocate_require_visible and not el.visible:
        raise ActuatorError(
            f"目标元素不可见: {el.element_id}",
            code=ErrorCode.TARGET_STALE,
            details={"element_id": el.element_id},
        )
    if config.actuator.relocate_require_enabled and not el.enabled:
        raise ActuatorError(
            f"目标元素未启用: {el.element_id}",
            code=ErrorCode.TARGET_STALE,
            details={"element_id": el.element_id},
        )

    # Stale vs planning frame
    if (
        prior_element is not None
        and prior_element.frame_id
        and el.frame_id
        and prior_element.frame_id != el.frame_id
    ):
        relocated = True

    return RelocateResult(
        element=el,
        point=el.center,
        element_id=el.element_id,
        relocated=relocated or method != "id",
        method=method,
        message=f"resolved via {method}",
    )
