"""Multi-resolution coordinate helpers for UI vision.

All ``UIElement.bbox`` values must be **virtual-desktop physical pixels**.
Image-space detections (OCR / vision rules) convert through the frame's
``origin`` + ``scale_x/y``. Logical (DIP) boxes are derived from ``dpi_scale``.
"""

from __future__ import annotations

from core.models import BBox, Point, ScreenFrame, UIElement


def image_bbox_to_screen(frame: ScreenFrame, box: BBox) -> BBox:
    """Map an axis-aligned box in stored-image pixels → physical screen pixels."""
    tl = frame.image_to_screen(box.x, box.y)
    br = frame.image_to_screen(box.x + box.width, box.y + box.height)
    return BBox(
        x=min(tl.x, br.x),
        y=min(tl.y, br.y),
        width=abs(br.x - tl.x),
        height=abs(br.y - tl.y),
    )


def screen_bbox_to_image(frame: ScreenFrame, box: BBox) -> BBox:
    """Map physical screen box → stored-image pixels."""
    tl = frame.screen_to_image(box.x, box.y)
    br = frame.screen_to_image(box.x + box.width, box.y + box.height)
    return BBox(
        x=min(tl.x, br.x),
        y=min(tl.y, br.y),
        width=max(0, abs(br.x - tl.x)),
        height=max(0, abs(br.y - tl.y)),
    )


def physical_to_logical_bbox(box: BBox, dpi_scale: float) -> BBox:
    if dpi_scale <= 0:
        dpi_scale = 1.0
    return BBox(
        x=int(round(box.x / dpi_scale)),
        y=int(round(box.y / dpi_scale)),
        width=max(0, int(round(box.width / dpi_scale))),
        height=max(0, int(round(box.height / dpi_scale))),
    )


def logical_to_physical_bbox(box: BBox, dpi_scale: float) -> BBox:
    if dpi_scale <= 0:
        dpi_scale = 1.0
    return BBox(
        x=int(round(box.x * dpi_scale)),
        y=int(round(box.y * dpi_scale)),
        width=max(0, int(round(box.width * dpi_scale))),
        height=max(0, int(round(box.height * dpi_scale))),
    )


def attach_dpi_fields(el: UIElement, frame: ScreenFrame) -> UIElement:
    """Ensure element carries DPI metadata and optional logical bbox from the frame."""
    scale = frame.dpi_scale or 1.0
    logical = el.bbox_logical or physical_to_logical_bbox(el.bbox, scale)
    return el.model_copy(
        update={
            "dpi_scale": scale,
            "dpi_x": frame.dpi_x or 96,
            "dpi_y": frame.dpi_y or 96,
            "bbox_logical": logical,
            "frame_id": el.frame_id or frame.frame_id,
        }
    )


def clamp_to_frame_region(box: BBox, frame: ScreenFrame) -> BBox:
    """Clamp physical box to the frame's capture region on the virtual desktop."""
    pw = (
        frame.physical_width
        if frame.physical_width is not None
        else int(round(frame.width * frame.scale_x))
    )
    ph = (
        frame.physical_height
        if frame.physical_height is not None
        else int(round(frame.height * frame.scale_y))
    )
    bounds = BBox(x=frame.origin_x, y=frame.origin_y, width=pw, height=ph)
    return box.clamp(bounds)


def center_in_physical(el: UIElement) -> Point:
    return el.center


def scale_invariant_key(box: BBox, *, grid: int = 32) -> tuple[int, int, int, int]:
    """Coarse grid key for cross-resolution matching (relative layout)."""
    return (
        box.x // grid,
        box.y // grid,
        max(1, box.width // grid),
        max(1, box.height // grid),
    )
