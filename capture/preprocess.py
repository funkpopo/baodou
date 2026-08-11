"""Scale, color convert, compress, ROI crop for capture frames."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

from core.models import BBox, FrameKind
from PIL import Image


@dataclass
class PreprocessResult:
    image: Image.Image
    scale_x: float
    scale_y: float
    width: int
    height: int
    format: str
    encoded: bytes | None
    preprocess_ms: float
    roi_applied: BBox | None = None


def fit_size(
    src_w: int,
    src_h: int,
    max_w: int,
    max_h: int,
) -> tuple[int, int, float, float]:
    """Return (dst_w, dst_h, scale_x, scale_y) preserving aspect, never upscale."""
    if src_w <= 0 or src_h <= 0:
        return 1, 1, 1.0, 1.0
    if max_w <= 0 or max_h <= 0:
        return src_w, src_h, 1.0, 1.0
    ratio = min(max_w / src_w, max_h / src_h, 1.0)
    dst_w = max(1, int(round(src_w * ratio)))
    dst_h = max(1, int(round(src_h * ratio)))
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    return dst_w, dst_h, scale_x, scale_y


def crop_roi(image: Image.Image, roi: BBox | None) -> tuple[Image.Image, BBox | None]:
    if roi is None:
        return image, None
    w, h = image.size
    box = roi.clamp(BBox(x=0, y=0, width=w, height=h))
    if box.width <= 0 or box.height <= 0:
        return image, None
    cropped = image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
    return cropped, box


def to_color_mode(image: Image.Image, mode: str) -> Image.Image:
    target = mode.upper()
    if image.mode == target:
        return image
    if target == "L":
        return image.convert("L")
    return image.convert("RGB")


def encode_image(
    image: Image.Image,
    *,
    fmt: str = "png",
    jpeg_quality: int = 85,
) -> bytes:
    buf = io.BytesIO()
    f = fmt.lower()
    if f in ("jpg", "jpeg"):
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def preprocess(
    image: Image.Image,
    *,
    max_width: int = 1280,
    max_height: int = 720,
    color_mode: str = "RGB",
    image_format: str = "png",
    jpeg_quality: int = 85,
    roi: BBox | None = None,
    encode: bool = False,
    frame_kind: FrameKind | None = None,  # reserved for kind-specific tweaks
) -> PreprocessResult:
    t0 = time.perf_counter()
    img, roi_applied = crop_roi(image, roi)
    img = to_color_mode(img, color_mode)
    src_w, src_h = img.size
    dst_w, dst_h, scale_x, scale_y = fit_size(src_w, src_h, max_width, max_height)
    if (dst_w, dst_h) != (src_w, src_h):
        img = img.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    encoded = encode_image(img, fmt=image_format, jpeg_quality=jpeg_quality) if encode else None
    return PreprocessResult(
        image=img,
        scale_x=scale_x,
        scale_y=scale_y,
        width=dst_w,
        height=dst_h,
        format=image_format.lower(),
        encoded=encoded,
        preprocess_ms=(time.perf_counter() - t0) * 1000,
        roi_applied=roi_applied,
    )
