"""Sensitive region masking before frames leave the capture layer."""

from __future__ import annotations

from core.config import PrivacySection
from core.models import BBox, SensitiveRegion
from PIL import Image, ImageDraw

from capture.window_win import find_privacy_windows


def _mask_color(cfg: PrivacySection) -> tuple[int, int, int]:
    c = cfg.mask_color or [0, 0, 0]
    if len(c) >= 3:
        return int(c[0]), int(c[1]), int(c[2])
    return 0, 0, 0


def manual_regions(cfg: PrivacySection) -> list[SensitiveRegion]:
    out: list[SensitiveRegion] = []
    for raw in cfg.manual_masks:
        try:
            out.append(
                SensitiveRegion(
                    x=int(raw["x"]),
                    y=int(raw["y"]),
                    width=int(raw["width"]),
                    height=int(raw["height"]),
                    reason=str(raw.get("reason") or "manual"),
                    absolute=bool(raw.get("absolute", True)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def privacy_window_regions(cfg: PrivacySection) -> list[SensitiveRegion]:
    regions: list[SensitiveRegion] = []
    try:
        wins = find_privacy_windows(cfg.privacy_window_titles)
    except Exception:  # noqa: BLE001 — never fail capture on privacy scan
        return regions
    for w in wins:
        regions.append(
            SensitiveRegion(
                x=w.left,
                y=w.top,
                width=w.width,
                height=w.height,
                reason="privacy_window",
                absolute=True,
            )
        )
    return regions


def collect_mask_regions(
    cfg: PrivacySection,
    *,
    extra: list[SensitiveRegion] | None = None,
    scan_privacy_windows: bool = True,
) -> list[SensitiveRegion]:
    if not cfg.enabled:
        return list(extra or [])
    regions = manual_regions(cfg)
    if scan_privacy_windows:
        regions.extend(privacy_window_regions(cfg))
    if extra:
        regions.extend(extra)
    return regions


def apply_masks(
    image: Image.Image,
    *,
    regions: list[SensitiveRegion],
    origin_x: int,
    origin_y: int,
    scale_x: float,
    scale_y: float,
    mask_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, list[SensitiveRegion]]:
    """
    Mask regions defined in virtual-desktop physical pixels onto the (possibly scaled) image.
    Returns (masked_image, regions_that_intersected).
    """
    if not regions:
        return image, []
    img = image.copy()
    draw = ImageDraw.Draw(img)
    applied: list[SensitiveRegion] = []
    for reg in regions:
        # physical → image coords
        ix0 = (reg.x - origin_x) / scale_x if scale_x else 0
        iy0 = (reg.y - origin_y) / scale_y if scale_y else 0
        ix1 = (reg.x + reg.width - origin_x) / scale_x if scale_x else 0
        iy1 = (reg.y + reg.height - origin_y) / scale_y if scale_y else 0
        x0, x1 = sorted((int(ix0), int(ix1)))
        y0, y1 = sorted((int(iy0), int(iy1)))
        # clamp to image
        x0 = max(0, min(img.width, x0))
        x1 = max(0, min(img.width, x1))
        y0 = max(0, min(img.height, y0))
        y1 = max(0, min(img.height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=mask_color)
        applied.append(reg)
    return img, applied


def mask_full_frame(image: Image.Image, color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    img = Image.new(image.mode, image.size, color)
    return img


def region_relative_to_capture(reg: SensitiveRegion, capture: BBox) -> BBox | None:
    inter = reg.as_bbox().clamp(capture)
    if inter.width <= 0 or inter.height <= 0:
        return None
    return inter
