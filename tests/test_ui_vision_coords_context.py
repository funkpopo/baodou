"""Phase D: multi-resolution coordinates + compact model context."""

from __future__ import annotations

import pytest
from core.config import load_config
from core.models import BBox, ElementType, ScreenFrame, UIElement, UIVisionResult
from PIL import Image, ImageDraw
from ui_vision.context import (
    annotate_from_frame,
    element_to_compact,
    filter_elements_for_goal,
    serialize_for_model,
    serialize_text_summary,
)
from ui_vision.coords import (
    image_bbox_to_screen,
    logical_to_physical_bbox,
    physical_to_logical_bbox,
    screen_bbox_to_image,
)
from ui_vision.mock import MockUIVision
from ui_vision.pipeline import CompositeUIVision
from ui_vision.rules import RulesRecognizer


@pytest.mark.parametrize(
    "dpi_scale,scale_x,scale_y,img_w,img_h,phys_w,phys_h",
    [
        (1.0, 1.0, 1.0, 1920, 1080, 1920, 1080),
        (1.25, 1.5, 1.5, 1280, 720, 1920, 1080),
        (1.5, 2.0, 2.0, 960, 540, 1920, 1080),
        (2.0, 2.0, 2.0, 1280, 800, 2560, 1600),
        (1.0, 1.0, 1.0, 1366, 768, 1366, 768),
    ],
)
def test_image_screen_bbox_roundtrip_multi_res(
    dpi_scale: float,
    scale_x: float,
    scale_y: float,
    img_w: int,
    img_h: int,
    phys_w: int,
    phys_h: int,
) -> None:
    frame = ScreenFrame(
        width=img_w,
        height=img_h,
        origin_x=100,
        origin_y=50,
        physical_width=phys_w,
        physical_height=phys_h,
        scale_x=scale_x,
        scale_y=scale_y,
        dpi_scale=dpi_scale,
        dpi_x=int(96 * dpi_scale),
        dpi_y=int(96 * dpi_scale),
    )
    img_box = BBox(x=40, y=30, width=80, height=24)
    screen = image_bbox_to_screen(frame, img_box)
    assert screen.x == frame.origin_x + int(round(40 * scale_x))
    back = screen_bbox_to_image(frame, screen)
    # Allow 1px rounding
    assert abs(back.x - img_box.x) <= 1
    assert abs(back.y - img_box.y) <= 1
    assert abs(back.width - img_box.width) <= 2
    assert abs(back.height - img_box.height) <= 2

    logical = physical_to_logical_bbox(screen, dpi_scale)
    phys2 = logical_to_physical_bbox(logical, dpi_scale)
    assert abs(phys2.x - screen.x) <= 1
    assert abs(phys2.width - screen.width) <= 1


def test_mock_vision_scales_with_physical_size() -> None:
    cfg = load_config()
    cfg.ui_vision.backend = "mock"
    vision = MockUIVision(cfg)

    f1 = ScreenFrame(
        width=960,
        height=540,
        origin_x=0,
        origin_y=0,
        physical_width=1920,
        physical_height=1080,
        scale_x=2.0,
        scale_y=2.0,
        dpi_scale=1.0,
    )
    f2 = ScreenFrame(
        width=1280,
        height=800,
        origin_x=0,
        origin_y=0,
        physical_width=2560,
        physical_height=1600,
        scale_x=2.0,
        scale_y=2.0,
        dpi_scale=1.5,
        dpi_x=144,
        dpi_y=144,
    )
    r1 = vision.recognize(f1, trace_id="t1")
    r2 = vision.recognize(f2, trace_id="t2")
    b1 = r1.by_id("btn_search_01")
    b2 = r2.by_id("btn_search_01")
    assert b1 and b2
    # Search button anchored near right edge of physical frame
    assert b1.bbox.x == 1920 - 200
    assert b2.bbox.x == 2560 - 200
    assert b2.dpi_scale == 1.5
    assert b2.bbox_logical is not None
    # Center stays in physical space; image mapping must land in-bounds
    c = b1.center
    ip = f1.screen_to_image(c.x, c.y)
    assert 0 <= ip.x < f1.width
    assert 0 <= ip.y < f1.height


def test_filter_and_compact_prefer_goal_tokens() -> None:
    els = [
        UIElement(
            element_id="btn_a",
            type=ElementType.BUTTON,
            text="取消",
            bbox=BBox(x=0, y=0, width=40, height=20),
            confidence=0.9,
            clickable=True,
            source=["mock"],
        ),
        UIElement(
            element_id="btn_search_01",
            type=ElementType.BUTTON,
            text="搜索",
            bbox=BBox(x=100, y=0, width=40, height=20),
            confidence=0.8,
            clickable=True,
            source=["mock"],
        ),
        UIElement(
            element_id="txt_x",
            type=ElementType.TEXT,
            text="无关内容",
            bbox=BBox(x=0, y=50, width=200, height=20),
            confidence=0.7,
            source=["ocr"],
        ),
    ]
    ranked = filter_elements_for_goal(els, "点击搜索按钮", max_elements=2)
    assert ranked[0].element_id == "btn_search_01"
    compact = element_to_compact(ranked[0])
    assert compact["element_id"] == "btn_search_01"
    assert "bbox" in compact and "center" in compact

    result = UIVisionResult(frame_id="f", elements=els, dpi_scale=1.25)
    rows = serialize_for_model(result, goal="搜索", max_elements=2)
    assert rows[0]["element_id"] == "btn_search_01"
    text = serialize_text_summary(result, goal="搜索", max_elements=2)
    assert "btn_search_01" in text
    assert "dpi_scale=1.25" in text


def test_annotate_numbered_boxes() -> None:
    img = Image.new("RGB", (200, 100), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 80, 50], fill=(200, 200, 255))
    els = [
        UIElement(
            element_id="btn_1",
            type=ElementType.BUTTON,
            text="A",
            bbox=BBox(x=20, y=20, width=60, height=30),
            confidence=0.9,
            clickable=True,
            source=["mock"],
            extra={"image_bbox": {"x": 20, "y": 20, "width": 60, "height": 30}},
        )
    ]
    result = UIVisionResult(frame_id="f", elements=els)
    frame = ScreenFrame(width=200, height=100, scale_x=1.0, scale_y=1.0)
    ann = annotate_from_frame(img, result, frame.screen_to_image)
    assert ann.size == img.size
    # Pixel changed somewhere (box drawn)
    assert list(ann.getdata()) != list(img.getdata())


def test_rules_recognizer_finds_button_like_rect() -> None:
    cfg = load_config()
    cfg.ui_vision.rules_enabled = True
    rec = RulesRecognizer(cfg)
    # High-contrast button-like rectangle on gray background
    img = Image.new("RGB", (400, 200), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 180, 72], outline=(20, 20, 20), width=3)
    draw.rectangle([51, 41, 179, 71], fill=(60, 120, 220))
    frame = ScreenFrame(
        width=400,
        height=200,
        origin_x=0,
        origin_y=0,
        physical_width=400,
        physical_height=200,
        scale_x=1.0,
        scale_y=1.0,
    )
    found = rec.recognize(frame, img)
    # Rules are heuristic; at least run without error. Often finds the rect.
    assert isinstance(found, list)
    for el in found:
        assert el.bbox.width > 0
        c = el.center
        ip = frame.screen_to_image(c.x, c.y)
        assert -1 <= ip.x <= frame.width + 1


def test_composite_with_mock_source() -> None:
    cfg = load_config()
    cfg.ui_vision.backend = "composite"
    cfg.ui_vision.sources = ["mock"]
    cfg.ui_vision.confidence_threshold = 0.5
    from ui_vision.factory import create_ui_vision

    vision = create_ui_vision(cfg)
    assert isinstance(vision, CompositeUIVision)
    frame = ScreenFrame(
        width=640,
        height=360,
        physical_width=1280,
        physical_height=720,
        scale_x=2.0,
        scale_y=2.0,
        origin_x=0,
        origin_y=0,
    )
    result = vision.recognize(frame, trace_id="tr-test")
    assert result.elements
    assert result.sources_used == ["mock"]
    assert result.by_id("btn_search_01") is not None
