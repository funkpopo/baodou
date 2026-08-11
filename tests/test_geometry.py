"""Coordinate / DPI geometry unit tests."""

from __future__ import annotations

import pytest
from capture.geometry import list_monitors, virtual_desktop
from capture.preprocess import fit_size
from core.models import BBox, Point, ScreenFrame


def test_fit_size_no_upscale() -> None:
    w, h, sx, sy = fit_size(800, 600, 1920, 1080)
    assert (w, h) == (800, 600)
    assert sx == pytest.approx(1.0)
    assert sy == pytest.approx(1.0)


def test_fit_size_downscale() -> None:
    w, h, sx, sy = fit_size(1920, 1080, 1280, 720)
    assert w == 1280 and h == 720
    assert sx == pytest.approx(1920 / 1280)
    assert sy == pytest.approx(1080 / 720)


def test_screen_frame_coord_roundtrip() -> None:
    frame = ScreenFrame(
        width=640,
        height=360,
        origin_x=100,
        origin_y=50,
        physical_width=1280,
        physical_height=720,
        scale_x=2.0,
        scale_y=2.0,
    )
    screen = frame.image_to_screen(10, 20)
    assert screen == Point(x=120, y=90)
    back = frame.screen_to_image(screen.x, screen.y)
    assert back == Point(x=10, y=20)


def test_bbox_clamp() -> None:
    outer = BBox(x=0, y=0, width=100, height=100)
    inner = BBox(x=80, y=80, width=50, height=50)
    c = inner.clamp(outer)
    assert c == BBox(x=80, y=80, width=20, height=20)


def test_list_monitors_smoke() -> None:
    mons = list_monitors()
    assert len(mons) >= 1
    vd = virtual_desktop()
    assert vd.width >= 800
    assert vd.height >= 600
