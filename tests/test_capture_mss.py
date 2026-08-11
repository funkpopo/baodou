"""Real mss capture integration tests (Windows desktop)."""

from __future__ import annotations

import time

import pytest
from capture.factory import create_capture
from capture.geometry import cursor_pos_physical, list_monitors
from capture.mss_backend import MssCapture
from capture.pipeline import CapturePipeline
from capture.preprocess import preprocess
from core.config import load_config
from core.models import FrameKind, Point
from PIL import Image


@pytest.fixture
def mss_config():
    cfg = load_config()
    cfg.capture.backend = "mss"
    cfg.capture.mode = "primary"
    cfg.capture.include_b64 = False
    cfg.capture.save_frames = False
    cfg.capture.privacy.enabled = False  # deterministic
    return cfg


def test_mss_capture_primary(mss_config) -> None:
    cap = MssCapture(mss_config)
    try:
        packet = cap.capture_packet(kind=FrameKind.VISION, force=True)
        assert packet.image is not None
        assert packet.meta.width > 0 and packet.meta.height > 0
        assert packet.meta.physical_width and packet.meta.physical_width >= packet.meta.width
        assert packet.meta.capture_ms is not None
        assert packet.meta.scale_x >= 1.0 - 1e-6
        # image center maps inside monitor
        cx = packet.meta.width / 2
        cy = packet.meta.height / 2
        screen = packet.meta.image_to_screen(cx, cy)
        mons = list_monitors()
        primary = next((m for m in mons if m.is_primary), mons[0])
        assert primary.bbox.contains(screen.x, screen.y) or primary.bbox.contains(
            min(screen.x, primary.left + primary.width - 1),
            min(screen.y, primary.top + primary.height - 1),
        )
    finally:
        cap.close()


def test_mss_capture_region(mss_config) -> None:
    mons = list_monitors()
    primary = next((m for m in mons if m.is_primary), mons[0])
    mss_config.capture.mode = "region"
    mss_config.capture.region = {
        "x": primary.left + 10,
        "y": primary.top + 10,
        "width": 320,
        "height": 240,
    }
    cap = MssCapture(mss_config)
    try:
        packet = cap.capture_packet(kind=FrameKind.PREVIEW, force=True)
        assert packet.image is not None
        assert packet.meta.origin_x == primary.left + 10
        assert packet.meta.origin_y == primary.top + 10
        assert packet.meta.physical_width == 320
        assert packet.meta.physical_height == 240
        # top-left of image → region origin
        p0 = packet.meta.image_to_screen(0, 0)
        assert p0 == Point(x=primary.left + 10, y=primary.top + 10)
    finally:
        cap.close()


def test_mss_capture_all_virtual(mss_config) -> None:
    mss_config.capture.mode = "all"
    cap = MssCapture(mss_config)
    try:
        packet = cap.capture_packet(kind=FrameKind.MODEL, force=True)
        assert packet.image is not None
        assert packet.meta.monitor_index == 0
        # dual 1920x1080 → virtual width 3840 typically
        assert packet.meta.physical_width and packet.meta.physical_width >= 1920
    finally:
        cap.close()


def test_coordinate_matches_cursor_when_on_primary(mss_config) -> None:
    """If cursor is on primary, converting a same-offset image point should be consistent."""
    cap = MssCapture(mss_config)
    try:
        packet = cap.capture_packet(kind=FrameKind.VISION, force=True)
        cur = cursor_pos_physical()
        # Map cursor into image space; if inside frame, round-trip
        img_pt = packet.meta.screen_to_image(cur.x, cur.y)
        if 0 <= img_pt.x < packet.meta.width and 0 <= img_pt.y < packet.meta.height:
            back = packet.meta.image_to_screen(img_pt.x, img_pt.y)
            assert abs(back.x - cur.x) <= 2
            assert abs(back.y - cur.y) <= 2
    finally:
        cap.close()


def test_pipeline_bounded_under_backpressure(mss_config) -> None:
    mss_config.capture.queue_size = 2
    mss_config.capture.streams["preview"].fps = 30
    mss_config.capture.streams["preview"].only_on_change = False
    mss_config.capture.streams["vision"].fps = 30
    mss_config.capture.streams["vision"].only_on_change = False
    pipe = CapturePipeline(mss_config)
    try:
        pipe.start()
        time.sleep(0.8)
        # intentionally do not consume → drops should happen
        stats = pipe.stats_dict()
        preview = stats["streams"]["preview"]
        assert preview["size"] <= 2
        assert preview["high_watermark"] <= 2
        # With 30fps and no consumer, drops expected
        assert preview["dropped"] >= 0  # may be 0 if machine slow; size bound is hard req
        assert pipe.stats.captures >= 1
    finally:
        pipe.close()


def test_pipeline_model_event_and_verify(mss_config) -> None:
    mss_config.capture.streams["preview"].fps = 2
    mss_config.capture.streams["vision"].fps = 2
    pipe = CapturePipeline(mss_config)
    try:
        pipe.start()
        pipe.request_model_frame(user_request=True)
        packet = None
        for _ in range(50):
            packet = pipe.latest(FrameKind.MODEL, timeout=0.05)
            if packet is not None:
                break
        assert packet is not None
        assert packet.meta.frame_kind == FrameKind.MODEL
        pipe.request_verify_frame()
        v = None
        for _ in range(50):
            v = pipe.latest(FrameKind.VERIFY, timeout=0.05)
            if v is not None:
                break
        assert v is not None
    finally:
        pipe.close()


def test_create_capture_factory(mss_config) -> None:
    backend = create_capture(mss_config)
    assert isinstance(backend, MssCapture)
    backend.close()


def test_preprocess_encode_jpeg() -> None:
    img = Image.new("RGB", (64, 64), (12, 34, 56))
    pp = preprocess(img, max_width=32, max_height=32, image_format="jpeg", encode=True)
    assert pp.encoded is not None
    assert pp.encoded[:2] == b"\xff\xd8"
    assert pp.width == 32
