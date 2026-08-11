"""Change detection, bounded queue, privacy mask tests."""

from __future__ import annotations

from capture.change import ChangeDetector
from capture.privacy import apply_masks
from capture.queue import FrameQueue
from core.models import SensitiveRegion
from PIL import Image


def test_change_detector_first_frame_changed() -> None:
    det = ChangeDetector(sample_size=32, threshold=0.01)
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    changed, score, h = det.update(img)
    assert changed is True
    assert score == 1.0
    assert len(h) == 32


def test_change_detector_identical_not_changed() -> None:
    det = ChangeDetector(sample_size=32, threshold=0.02)
    img = Image.new("RGB", (200, 100), (10, 20, 30))
    det.update(img)
    changed, score, _ = det.update(img.copy())
    assert changed is False
    assert score < 0.02


def test_change_detector_large_diff() -> None:
    det = ChangeDetector(sample_size=32, threshold=0.02)
    a = Image.new("RGB", (200, 100), (0, 0, 0))
    b = Image.new("RGB", (200, 100), (255, 255, 255))
    det.update(a)
    changed, score, _ = det.update(b)
    assert changed is True
    assert score > 0.5


def test_frame_queue_drops_oldest_under_pressure() -> None:
    q: FrameQueue[int] = FrameQueue(maxsize=2, drop_policy="newest", name="t")
    assert q.put(1)
    assert q.put(2)
    assert q.put(3)  # drops 1
    assert len(q) == 2
    assert q.get() == 2
    assert q.get() == 3
    assert q.stats.dropped == 1
    assert q.stats.high_watermark == 2


def test_frame_queue_drop_incoming_policy() -> None:
    q: FrameQueue[int] = FrameQueue(maxsize=1, drop_policy="oldest", name="t")
    assert q.put(1)
    assert q.put(2) is False
    assert q.get() == 1
    assert q.stats.dropped == 1


def test_frame_queue_bounded_no_growth() -> None:
    q: FrameQueue[int] = FrameQueue(maxsize=4, drop_policy="newest", name="stress")
    for i in range(1000):
        q.put(i)
    assert len(q) == 4
    assert q.stats.dropped == 996
    assert q.stats.high_watermark == 4


def test_privacy_mask_blacks_out_region() -> None:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    regions = [SensitiveRegion(x=10, y=10, width=20, height=20, reason="manual")]
    # origin 0,0 scale 1 → image coords match physical
    out, applied = apply_masks(
        img,
        regions=regions,
        origin_x=0,
        origin_y=0,
        scale_x=1.0,
        scale_y=1.0,
        mask_color=(0, 0, 0),
    )
    assert len(applied) == 1
    assert out.getpixel((15, 15)) == (0, 0, 0)
    assert out.getpixel((50, 50)) == (255, 255, 255)


def test_privacy_mask_with_scale() -> None:
    # physical 200x200 stored as 100x100 → scale 2
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    regions = [SensitiveRegion(x=0, y=0, width=40, height=40, reason="manual")]
    out, applied = apply_masks(
        img,
        regions=regions,
        origin_x=0,
        origin_y=0,
        scale_x=2.0,
        scale_y=2.0,
        mask_color=(0, 0, 0),
    )
    assert applied
    # physical 40 → image 20
    assert out.getpixel((5, 5)) == (0, 0, 0)
    assert out.getpixel((50, 50)) == (255, 0, 0)
