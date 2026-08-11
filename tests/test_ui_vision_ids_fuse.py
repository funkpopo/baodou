"""Phase D: element ids, fusion, hierarchy, staleness."""

from __future__ import annotations

from core.models import BBox, ElementType, UIElement, bbox_iou
from ui_vision.fuse import assign_hierarchy, fuse_elements
from ui_vision.ids import assign_ids, content_hash, element_stale, make_element_id


def test_bbox_iou_basic() -> None:
    a = BBox(x=0, y=0, width=100, height=100)
    b = BBox(x=50, y=50, width=100, height=100)
    assert 0.14 < bbox_iou(a, b) < 0.15
    assert bbox_iou(a, a) == 1.0


def test_stable_element_id_and_hash() -> None:
    box = BBox(x=100, y=200, width=80, height=30)
    h1 = content_hash(type=ElementType.BUTTON, text="搜索", role="button", bbox=box)
    h2 = content_hash(type=ElementType.BUTTON, text="搜索", role="button", bbox=box)
    assert h1 == h2
    # Small jitter within quantize grid keeps hash
    box2 = BBox(x=102, y=201, width=81, height=30)
    h3 = content_hash(type=ElementType.BUTTON, text="搜索", role="button", bbox=box2)
    assert h1 == h3
    id1 = make_element_id(type=ElementType.BUTTON, text="搜索", bbox=box, role="button")
    id2 = make_element_id(type=ElementType.BUTTON, text="搜索", bbox=box, role="button")
    assert id1 == id2
    assert id1.startswith("btn_")


def test_fuse_merges_uia_and_ocr() -> None:
    box = BBox(x=10, y=10, width=100, height=40)
    uia = UIElement(
        element_id="tmp_a",
        type=ElementType.BUTTON,
        role="button",
        text="",
        bbox=box,
        confidence=0.85,
        clickable=True,
        source=["uia"],
        frame_id="f1",
    )
    ocr = UIElement(
        element_id="tmp_b",
        type=ElementType.TEXT,
        role="text",
        text="搜索",
        bbox=BBox(x=12, y=12, width=96, height=36),
        confidence=0.7,
        source=["ocr"],
        frame_id="f1",
    )
    fused = fuse_elements([[uia], [ocr]], confidence_threshold=0.5, max_elements=16)
    assert len(fused) == 1
    el = fused[0]
    assert el.text == "搜索"
    assert el.type == ElementType.BUTTON
    assert set(el.source) == {"uia", "ocr"}
    assert el.confidence >= 0.85
    assert el.element_id  # assigned


def test_hierarchy_parent_child() -> None:
    win = UIElement(
        element_id="win",
        type=ElementType.WINDOW,
        bbox=BBox(x=0, y=0, width=800, height=600),
        confidence=0.9,
        source=["uia"],
    )
    btn = UIElement(
        element_id="btn",
        type=ElementType.BUTTON,
        text="OK",
        bbox=BBox(x=100, y=100, width=80, height=30),
        confidence=0.9,
        clickable=True,
        source=["uia"],
    )
    out = assign_hierarchy([win, btn])
    by_id = {e.element_id: e for e in out}
    assert by_id["btn"].parent_id == "win"
    assert by_id["btn"].depth >= 1
    assert by_id["win"].depth == 0


def test_element_stale_detection() -> None:
    box = BBox(x=10, y=10, width=50, height=20)
    prev = UIElement(
        element_id="btn_x",
        type=ElementType.BUTTON,
        text="Go",
        bbox=box,
        confidence=0.9,
        source=["uia"],
        frame_id="f1",
        content_hash=content_hash(type=ElementType.BUTTON, text="Go", role="", bbox=box),
    )
    same = prev.model_copy(update={"frame_id": "f2"})
    assert element_stale(prev, {"btn_x": same}, frame_id="f2") is False

    moved = prev.model_copy(
        update={
            "frame_id": "f2",
            "bbox": BBox(x=400, y=400, width=50, height=20),
            "content_hash": "different",
        }
    )
    assert element_stale(prev, {"btn_x": moved}, frame_id="f2") is True
    assert element_stale(prev, {}, frame_id="f2") is True


def test_assign_ids_collision_safe() -> None:
    box = BBox(x=0, y=0, width=10, height=10)
    els = [
        UIElement(
            element_id="",
            type=ElementType.BUTTON,
            text="A",
            bbox=box,
            confidence=0.9,
            source=["mock"],
        ),
        UIElement(
            element_id="",
            type=ElementType.BUTTON,
            text="A",
            bbox=box,
            confidence=0.9,
            source=["mock"],
        ),
    ]
    out = assign_ids(els)
    assert out[0].element_id != out[1].element_id or out[0].content_hash
    assert len({e.element_id for e in out}) == 2
