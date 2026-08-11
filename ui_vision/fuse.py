"""Multi-source dedup, fusion, confidence calibration, and hierarchy."""

from __future__ import annotations

from core.models import BBox, ElementType, UIElement, bbox_iou

from ui_vision.ids import assign_ids, content_hash

# Prefer richer / more trusted sources when merging.
_SOURCE_WEIGHT: dict[str, float] = {
    "uia": 1.0,
    "ocr": 0.75,
    "rules": 0.55,
    "mock": 0.9,
}

_TYPE_RANK = {
    ElementType.WINDOW: 0,
    ElementType.DIALOG: 1,
    ElementType.MENU: 2,
    ElementType.TAB: 3,
    ElementType.LIST: 4,
    ElementType.TABLE: 4,
    ElementType.BUTTON: 5,
    ElementType.INPUT: 5,
    ElementType.CHECKBOX: 5,
    ElementType.RADIO: 5,
    ElementType.LINK: 5,
    ElementType.MENU_ITEM: 5,
    ElementType.ICON: 6,
    ElementType.TEXT: 7,
    ElementType.IMAGE: 7,
    ElementType.OTHER: 8,
}


def _src_weight(sources: list[str]) -> float:
    if not sources:
        return 0.4
    return max(_SOURCE_WEIGHT.get(s, 0.5) for s in sources)


def _merge_pair(a: UIElement, b: UIElement) -> UIElement:
    """Merge two overlapping detections of the same control."""
    # Prefer higher-weight source for role/type; union sources.
    wa, wb = _src_weight(a.source), _src_weight(b.source)
    primary, secondary = (a, b) if wa >= wb else (b, a)

    sources = list(dict.fromkeys([*primary.source, *secondary.source]))
    text = primary.text or secondary.text
    name = primary.name or secondary.name
    role = primary.role or secondary.role
    # Prefer non-OTHER type from higher weight; else more specific.
    et = primary.type
    if primary.type == ElementType.OTHER and secondary.type != ElementType.OTHER:
        et = secondary.type
    if _TYPE_RANK.get(secondary.type, 9) < _TYPE_RANK.get(et, 9) and wb >= wa * 0.9:
        et = secondary.type

    # Geometry: weighted average of centers, max extent loosely via IoU primary box
    # Prefer UIA box when present.
    if "uia" in a.source and "uia" not in b.source:
        box = a.bbox
    elif "uia" in b.source and "uia" not in a.source:
        box = b.bbox
    else:
        box = primary.bbox

    conf = min(1.0, max(a.confidence, b.confidence) + 0.05 * (len(sources) - 1))
    conflict = a.type != b.type and a.type != ElementType.OTHER and b.type != ElementType.OTHER
    needs_review = conflict or conf < 0.55 or (a.needs_review or b.needs_review)

    clickable = primary.clickable or secondary.clickable
    editable = primary.editable or secondary.editable
    enabled = primary.enabled and secondary.enabled
    visible = primary.visible or secondary.visible
    native = primary.native_id or secondary.native_id
    parent = primary.parent_id or secondary.parent_id
    depth = (
        min(primary.depth, secondary.depth)
        if primary.depth and secondary.depth
        else max(primary.depth, secondary.depth)
    )

    ch = content_hash(
        type=et,
        text=text or name,
        role=role,
        bbox=box,
        enabled=enabled,
        native_id=native,
    )
    return UIElement(
        element_id=primary.element_id or secondary.element_id or "tmp_merge",
        type=et,
        role=role,
        text=text,
        name=name,
        bbox=box,
        bbox_logical=primary.bbox_logical or secondary.bbox_logical,
        confidence=conf,
        visible=visible,
        enabled=enabled,
        clickable=clickable,
        editable=editable,
        source=sources,
        frame_id=primary.frame_id or secondary.frame_id,
        parent_id=parent,
        depth=depth,
        z_order=max(primary.z_order, secondary.z_order),
        content_hash=ch,
        dpi_scale=primary.dpi_scale or secondary.dpi_scale,
        dpi_x=primary.dpi_x or secondary.dpi_x,
        dpi_y=primary.dpi_y or secondary.dpi_y,
        needs_review=needs_review,
        conflict=conflict,
        native_id=native,
        extra={**secondary.extra, **primary.extra},
    )


def fuse_elements(
    batches: list[list[UIElement]],
    *,
    iou_threshold: float = 0.45,
    confidence_threshold: float = 0.5,
    max_elements: int = 64,
) -> list[UIElement]:
    """Greedy IoU clustering across sources → calibrated element list."""
    flat: list[UIElement] = []
    for batch in batches:
        flat.extend(batch)
    if not flat:
        return []

    # Sort: higher confidence / weight first so they seed clusters.
    flat.sort(key=lambda e: (_src_weight(e.source), e.confidence), reverse=True)

    clusters: list[UIElement] = []
    for el in flat:
        merged = False
        for i, seed in enumerate(clusters):
            if bbox_iou(el.bbox, seed.bbox) >= iou_threshold:
                # Only merge if types compatible or one is OTHER/TEXT filler.
                compatible = (
                    el.type == seed.type
                    or el.type == ElementType.OTHER
                    or seed.type == ElementType.OTHER
                    or (
                        el.type == ElementType.TEXT
                        and seed.type
                        in {
                            ElementType.BUTTON,
                            ElementType.LINK,
                            ElementType.MENU_ITEM,
                            ElementType.TAB,
                        }
                    )
                    or (
                        seed.type == ElementType.TEXT
                        and el.type
                        in {
                            ElementType.BUTTON,
                            ElementType.LINK,
                            ElementType.MENU_ITEM,
                            ElementType.TAB,
                        }
                    )
                )
                if compatible:
                    clusters[i] = _merge_pair(seed, el)
                    merged = True
                    break
        if not merged:
            clusters.append(el)

    # Confidence calibration: multi-source boost already applied; clip + filter.
    calibrated: list[UIElement] = []
    for el in clusters:
        conf = el.confidence
        if len(el.source) >= 2:
            conf = min(1.0, conf + 0.03)
        if conf < confidence_threshold:
            el = el.model_copy(update={"needs_review": True, "confidence": conf})
            # Drop very low confidence unless UIA (trusted structure).
            if conf < confidence_threshold * 0.6 and "uia" not in el.source:
                continue
        else:
            el = el.model_copy(update={"confidence": conf})
        calibrated.append(el)

    # Hierarchy: parent by containment of larger boxes.
    calibrated = assign_hierarchy(calibrated)
    calibrated = assign_ids(calibrated)

    # Prefer interactive + higher conf; cap count.
    calibrated.sort(
        key=lambda e: (
            0 if e.clickable or e.editable else 1,
            _TYPE_RANK.get(e.type, 9),
            -e.confidence,
            e.depth,
        )
    )
    return calibrated[:max_elements]


def assign_hierarchy(elements: list[UIElement]) -> list[UIElement]:
    """Set parent_id / depth from geometric containment (window → container → control)."""
    if not elements:
        return elements
    # Larger area first as potential parents.
    by_area = sorted(elements, key=lambda e: e.bbox.width * e.bbox.height, reverse=True)
    parents: dict[int, int | None] = {id(e): None for e in by_area}

    for i, child in enumerate(by_area):
        cbox = child.bbox
        carea = max(1, cbox.width * cbox.height)
        best: tuple[int, int] | None = None  # (area, parent_obj_id)
        for parent in by_area[:i]:
            pbox = parent.bbox
            # child center inside parent and parent substantially larger
            cx, cy = cbox.center()
            if not pbox.contains(cx, cy):
                continue
            parea = max(1, pbox.width * pbox.height)
            if parea < carea * 1.15:
                continue
            # Prefer tightest parent
            if best is None or parea < best[0]:
                best = (parea, id(parent))
        if best is not None:
            parents[id(child)] = best[1]

    # Depth from roots
    depth_cache: dict[int, int] = {}

    def _depth(oid: int) -> int:
        if oid in depth_cache:
            return depth_cache[oid]
        p = parents.get(oid)
        if p is None:
            depth_cache[oid] = 0
            return 0
        depth_cache[oid] = _depth(p) + 1
        return depth_cache[oid]

    # Need temporary ids for parent links before final assign_ids.
    tmp_ids: dict[int, str] = {}
    for i, e in enumerate(by_area):
        tmp_ids[id(e)] = e.element_id or f"tmp_{i}"

    out: list[UIElement] = []
    for e in elements:
        oid = id(e)
        # Re-find in by_area identity — elements list may be same objects
        p_oid = parents.get(oid)
        parent_id = tmp_ids.get(p_oid) if p_oid is not None else None
        # If object identity lost (copies), recompute via geometry on final list later.
        d = _depth(oid) if oid in parents else e.depth
        eid = tmp_ids.get(oid, e.element_id or "tmp_x")
        out.append(
            e.model_copy(
                update={
                    "element_id": eid,
                    "parent_id": parent_id if parent_id != eid else None,
                    "depth": d,
                }
            )
        )

    # Second pass: fix parent_id after we may have used tmp ids; map via containment again
    # using element_id of largest containing box.
    by_id = {e.element_id: e for e in out}
    fixed: list[UIElement] = []
    for e in out:
        parent_id = None
        carea = max(1, e.bbox.width * e.bbox.height)
        best_area = None
        for cand in out:
            if cand.element_id == e.element_id:
                continue
            parea = cand.bbox.width * cand.bbox.height
            if parea < carea * 1.15:
                continue
            cx, cy = e.bbox.center()
            if cand.bbox.contains(cx, cy) and (best_area is None or parea < best_area):
                best_area = parea
                parent_id = cand.element_id
        depth = 0
        walk = parent_id
        seen: set[str] = set()
        while walk and walk in by_id and walk not in seen:
            seen.add(walk)
            depth += 1
            walk = by_id[walk].parent_id
            # parent_id not yet updated on by_id — approximate via best only one level for depth
            break
        # Compute depth properly after all parent_ids set
        fixed.append(e.model_copy(update={"parent_id": parent_id}))

    by_id = {e.element_id: e for e in fixed}

    def depth_of(eid: str, stack: set[str] | None = None) -> int:
        stack = stack or set()
        if eid in stack:
            return 0
        el = by_id.get(eid)
        if el is None or not el.parent_id:
            return 0
        stack.add(eid)
        return 1 + depth_of(el.parent_id, stack)

    return [e.model_copy(update={"depth": depth_of(e.element_id)}) for e in fixed]


def filter_roi(elements: list[UIElement], roi: BBox | None) -> list[UIElement]:
    if roi is None:
        return elements
    out: list[UIElement] = []
    for e in elements:
        cx, cy = e.bbox.center()
        if roi.contains(cx, cy):
            out.append(e)
        else:
            # keep if significant overlap
            if bbox_iou(e.bbox, roi) > 0.05 or e.bbox.clamp(roi).width > 0:
                inter = e.bbox.clamp(roi)
                if inter.width * inter.height > 0:
                    out.append(e)
    return out
