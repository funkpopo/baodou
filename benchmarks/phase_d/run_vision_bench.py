"""Phase D UI vision benchmark: multi-source recognize + multi-res coord check."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from capture.geometry import list_monitors
from capture.mss_backend import MssCapture
from core.config import load_config
from core.logging import setup_logging
from core.models import FrameKind
from ui_vision.context import annotate_from_frame, serialize_for_model, serialize_text_summary
from ui_vision.factory import create_ui_vision

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"


def _coord_audit(frame, elements: list) -> dict:
    ok = 0
    bad = 0
    samples = []
    for el in elements:
        c = el.center
        ip = frame.screen_to_image(c.x, c.y)
        inside = -2 <= ip.x < frame.width + 2 and -2 <= ip.y < frame.height + 2
        if inside:
            ok += 1
        else:
            bad += 1
        if len(samples) < 15:
            samples.append(
                {
                    "element_id": el.element_id,
                    "type": el.type.value,
                    "text": (el.text or "")[:40],
                    "center": c.model_dump(),
                    "image_xy": ip.model_dump(),
                    "dpi_scale": el.dpi_scale,
                    "source": el.source,
                    "inside_image": inside,
                }
            )
    return {
        "centers_inside": ok,
        "centers_outside": bad,
        "coord_ok": bad == 0 or ok >= bad * 3,
        "samples": samples,
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg.capture.backend = "mss"
    cfg.capture.mode = "primary"
    setup_logging(
        level="INFO", json_logs=True, log_dir=cfg.app.log_dir, project_root=cfg.project_root
    )

    report: dict = {
        "monitors": [m.model_dump(mode="json") for m in list_monitors()],
        "ui_vision_backend": cfg.ui_vision.backend,
        "sources": cfg.ui_vision.sources,
    }

    # Synthetic multi-res mock pass (always)
    from core.models import ScreenFrame
    from ui_vision.mock import MockUIVision

    mock = MockUIVision(cfg)
    multi_res = {}
    for name, kwargs in {
        "1080p": dict(
            width=1280,
            height=720,
            physical_width=1920,
            physical_height=1080,
            scale_x=1.5,
            scale_y=1.5,
            dpi_scale=1.0,
        ),
        "150pct_scaled": dict(
            width=1280,
            height=720,
            physical_width=1920,
            physical_height=1080,
            scale_x=1.5,
            scale_y=1.5,
            dpi_scale=1.5,
            dpi_x=144,
            dpi_y=144,
        ),
        "2k": dict(
            width=1280,
            height=800,
            physical_width=2560,
            physical_height=1600,
            scale_x=2.0,
            scale_y=2.0,
            dpi_scale=1.25,
            dpi_x=120,
            dpi_y=120,
        ),
    }.items():
        frame = ScreenFrame(origin_x=0, origin_y=0, **kwargs)
        res = mock.recognize(frame)
        multi_res[name] = {
            **_coord_audit(frame, res.elements),
            "element_count": len(res.elements),
            "btn_search": res.by_id("btn_search_01").bbox.model_dump()
            if res.by_id("btn_search_01")
            else None,
        }
    report["multi_res_mock"] = multi_res

    # Live desktop recognition
    cap = MssCapture(cfg)
    vision = create_ui_vision(cfg)
    try:
        t0 = time.perf_counter()
        packet = cap.capture_packet(kind=FrameKind.VISION, force=True, encode=True)
        result = vision.recognize(
            packet.meta,
            image=packet.image,
            goal="点击按钮",
        )
        dt = (time.perf_counter() - t0) * 1000
        audit = _coord_audit(packet.meta, result.elements)
        compact = serialize_for_model(result, goal="点击按钮", max_elements=24)
        ann_path = ART / "annotate_primary.png"
        if packet.image is not None:
            ann = annotate_from_frame(
                packet.image, result, packet.meta.screen_to_image, goal="点击按钮"
            )
            ann.save(ann_path)
            packet.save(ART / "vision_frame.png")

        report["live"] = {
            "latency_ms": round(dt, 2),
            "vision": result.log_summary(),
            "frame": packet.meta.log_summary(),
            "coord_audit": audit,
            "compact_count": len(compact),
            "compact_preview": compact[:12],
            "text_summary": serialize_text_summary(result, goal="点击按钮", max_elements=12),
            "annotate": str(ann_path),
            "type_histogram": _type_hist(result.elements),
            "source_histogram": _source_hist(result.elements),
        }
    finally:
        vision.close()
        cap.close()

    mock_ok = all(v.get("coord_ok") for v in multi_res.values())
    live_ok = bool(report.get("live", {}).get("coord_audit", {}).get("coord_ok", False))
    report["ok"] = mock_ok and live_ok
    report["acceptance"] = {
        "multi_res_coord_ok": mock_ok,
        "live_coord_ok": live_ok,
        "live_element_count": report.get("live", {}).get("vision", {}).get("element_count", 0),
    }

    out = RESULTS / "phase_d_vision.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0 if report["ok"] else 1


def _type_hist(elements) -> dict[str, int]:
    h: dict[str, int] = {}
    for e in elements:
        k = e.type.value
        h[k] = h.get(k, 0) + 1
    return h


def _source_hist(elements) -> dict[str, int]:
    h: dict[str, int] = {}
    for e in elements:
        for s in e.source:
            h[s] = h.get(s, 0) + 1
    return h


if __name__ == "__main__":
    raise SystemExit(main())
