"""Phase C capture benchmark: FPS, queue bounds, multi-monitor, coordinate check."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from capture.geometry import cursor_pos_physical, list_monitors, virtual_desktop
from capture.mss_backend import MssCapture
from capture.pipeline import CapturePipeline
from core.config import load_config
from core.logging import setup_logging
from core.models import FrameKind

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg.capture.backend = "mss"
    cfg.capture.save_frames = True
    cfg.capture.save_dir = str(ART.relative_to(ROOT)) if ART.is_relative_to(ROOT) else str(ART)
    cfg.capture.privacy.enabled = True
    setup_logging(
        level="INFO", json_logs=True, log_dir=cfg.app.log_dir, project_root=cfg.project_root
    )

    report: dict = {
        "monitors": [m.model_dump(mode="json") for m in list_monitors()],
        "virtual_desktop": virtual_desktop().model_dump(mode="json"),
        "cursor": cursor_pos_physical().model_dump(),
    }

    # One-shot modes
    shots = {}
    for mode in ("primary", "all"):
        cfg.capture.mode = mode
        cap = MssCapture(cfg)
        try:
            t0 = time.perf_counter()
            packet = cap.capture_packet(kind=FrameKind.VISION, force=True, encode=True)
            dt = (time.perf_counter() - t0) * 1000
            path = ART / f"shot_{mode}.png"
            packet.save(path)
            # coord check: image center
            center = packet.meta.image_to_screen(packet.meta.width / 2, packet.meta.height / 2)
            shots[mode] = {
                **packet.meta.log_summary(),
                "wall_ms": dt,
                "center_screen": center.model_dump(),
                "path": str(path),
            }
        finally:
            cap.close()
    report["oneshot"] = shots

    # Stream for ~2s with small queue — ensure no unbounded growth
    cfg.capture.mode = "primary"
    cfg.capture.queue_size = 3
    cfg.capture.streams["preview"].fps = 12
    cfg.capture.streams["vision"].fps = 8
    cfg.capture.streams["vision"].only_on_change = False
    pipe = CapturePipeline(cfg)
    consumed = 0
    t0 = time.monotonic()
    pipe.start()
    deadline = t0 + 2.0
    while time.monotonic() < deadline:
        p = pipe.latest(FrameKind.PREVIEW, timeout=0.05)
        if p:
            consumed += 1
            p.release()
        v = pipe.latest(FrameKind.VISION, timeout=0.0)
        if v:
            consumed += 1
            v.release()
    # also request model once
    pipe.request_model_frame(user_request=True)
    time.sleep(0.3)
    model = pipe.latest(FrameKind.MODEL, timeout=0.5)
    stats = pipe.stats_dict()
    pipe.close()

    report["stream"] = {
        "duration_sec": 2.0,
        "consumed": consumed,
        "model_got": model is not None,
        "stats": stats,
        "queues_bounded": all(
            stats["streams"][k]["high_watermark"] <= cfg.capture.queue_size
            for k in ("preview", "vision", "model", "verify")
        ),
    }

    out = RESULTS / "phase_c_capture.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(out),
                "queues_bounded": report["stream"]["queues_bounded"],
                "monitors": len(report["monitors"]),
                "consumed": consumed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["stream"]["queues_bounded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
