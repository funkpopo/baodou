"""Multi-source UI vision pipeline: UIA + OCR + rules → fuse → result."""

from __future__ import annotations

import contextlib
import time

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import ErrorCode, VisionError
from core.logging import get_logger, log_event
from core.models import BBox, ScreenFrame, UIElement, UIVisionResult
from PIL import Image

from ui_vision.base import UIRecognizer, UIVisionBackend
from ui_vision.context import filter_elements_for_goal
from ui_vision.coords import attach_dpi_fields
from ui_vision.fuse import filter_roi, fuse_elements
from ui_vision.ids import element_stale

_log = get_logger("ui_vision.pipeline")


class CompositeUIVision(UIVisionBackend):
    """Orchestrates plugin recognizers with fusion and ROI support."""

    def __init__(
        self,
        config: AppConfig,
        recognizers: list[UIRecognizer] | None = None,
    ) -> None:
        self.config = config
        self.recognizers = recognizers or []
        self._last_by_id: dict[str, UIElement] = {}
        self._last_frame_id: str = ""

    def recognize(
        self,
        frame: ScreenFrame,
        *,
        trace_id: str = "",
        image: Image.Image | None = None,
        roi: BBox | None = None,
        goal: str | None = None,
    ) -> UIVisionResult:
        get_global_token().check()
        t0 = time.perf_counter()
        timeout_ms = self.config.ui_vision.timeout_ms
        batches: list[list[UIElement]] = []
        sources_used: list[str] = []
        notes: list[str] = []

        if not self.recognizers:
            raise VisionError(
                "没有可用的 UI 识别器",
                code=ErrorCode.VISION_FAILED,
                details={"sources": self.config.ui_vision.sources},
            )

        for rec in self.recognizers:
            if (time.perf_counter() - t0) * 1000 > timeout_ms:
                notes.append("timeout_partial")
                break
            get_global_token().check()
            try:
                found = rec.recognize(frame, image, roi=roi, trace_id=trace_id)
                # Normalize DPI fields
                found = [attach_dpi_fields(e, frame) for e in found]
                if roi is not None:
                    found = filter_roi(found, roi)
                batches.append(found)
                sources_used.append(rec.name)
                notes.append(f"{rec.name}:{len(found)}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{rec.name}:error")
                _log.warning(
                    "recognizer_failed",
                    extra={
                        "event": "recognizer_failed",
                        "source": rec.name,
                        "error": str(exc),
                        "trace_id": trace_id,
                    },
                )

        fused = fuse_elements(
            batches,
            iou_threshold=self.config.ui_vision.fuse_iou_threshold,
            confidence_threshold=self.config.ui_vision.confidence_threshold,
            max_elements=self.config.ui_vision.max_elements * 2,
        )

        # Goal-aware soft ranking without dropping structure entirely
        if goal:
            ranked = filter_elements_for_goal(
                fused, goal, max_elements=self.config.ui_vision.max_elements
            )
            # Keep high-confidence UIA windows even if not goal-ranked
            extra = [
                e
                for e in fused
                if e.element_id not in {x.element_id for x in ranked}
                and e.type.value in ("window", "dialog")
            ][:4]
            fused = (ranked + extra)[: self.config.ui_vision.max_elements]
        else:
            fused = fused[: self.config.ui_vision.max_elements]

        review_count = sum(1 for e in fused if e.needs_review or e.conflict)
        latency = (time.perf_counter() - t0) * 1000
        if latency > timeout_ms and not fused:
            raise VisionError(
                f"UI 识别超时 ({latency:.0f}ms)",
                code=ErrorCode.VISION_TIMEOUT,
                details={"timeout_ms": timeout_ms, "sources": sources_used},
            )

        result = UIVisionResult(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            elements=fused,
            latency_ms=latency,
            source="+".join(sources_used) if sources_used else "none",
            sources_used=sources_used,
            notes="; ".join(notes),
            roi=roi,
            dpi_scale=frame.dpi_scale or 1.0,
            review_count=review_count,
        )

        # Staleness bookkeeping vs previous frame
        self._last_by_id = {e.element_id: e for e in fused}
        self._last_frame_id = frame.frame_id

        log_event(
            _log,
            "vision.result",
            **result.log_summary(),
            goal=bool(goal),
            roi=bool(roi),
        )
        return result

    def is_stale(self, element: UIElement, current: UIVisionResult | None = None) -> bool:
        """Whether a previously chosen element is no longer safe to click."""
        mapping = (
            {e.element_id: e for e in current.elements} if current is not None else self._last_by_id
        )
        fid = current.frame_id if current is not None else self._last_frame_id
        return element_stale(element, mapping, frame_id=fid)

    def close(self) -> None:
        for rec in self.recognizers:
            with contextlib.suppress(Exception):
                rec.close()


def load_image_for_frame(frame: ScreenFrame) -> Image.Image | None:
    """Best-effort load PIL image from frame path/b64 for OCR/rules."""
    if frame.image_path:
        try:
            return Image.open(frame.image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    if frame.image_b64:
        import base64
        import io

        try:
            raw = base64.b64decode(frame.image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    return None
