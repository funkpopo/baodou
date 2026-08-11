"""Mock capture: synthetic frame metadata without real screenshot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import CaptureError, ErrorCode
from core.logging import get_logger, log_event
from core.models import CaptureMode, FrameKind, ScreenFrame

from capture.base import CaptureBackend

_log = get_logger("capture.mock")


@dataclass
class MockPacket:
    """Minimal packet shape compatible with TaskAgent / mss CapturePacket."""

    meta: ScreenFrame
    image: Any = None

    def attach_b64(self) -> None:
        return None

    def save(self, path: Any) -> None:
        return None

    def release(self) -> None:
        return None


class MockCapture(CaptureBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._fail_next = False
        self._seq = 0

    def fail_next(self) -> None:
        """Test hook: next capture raises CaptureError."""
        self._fail_next = True

    def capture(self, *, trace_id: str = "") -> ScreenFrame:
        return self.capture_packet(trace_id=trace_id).meta

    def capture_packet(
        self,
        *,
        kind: FrameKind = FrameKind.RAW,
        force: bool = True,
        encode: bool = False,
        trace_id: str = "",
    ) -> MockPacket:
        get_global_token().check()
        t0 = time.perf_counter()
        if self._fail_next:
            self._fail_next = False
            raise CaptureError("mock 截图失败", code=ErrorCode.CAPTURE_FAILED)

        # Simulate modest capture latency.
        time.sleep(0.005)
        elapsed = (time.perf_counter() - t0) * 1000
        w = self.config.capture.max_width
        h = self.config.capture.max_height
        self._seq += 1
        # Vary pixel_hash slightly so verify sees a "new" frame when forced.
        pixel_hash = f"mockhash-{self._seq:04d}" if force else "mockhash-stable"
        frame = ScreenFrame(
            trace_id=trace_id,
            mode=CaptureMode(self.config.capture.mode)
            if self.config.capture.mode in CaptureMode._value2member_map_
            else CaptureMode.PRIMARY,
            frame_kind=kind,
            monitor_index=getattr(self.config.capture, "monitor_index", 0) or 0,
            width=w,
            height=h,
            origin_x=0,
            origin_y=0,
            physical_width=w,
            physical_height=h,
            logical_width=float(w),
            logical_height=float(h),
            dpi_scale=1.0,
            scale_x=1.0,
            scale_y=1.0,
            image_format=self.config.capture.image_format,
            image_path=None,
            image_b64=None,
            capture_ms=elapsed,
            changed=True,
            change_score=0.05 if force else 0.0,
            pixel_hash=pixel_hash,
            extra={"backend": "mock", "synthetic": True, "seq": self._seq},
        )
        log_event(_log, "capture.frame", **frame.log_summary())
        return MockPacket(meta=frame, image=None)
