"""Real screen capture via mss + optional window/region modes."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import mss
from core.cancel import get_global_token
from core.config import AppConfig, StreamProfile
from core.errors import CaptureError, ErrorCode
from core.logging import get_logger, log_event
from core.models import (
    BBox,
    CaptureMode,
    FrameKind,
    ScreenFrame,
    SensitiveRegion,
    WindowInfo,
)
from PIL import Image

from capture.base import CaptureBackend
from capture.change import ChangeDetector
from capture.frame import FramePacket
from capture.geometry import (
    ensure_dpi_awareness,
    list_monitors,
    resolve_capture_region,
    virtual_desktop,
)
from capture.preprocess import preprocess
from capture.privacy import _mask_color, apply_masks, collect_mask_regions
from capture.window_win import find_window

_log = get_logger("capture.mss")


def _grab_pil(sct: mss.mss, region: dict[str, int]) -> Image.Image:
    shot = sct.grab(region)
    # mss returns BGRA
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


class MssCapture(CaptureBackend):
    def __init__(self, config: AppConfig) -> None:
        ensure_dpi_awareness()
        self.config = config
        self._sct = mss.mss()
        self._detector = ChangeDetector(
            sample_size=config.capture.change_sample_size,
            threshold=config.capture.change_threshold,
        )
        self._last_window: WindowInfo | None = None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._sct.close()

    def list_monitors(self):
        return list_monitors()

    def reset_change_detector(self) -> None:
        self._detector.reset()

    def _resolve_window(self) -> WindowInfo | None:
        cfg = self.config.capture
        if cfg.mode != "window":
            return None
        win = find_window(title_substr=cfg.window_title, hwnd=cfg.window_hwnd)
        self._last_window = win
        return win

    def _region_and_mode(self) -> tuple[object, CaptureMode, WindowInfo | None]:
        cfg = self.config.capture
        mode = (
            CaptureMode(cfg.mode)
            if cfg.mode in CaptureMode._value2member_map_
            else CaptureMode.PRIMARY
        )
        window: WindowInfo | None = None
        window_bbox: BBox | None = None
        if mode == CaptureMode.WINDOW:
            window = self._resolve_window()
            assert window is not None
            window_bbox = window.bbox
        region = resolve_capture_region(
            mode=cfg.mode,
            monitor_index=cfg.monitor_index,
            region=cfg.region,
            window_bbox=window_bbox,
        )
        return region, mode, window

    def capture(self, *, trace_id: str = "") -> ScreenFrame:
        """CaptureBackend API — returns metadata; pixels only if include_b64/save."""
        packet = self.capture_packet(trace_id=trace_id, kind=FrameKind.RAW, force=True)
        if self.config.capture.include_b64 and packet.image is not None:
            packet.attach_b64()
        # Release heavy pixels from long-lived ScreenFrame path unless saved
        meta = packet.meta
        packet.release()
        return meta

    def capture_packet(
        self,
        *,
        trace_id: str = "",
        kind: FrameKind = FrameKind.RAW,
        force: bool = False,
        profile: StreamProfile | None = None,
        roi: BBox | None = None,
        extra_masks: list[SensitiveRegion] | None = None,
        encode: bool = False,
        save_path: Path | None = None,
    ) -> FramePacket:
        get_global_token().check()
        t0 = time.perf_counter()
        cfg = self.config.capture
        profile = profile or self._profile_for(kind)

        try:
            region, mode, window = self._region_and_mode()
        except CaptureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"解析采集区域失败: {exc}", cause=exc) from exc

        try:
            raw = _grab_pil(self._sct, region.as_mss_dict())
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(
                f"截图失败: {exc}",
                code=ErrorCode.CAPTURE_FAILED,
                cause=exc,
                details=region.as_mss_dict(),
            ) from exc

        capture_ms = (time.perf_counter() - t0) * 1000
        phys_w, phys_h = raw.size

        # Change detection on full physical RGB before scale (downsamples internally)
        if force:
            changed, score, pixel_hash = True, *self._detector.peek(raw)[1:]
            # still update baseline
            self._detector.update(raw)
            changed = True
        else:
            changed, score, pixel_hash = self._detector.update(raw)

        pp = preprocess(
            raw,
            max_width=profile.max_width,
            max_height=profile.max_height,
            color_mode=cfg.color_mode,
            image_format=profile.image_format,
            jpeg_quality=profile.jpeg_quality,
            roi=roi,
            encode=False,
            frame_kind=kind,
        )
        # Combined scale: physical → (optional roi) → stored
        # preprocess scale is relative to post-ROI image.
        scale_x = pp.scale_x
        scale_y = pp.scale_y
        origin_x = region.left
        origin_y = region.top
        if pp.roi_applied is not None:
            origin_x += pp.roi_applied.x
            origin_y += pp.roi_applied.y

        masks = collect_mask_regions(
            cfg.privacy,
            extra=extra_masks,
            scan_privacy_windows=cfg.privacy.enabled,
        )
        img, applied = apply_masks(
            pp.image,
            regions=masks,
            origin_x=origin_x,
            origin_y=origin_y,
            scale_x=scale_x,
            scale_y=scale_y,
            mask_color=_mask_color(cfg.privacy),
        )

        logical_w = phys_w / region.dpi_scale if region.dpi_scale else float(phys_w)
        logical_h = phys_h / region.dpi_scale if region.dpi_scale else float(phys_h)

        meta = ScreenFrame(
            trace_id=trace_id,
            mode=mode,
            frame_kind=kind,
            monitor_index=region.mon_index,
            width=pp.width,
            height=pp.height,
            origin_x=origin_x,
            origin_y=origin_y,
            physical_width=phys_w
            if pp.roi_applied is None
            else (pp.roi_applied.width if pp.roi_applied else phys_w),
            physical_height=phys_h
            if pp.roi_applied is None
            else (pp.roi_applied.height if pp.roi_applied else phys_h),
            logical_width=logical_w,
            logical_height=logical_h,
            dpi_scale=region.dpi_scale,
            dpi_x=region.dpi_x,
            dpi_y=region.dpi_y,
            orientation=region.orientation,
            scale_x=scale_x,
            scale_y=scale_y,
            image_format=profile.image_format,
            color_mode=cfg.color_mode,
            capture_ms=capture_ms,
            preprocess_ms=pp.preprocess_ms,
            changed=changed,
            change_score=score,
            pixel_hash=pixel_hash,
            window=window,
            monitor=region.to_monitor_info(),
            masked_regions=applied,
            extra={
                "backend": "mss",
                "region": region.as_mss_dict(),
                "force": force,
                "virtual_desktop": virtual_desktop().model_dump(),
            },
        )

        packet = FramePacket(meta=meta, image=img, kind=kind)
        if encode or cfg.include_b64:
            packet.ensure_encoded(fmt=profile.image_format, jpeg_quality=profile.jpeg_quality)
            if cfg.include_b64:
                packet.attach_b64()

        if save_path is not None:
            packet.save(save_path)
        elif cfg.save_frames:
            root = self.config.project_root / cfg.save_dir
            ext = "jpg" if profile.image_format.lower() in ("jpg", "jpeg") else "png"
            packet.save(root / f"{meta.frame_id}_{kind.value}.{ext}")

        log_event(_log, "capture.frame", **meta.log_summary())
        # Free raw full-res early
        del raw
        return packet

    def _profile_for(self, kind: FrameKind) -> StreamProfile:
        streams = self.config.capture.streams or {}
        key = kind.value
        if key in streams:
            return streams[key]
        # fallback to capture defaults
        return StreamProfile(
            fps=self.config.capture.target_fps,
            max_width=self.config.capture.max_width,
            max_height=self.config.capture.max_height,
            image_format=self.config.capture.image_format,
            jpeg_quality=self.config.capture.jpeg_quality,
        )
