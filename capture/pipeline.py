"""Realtime multi-stream capture pipeline with bounded queues and change gates."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from core.cancel import CancellationToken, get_global_token
from core.config import AppConfig, StreamProfile
from core.logging import get_logger, log_event, new_trace_id, trace_scope
from core.models import FrameKind, SensitiveRegion

from capture.factory import create_capture
from capture.frame import FramePacket
from capture.mss_backend import MssCapture
from capture.queue import FrameQueue

_log = get_logger("capture.pipeline")

OnFrame = Callable[[FramePacket], None]


@dataclass
class StreamRuntime:
    kind: FrameKind
    profile: StreamProfile
    queue: FrameQueue[FramePacket]
    interval: float
    last_emit: float = 0.0
    emitted: int = 0
    skipped_unchanged: int = 0
    skipped_rate: int = 0


@dataclass
class PipelineStats:
    ticks: int = 0
    captures: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.monotonic)
    streams: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "captures": self.captures,
            "errors": self.errors,
            "uptime_sec": time.monotonic() - self.started_at,
            "streams": self.streams,
        }


class CapturePipeline:
    """
    Producer loop that grabs the screen and fans out into per-kind bounded queues.

    - preview / vision: rate-limited by stream fps
    - model / verify: event-driven via request_model_frame / request_verify_frame
    - only_on_change: skip enqueue when detector says unchanged (unless force)
    """

    def __init__(
        self,
        config: AppConfig,
        token: CancellationToken | None = None,
        *,
        backend: MssCapture | None = None,
    ) -> None:
        self.config = config
        self.token = token or get_global_token()
        self.backend = backend or create_capture(config)
        if not isinstance(self.backend, MssCapture):
            # Wrap mock by adapting capture() only path — still allow pipeline tests with mock
            self._use_packet_api = hasattr(self.backend, "capture_packet")
        else:
            self._use_packet_api = True

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._model_requested = False
        self._verify_requested = False
        self._user_request = False
        self._extra_masks: list[SensitiveRegion] = []
        self.stats = PipelineStats()

        qsize = config.capture.queue_size
        policy = config.capture.drop_policy
        self.streams: dict[FrameKind, StreamRuntime] = {}
        for kind in (FrameKind.PREVIEW, FrameKind.VISION, FrameKind.MODEL, FrameKind.VERIFY):
            profile = self._profile(kind)
            interval = (1.0 / profile.fps) if profile.fps and profile.fps > 0 else 0.0
            self.streams[kind] = StreamRuntime(
                kind=kind,
                profile=profile,
                queue=FrameQueue(maxsize=qsize, drop_policy=policy, name=kind.value),
                interval=interval,
            )

    def _profile(self, kind: FrameKind) -> StreamProfile:
        streams = self.config.capture.streams or {}
        if kind.value in streams:
            return streams[kind.value]
        from core.config import StreamProfile as SP

        return SP(
            fps=self.config.capture.target_fps
            if kind in (FrameKind.PREVIEW, FrameKind.VISION)
            else 0.0,
            max_width=self.config.capture.max_width,
            max_height=self.config.capture.max_height,
            image_format=self.config.capture.image_format,
            jpeg_quality=self.config.capture.jpeg_quality,
            only_on_change=kind in (FrameKind.VISION, FrameKind.MODEL),
        )

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="capture-pipeline", daemon=True)
            self._thread.start()
            log_event(_log, "capture.pipeline_start", queue_size=self.config.capture.queue_size)

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        log_event(_log, "capture.pipeline_stop", **self.stats_dict())

    def close(self) -> None:
        self.stop()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> CapturePipeline:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- event triggers ---

    def request_model_frame(self, *, user_request: bool = True) -> None:
        with self._lock:
            self._model_requested = True
            if user_request:
                self._user_request = True

    def request_verify_frame(self) -> None:
        with self._lock:
            self._verify_requested = True

    def set_extra_masks(self, masks: list[SensitiveRegion]) -> None:
        with self._lock:
            self._extra_masks = list(masks)

    def get_queue(self, kind: FrameKind) -> FrameQueue[FramePacket]:
        return self.streams[kind].queue

    def latest(self, kind: FrameKind, *, timeout: float | None = 0.0) -> FramePacket | None:
        return self.streams[kind].queue.get(timeout=timeout)

    def stats_dict(self) -> dict:
        self.stats.streams = {
            k.value: {
                **s.queue.snapshot_stats(),
                "emitted": s.emitted,
                "skipped_unchanged": s.skipped_unchanged,
                "skipped_rate": s.skipped_rate,
                "fps_cfg": s.profile.fps,
            }
            for k, s in self.streams.items()
        }
        return self.stats.to_dict()

    def _loop(self) -> None:
        # Base tick: max of active stream rates or target_fps
        rates = [s.profile.fps for s in self.streams.values() if s.profile.fps > 0]
        base_fps = max(rates) if rates else self.config.capture.target_fps
        base_fps = max(base_fps, 1.0)
        tick = 1.0 / base_fps

        while True:
            with self._lock:
                if not self._running:
                    break
            if self.token.is_cancelled:
                break
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                self.stats.errors += 1
                _log.exception("capture.pipeline_tick_error")
            self.stats.ticks += 1
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, tick - elapsed)
            if sleep_for > 0:
                self.token.wait(sleep_for)

    def _tick(self) -> None:
        with self._lock:
            model_req = self._model_requested
            verify_req = self._verify_requested
            user_req = self._user_request
            masks = list(self._extra_masks)
            self._model_requested = False
            self._verify_requested = False
            self._user_request = False

        now = time.monotonic()
        want: list[tuple[FrameKind, bool]] = []  # kind, force

        for kind, rt in self.streams.items():
            if kind == FrameKind.MODEL:
                if model_req:
                    force = user_req and self.config.capture.force_on_user_request
                    want.append((kind, force or not rt.profile.only_on_change))
                continue
            if kind == FrameKind.VERIFY:
                if verify_req:
                    force = self.config.capture.force_on_verify
                    want.append((kind, force))
                continue
            # rate-limited streams
            if rt.interval <= 0:
                continue
            if now - rt.last_emit < rt.interval:
                rt.skipped_rate += 1
                continue
            want.append((kind, not rt.profile.only_on_change))

        if not want:
            return

        # One physical grab, fan-out with per-kind preprocess would re-grab;
        # for simplicity & coordinate accuracy we grab once per kind group with
        # shared force if any force requested.
        any_force = any(f for _, f in want)
        # Grab a high-quality base (vision profile) then re-encode per kind if needed.
        # Simpler approach: capture once per wanted kind (still bounded by fps).
        # To reduce cost: single grab with max resolution among wanted, then scale down.
        trace_id = new_trace_id()
        with trace_scope(trace_id):
            # Determine max dimensions among wanted profiles
            max_w = max(self.streams[k].profile.max_width for k, _ in want)
            max_h = max(self.streams[k].profile.max_height for k, _ in want)
            base_kind = (
                FrameKind.VISION if any(k == FrameKind.VISION for k, _ in want) else want[0][0]
            )
            from core.config import StreamProfile

            base_profile = StreamProfile(
                fps=0,
                max_width=max_w,
                max_height=max_h,
                image_format="png",
                jpeg_quality=95,
                only_on_change=False,
            )

            if self._use_packet_api:
                base = self.backend.capture_packet(  # type: ignore[attr-defined]
                    trace_id=trace_id,
                    kind=base_kind,
                    force=any_force,
                    profile=base_profile,
                    extra_masks=masks,
                    encode=False,
                )
            else:
                # Mock path: synthesize empty packet from ScreenFrame
                from PIL import Image

                from capture.frame import FramePacket

                meta = self.backend.capture(trace_id=trace_id)
                meta.frame_kind = base_kind
                img = Image.new("RGB", (meta.width, meta.height), (32, 32, 40))
                base = FramePacket(meta=meta, image=img, kind=base_kind)

            self.stats.captures += 1
            changed = bool(base.meta.changed)

            for kind, force in want:
                rt = self.streams[kind]
                if rt.profile.only_on_change and not force and not changed:
                    rt.skipped_unchanged += 1
                    continue

                packet = self._derive_packet(base, kind, rt.profile)
                accepted = rt.queue.put(packet)
                if accepted:
                    rt.emitted += 1
                    rt.last_emit = now
                    log_event(
                        _log,
                        "capture.stream_emit",
                        kind=kind.value,
                        frame_id=packet.frame_id,
                        queue_size=len(rt.queue),
                        changed=packet.meta.changed,
                    )
                else:
                    packet.release()

            # Base may be unused if all derived; release base image if no longer needed
            base.release()

    def _derive_packet(
        self,
        base: FramePacket,
        kind: FrameKind,
        profile: StreamProfile,
    ) -> FramePacket:
        from capture.preprocess import preprocess

        if base.image is None:
            meta = base.meta.model_copy(deep=True)
            meta.frame_kind = kind
            return FramePacket(meta=meta, image=None, kind=kind)

        # If profile matches base size roughly, reuse
        if (
            base.meta.width <= profile.max_width
            and base.meta.height <= profile.max_height
            and profile.image_format == base.meta.image_format
        ):
            meta = base.meta.model_copy(deep=True)
            meta.frame_kind = kind
            # Shallow copy image — consumers should not mutate; release is independent
            img = base.image.copy()
            return FramePacket(meta=meta, image=img, kind=kind)

        pp = preprocess(
            base.image,
            max_width=profile.max_width,
            max_height=profile.max_height,
            color_mode=self.config.capture.color_mode,
            image_format=profile.image_format,
            jpeg_quality=profile.jpeg_quality,
            encode=False,
            frame_kind=kind,
        )
        meta = base.meta.model_copy(deep=True)
        meta.frame_kind = kind
        meta.width = pp.width
        meta.height = pp.height
        meta.scale_x = base.meta.scale_x * pp.scale_x
        meta.scale_y = base.meta.scale_y * pp.scale_y
        meta.image_format = profile.image_format
        meta.preprocess_ms = (meta.preprocess_ms or 0) + pp.preprocess_ms
        return FramePacket(meta=meta, image=pp.image, kind=kind)

    def capture_once(
        self,
        *,
        kind: FrameKind = FrameKind.VISION,
        force: bool = True,
        trace_id: str | None = None,
    ) -> FramePacket:
        """Synchronous one-shot capture (no background thread)."""
        tid = trace_id or new_trace_id()
        profile = self._profile(kind)
        if self._use_packet_api:
            return self.backend.capture_packet(  # type: ignore[attr-defined]
                trace_id=tid,
                kind=kind,
                force=force,
                profile=profile,
                extra_masks=list(self._extra_masks),
                encode=False,
            )
        from PIL import Image

        meta = self.backend.capture(trace_id=tid)
        meta.frame_kind = kind
        img = Image.new("RGB", (meta.width, meta.height), (32, 32, 40))
        return FramePacket(meta=meta, image=img, kind=kind)
