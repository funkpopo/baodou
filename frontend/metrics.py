"""System + pipeline metrics for the observability panel (Phase H)."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.models import MetricsSnapshot

_log_name = "frontend.metrics"


@dataclass
class MetricsCollector:
    """Thread-safe latency / resource collector.

    Call ``record_*`` from pipeline hops; ``snapshot()`` for the UI panel.
    """

    recent_errors_max: int = 12
    capture_latency_ms: float | None = None
    vision_latency_ms: float | None = None
    model_latency_ms: float | None = None
    end_to_end_ms: float | None = None
    queue_length: int = 0
    queue_dropped: int = 0
    _errors: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _proc: Any = field(default=None, repr=False, init=False)
    _psutil: Any = field(default=None, repr=False, init=False)
    _gpu_name: str = ""
    _gpu_util: float | None = None
    _gpu_mem: float | None = None
    _last_cpu: float | None = None

    def __post_init__(self) -> None:
        self._errors = deque(maxlen=max(1, self.recent_errors_max))
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            # Prime CPU percent (first call returns 0.0)
            self._proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            self._psutil = None
            self._proc = None
        self._detect_gpu_name()

    def _detect_gpu_name(self) -> None:
        """Best-effort GPU label (Intel Arc / SYCL device from config defaults)."""
        # Prefer env hint used by llama-server; fall back to generic label.
        name = os.environ.get("BAODOU_GPU_NAME") or os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
        if name:
            self._gpu_name = str(name)[:80]
            return
        # Windows: try wmic once (non-fatal)
        try:
            import subprocess

            r = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
            names = [ln for ln in lines if ln.lower() != "name"]
            if names:
                # Prefer Intel Arc if present
                arc = next((n for n in names if "arc" in n.lower() or "intel" in n.lower()), None)
                self._gpu_name = (arc or names[0])[:80]
        except Exception:  # noqa: BLE001
            self._gpu_name = ""

    def record_capture(self, ms: float | None) -> None:
        with self._lock:
            self.capture_latency_ms = float(ms) if ms is not None else None

    def record_vision(self, ms: float | None) -> None:
        with self._lock:
            self.vision_latency_ms = float(ms) if ms is not None else None

    def record_model(self, ms: float | None) -> None:
        with self._lock:
            self.model_latency_ms = float(ms) if ms is not None else None

    def record_e2e(self, ms: float | None) -> None:
        with self._lock:
            self.end_to_end_ms = float(ms) if ms is not None else None

    def record_queue(self, length: int, dropped: int = 0) -> None:
        with self._lock:
            self.queue_length = max(0, int(length))
            self.queue_dropped = max(0, int(dropped))

    def push_error(self, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        with self._lock:
            self._errors.appendleft(msg[:200])

    def clear_errors(self) -> None:
        with self._lock:
            self._errors.clear()

    def _resource_sample(self) -> tuple[float | None, float | None, float | None]:
        cpu = mem_rss = mem_pct = None
        if self._psutil is None:
            return cpu, mem_rss, mem_pct
        try:
            cpu = float(self._psutil.cpu_percent(interval=None))
            if self._proc is not None:
                # Process CPU can exceed 100 on multi-core; clamp for display
                p_cpu = float(self._proc.cpu_percent(interval=None))
                self._last_cpu = p_cpu
                info = self._proc.memory_info()
                mem_rss = float(info.rss) / (1024 * 1024)
                mem_pct = float(self._proc.memory_percent())
            else:
                vm = self._psutil.virtual_memory()
                mem_rss = float(vm.used) / (1024 * 1024)
                mem_pct = float(vm.percent)
        except Exception:  # noqa: BLE001
            pass
        return cpu, mem_rss, mem_pct

    def snapshot(self) -> MetricsSnapshot:
        cpu, mem_rss, mem_pct = self._resource_sample()
        with self._lock:
            return MetricsSnapshot(
                capture_latency_ms=self.capture_latency_ms,
                vision_latency_ms=self.vision_latency_ms,
                model_latency_ms=self.model_latency_ms,
                end_to_end_ms=self.end_to_end_ms,
                queue_length=self.queue_length,
                queue_dropped=self.queue_dropped,
                cpu_percent=cpu,
                memory_rss_mb=mem_rss,
                memory_percent=mem_pct,
                gpu_name=self._gpu_name,
                gpu_util_percent=self._gpu_util,
                gpu_mem_mb=self._gpu_mem,
                recent_errors=list(self._errors),
            )

    def from_task_result(self, result: Any) -> None:
        """Pull latencies from a TaskRunResult / pipeline payloads."""
        try:
            e2e = getattr(result, "elapsed_ms", None)
            self.record_e2e(e2e)
            vision = getattr(result, "vision", None)
            if vision is not None and getattr(vision, "latency_ms", None) is not None:
                self.record_vision(vision.latency_ms)
            obs = getattr(result, "observation", None)
            if obs is not None and getattr(obs, "latency_ms", None) is not None:
                self.record_model(obs.latency_ms)
            # Scan events for finer latency
            for evt in getattr(result, "events", []) or []:
                kind = getattr(evt, "kind", None)
                payload = getattr(evt, "payload", None) or {}
                if not isinstance(payload, dict):
                    continue
                k = kind.value if hasattr(kind, "value") else str(kind)
                if k == "frame" and payload.get("capture_ms") is not None:
                    self.record_capture(payload.get("capture_ms"))
                if k == "vision" and payload.get("latency_ms") is not None:
                    self.record_vision(payload.get("latency_ms"))
                if k == "inference" and payload.get("latency_ms") is not None:
                    self.record_model(payload.get("latency_ms"))
            err = getattr(result, "error", None)
            if err is not None:
                msg = getattr(err, "message", None) or str(err)
                self.push_error(msg)
        except Exception:  # noqa: BLE001
            pass


class TimedPhase:
    """Context manager that records elapsed ms into a MetricsCollector field."""

    def __init__(self, collector: MetricsCollector, field_name: str) -> None:
        self.collector = collector
        self.field_name = field_name
        self._t0 = 0.0

    def __enter__(self) -> TimedPhase:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        ms = (time.perf_counter() - self._t0) * 1000
        fn = getattr(self.collector, f"record_{self.field_name}", None)
        if callable(fn):
            fn(ms)
