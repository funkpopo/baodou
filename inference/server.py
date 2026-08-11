"""llama-server process lifecycle: load, warmup, cancel/stop, release, recover."""

from __future__ import annotations

import atexit
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.cancel import get_global_token
from core.config import PROJECT_ROOT, AppConfig
from core.errors import ErrorCode, InferenceError
from core.logging import get_logger, log_event

_log = get_logger("inference.server")

# Module-level handle so atexit can stop a server we started.
_MANAGED: LlamaServerManager | None = None


def _http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[dict[str, Any] | None, str | None, float]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            dt = (time.perf_counter() - t0) * 1000
            return json.loads(raw) if raw else {}, None, dt
    except HTTPError as exc:
        dt = (time.perf_counter() - t0) * 1000
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:  # noqa: BLE001
            detail = str(exc)
        return None, f"HTTPError {exc.code}: {detail}", dt
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        dt = (time.perf_counter() - t0) * 1000
        return None, f"{type(exc).__name__}: {exc}", dt


class LlamaServerManager:
    """Manage D:\\llama\\llama-server with oneAPI + SYCL (no CPU fallback)."""

    def __init__(self, config: AppConfig, *, log_dir: Path | None = None) -> None:
        self.config = config
        self.base = config.inference.base_url
        self.log_dir = log_dir or (PROJECT_ROOT / "benchmarks" / "phase_e" / "artifacts")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[Any] | None = None
        self._started_by_us = False
        self._load_ms: float | None = None
        self._warmup_ms: float | None = None
        self._last_props: dict[str, Any] = {}
        self._atexit_registered = False

    # ------------------------------------------------------------------ health
    def health(self) -> bool:
        out, err, _ = _http_json(
            "GET",
            f"{self.base}/health",
            timeout=self.config.inference.health_timeout_sec,
        )
        if err or not isinstance(out, dict):
            return False
        # llama-server returns {"status":"ok"} when ready
        status = out.get("status")
        return status in (None, "ok") or out.get("error") in (None, False)

    def props(self) -> dict[str, Any]:
        out, err, _ = _http_json(
            "GET",
            f"{self.base}/props",
            timeout=self.config.inference.health_timeout_sec,
        )
        if err or not out:
            models, merr, _ = _http_json(
                "GET",
                f"{self.base}/v1/models",
                timeout=self.config.inference.health_timeout_sec,
            )
            return {"props_error": err, "models": models, "models_error": merr}
        self._last_props = out
        return out

    def is_running_managed(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ lifecycle
    def ensure_running(self, *, warmup: bool | None = None) -> dict[str, Any]:
        """Load if needed; optionally warmup. Idempotent."""
        get_global_token().check()
        if self.health():
            info = {
                "already_running": True,
                "started_by_us": self._started_by_us,
                "base": self.base,
                "props": self.props(),
            }
            do_warm = self.config.inference.warmup_on_start if warmup is None else warmup
            if do_warm and self._warmup_ms is None:
                info["warmup"] = self.warmup()
            return info

        if not self.config.inference.auto_start_server:
            raise InferenceError(
                f"llama-server 不可用: {self.base}",
                code=ErrorCode.MODEL_UNAVAILABLE,
                details={"base_url": self.base, "hint": "start server or set auto_start_server"},
            )
        return self.start(warmup=warmup)

    def start(self, *, warmup: bool | None = None) -> dict[str, Any]:
        """Start llama-server via oneAPI setvars + SYCL device."""
        global _MANAGED
        get_global_token().check()

        if self.health():
            return self.ensure_running(warmup=warmup)

        cfg = self.config
        llama_dir = cfg.paths.llama_dir
        exe = llama_dir / "llama-server.exe"
        model = cfg.paths.model_gguf
        mmproj = cfg.paths.mmproj_gguf
        setvars = cfg.paths.oneapi_setvars

        if not exe.exists():
            raise InferenceError(
                f"llama-server 不存在: {exe}",
                code=ErrorCode.MODEL_LOAD_FAILED,
                details={"path": str(exe)},
            )
        if not model.exists():
            raise InferenceError(
                f"模型文件不存在: {model}",
                code=ErrorCode.MODEL_LOAD_FAILED,
                details={"path": str(model)},
            )
        if not mmproj.exists():
            raise InferenceError(
                f"mmproj 不存在: {mmproj}",
                code=ErrorCode.MODEL_LOAD_FAILED,
                details={"path": str(mmproj)},
            )
        if not setvars.exists():
            raise InferenceError(
                f"oneAPI setvars 不存在: {setvars}",
                code=ErrorCode.MODEL_LOAD_FAILED,
                details={"path": str(setvars)},
            )

        # Refuse CPU-only device config
        device = (cfg.inference.device or "").strip()
        if not device or device.upper() in {"CPU", "NONE"}:
            raise InferenceError(
                f"禁止 CPU 默认推理: device={device!r}，请使用 SYCL0",
                code=ErrorCode.MODEL_LOAD_FAILED,
                details={"device": device},
            )

        n_threads = cfg.inference.n_threads
        if n_threads <= 0:
            import os

            n_threads = max(4, (os.cpu_count() or 8) - 2)

        bat = self.log_dir / "start_server.bat"
        out_log = self.log_dir / "server.out"
        err_log = self.log_dir / "server.err"

        # Build argv carefully; bat wraps oneAPI env.
        # --jinja: model chat template; --mmproj-offload: vision on GPU
        lines = [
            "@echo off",
            f'call "{setvars}" >nul 2>&1',
            f"set PATH={llama_dir};%PATH%",
            f'"{exe}" ^',
            f'  -m "{model}" ^',
            f'  --mmproj "{mmproj}" ^',
        ]
        if cfg.inference.mmproj_offload:
            lines.append("  --mmproj-offload ^")
        lines.extend(
            [
                f"  -ngl {cfg.inference.n_gpu_layers} ^",
                f"  -dev {device} ^",
                f"  -c {cfg.inference.n_ctx} ^",
                f"  -t {n_threads} ^",
                f"  -b {cfg.inference.n_batch} ^",
                f"  --host {cfg.inference.host} ^",
                f"  --port {cfg.inference.port} ^",
                "  -np 1 ^",
                "  --jinja",
            ]
        )
        if cfg.inference.flash_attn:
            # Supported on many builds; if unknown, server may ignore or fail — documented.
            lines[-1] = "  --jinja ^"
            lines.append("  -fa on")

        bat.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for p in (out_log, err_log):
            if p.exists():
                with contextlib.suppress(OSError):
                    p.unlink()

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        log_event(
            _log,
            "server.starting",
            device=device,
            n_gpu_layers=cfg.inference.n_gpu_layers,
            n_ctx=cfg.inference.n_ctx,
            port=cfg.inference.port,
            bat=str(bat),
        )
        with open(out_log, "w", encoding="utf-8") as fo, open(err_log, "w", encoding="utf-8") as fe:
            self._proc = subprocess.Popen(
                ["cmd.exe", "/c", str(bat)],
                stdout=fo,
                stderr=fe,
                cwd=str(llama_dir),
                creationflags=creationflags,
            )
        self._started_by_us = True
        _MANAGED = self
        if not self._atexit_registered:
            atexit.register(_atexit_stop)
            self._atexit_registered = True

        timeout = cfg.inference.server_start_timeout_sec
        deadline = time.time() + timeout
        while time.time() < deadline:
            get_global_token().check()
            if self._proc.poll() is not None:
                err_tail = ""
                if err_log.exists():
                    err_tail = err_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise InferenceError(
                    f"llama-server 启动后立即退出 code={self._proc.returncode}",
                    code=ErrorCode.MODEL_LOAD_FAILED,
                    details={"stderr_tail": err_tail, "err_log": str(err_log)},
                )
            if self.health():
                self._load_ms = (time.perf_counter() - t0) * 1000
                props = self.props()
                modalities = props.get("modalities") if isinstance(props, dict) else None
                vision_ok = bool(isinstance(modalities, dict) and modalities.get("vision"))
                info = {
                    "already_running": False,
                    "started_by_us": True,
                    "base": self.base,
                    "load_ms": round(self._load_ms, 2),
                    "pid": self._proc.pid,
                    "device": device,
                    "n_gpu_layers": cfg.inference.n_gpu_layers,
                    "n_ctx": cfg.inference.n_ctx,
                    "n_batch": cfg.inference.n_batch,
                    "flash_attn": cfg.inference.flash_attn,
                    "props": props,
                    "vision": vision_ok,
                    "backend": "llama-server SYCL",
                    "cpu_fallback": False,
                    "out_log": str(out_log),
                    "err_log": str(err_log),
                }
                log_event(_log, "server.ready", **{k: v for k, v in info.items() if k != "props"})
                do_warm = cfg.inference.warmup_on_start if warmup is None else warmup
                if do_warm:
                    info["warmup"] = self.warmup()
                return info
            time.sleep(1.0)

        raise InferenceError(
            f"llama-server 在 {timeout}s 内未就绪",
            code=ErrorCode.MODEL_LOAD_FAILED,
            details={"err_log": str(err_log), "out_log": str(out_log)},
        )

    def warmup(self) -> dict[str, Any]:
        """Tiny completion to prime kernels / KV."""
        get_global_token().check()
        body = {
            "messages": [
                {"role": "system", "content": 'Reply with JSON only: {"ok": true}'},
                {"role": "user", "content": "ping"},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        out, err, dt = _http_json(
            "POST",
            f"{self.base}/v1/chat/completions",
            body,
            timeout=min(120.0, self.config.inference.request_timeout_sec),
        )
        self._warmup_ms = (time.perf_counter() - t0) * 1000
        result = {
            "ok": err is None and out is not None,
            "warmup_ms": round(self._warmup_ms, 2),
            "http_ms": round(dt, 2),
            "error": err,
        }
        log_event(_log, "server.warmup", **result)
        return result

    def stop(self, *, force: bool = False) -> None:
        """Release the managed process (does not kill externally started servers)."""
        global _MANAGED
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if not self._started_by_us and not force:
            return
        log_event(_log, "server.stopping", pid=getattr(proc, "pid", None), force=force)
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log_event(_log, "server.stop_error", error=str(exc))
        self._started_by_us = False
        if _MANAGED is self:
            _MANAGED = None

    def recover(self) -> dict[str, Any]:
        """Stop managed process (if any) and start again."""
        log_event(_log, "server.recover")
        self.stop(force=True)
        time.sleep(1.0)
        self._warmup_ms = None
        self._load_ms = None
        return self.start(warmup=True)

    def status(self) -> dict[str, Any]:
        healthy = self.health()
        return {
            "healthy": healthy,
            "base": self.base,
            "started_by_us": self._started_by_us,
            "managed_pid": self._proc.pid if self._proc and self._proc.poll() is None else None,
            "load_ms": self._load_ms,
            "warmup_ms": self._warmup_ms,
            "props": self.props() if healthy else {},
        }

    def close(self) -> None:
        self.stop()


def _atexit_stop() -> None:
    global _MANAGED
    if _MANAGED is not None:
        with contextlib.suppress(Exception):
            _MANAGED.stop()
        _MANAGED = None
