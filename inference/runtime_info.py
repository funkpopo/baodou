"""Record llama.cpp binary version, compile options, and host deps (Phase E)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.config import AppConfig


def probe_llama_binary(llama_dir: Path) -> dict[str, Any]:
    """Inspect D:\\llama without starting the full model."""
    exe = llama_dir / "llama-server.exe"
    info: dict[str, Any] = {
        "llama_dir": str(llama_dir),
        "llama_server": str(exe),
        "exists": exe.exists(),
        "version": None,
        "build": None,
        "compiler": None,
        "platform": None,
        "dlls": {},
        "error": None,
    }
    if not exe.exists():
        info["error"] = "llama-server.exe missing"
        return info

    # Companion libs that define the SYCL stack
    for name in (
        "ggml-sycl.dll",
        "ggml.dll",
        "ggml-base.dll",
        "llama-common.dll",
        "mtmd.dll",
        "llama-server-impl.dll",
    ):
        p = llama_dir / name
        info["dlls"][name] = p.exists()

    try:
        proc = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(llama_dir),
        )
        raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
        info["raw_version"] = raw.strip()[:500]
        # e.g. version: 10356 (0666ad2b2)
        m = re.search(r"version:\s*(\d+)\s*\(([^)]+)\)", raw)
        if m:
            info["version"] = int(m.group(1))
            info["build"] = m.group(2)
        m2 = re.search(r"built with\s+(.+?)\s+for\s+(.+)", raw)
        if m2:
            info["compiler"] = m2.group(1).strip()
            info["platform"] = m2.group(2).strip()
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_runtime_record(config: AppConfig) -> dict[str, Any]:
    """Full Phase E runtime record for docs / benchmarks."""
    llama = probe_llama_binary(config.paths.llama_dir)
    return {
        "phase": "E",
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "paths": {
            "model_gguf": str(config.paths.model_gguf),
            "mmproj_gguf": str(config.paths.mmproj_gguf),
            "model_exists": config.paths.model_gguf.exists(),
            "mmproj_exists": config.paths.mmproj_gguf.exists(),
            "llama_dir": str(config.paths.llama_dir),
            "oneapi_setvars": str(config.paths.oneapi_setvars),
            "oneapi_exists": config.paths.oneapi_setvars.exists(),
        },
        "inference": {
            "host": config.inference.host,
            "port": config.inference.port,
            "n_ctx": config.inference.n_ctx,
            "n_gpu_layers": config.inference.n_gpu_layers,
            "device": config.inference.device,
            "n_threads": config.inference.n_threads,
            "n_batch": config.inference.n_batch,
            "flash_attn": config.inference.flash_attn,
            "max_tokens": config.inference.max_tokens,
            "temperature": config.inference.temperature,
            "enable_thinking": config.inference.enable_thinking,
            "mmproj_offload": config.inference.mmproj_offload,
            "constraint_mode": config.inference.constraint_mode,
            "stream": config.inference.stream,
            "backend": config.inference.backend,
        },
        "llama": llama,
        "compile_notes": {
            "backend": "ggml-sycl",
            "gpu_policy": "force -dev SYCL0 -ngl 99; never default CPU",
            "chat_template": "--jinja (model built-in)",
            "vision": "--mmproj + --mmproj-offload",
            "source_tree": "prebuilt D:\\llama (user-provided SYCL build)",
        },
    }
