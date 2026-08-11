"""Collect host hardware baseline for Phase A feasibility (Windows + Intel Arc SYCL)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

try:
    import mss
except ImportError:  # pragma: no cover
    mss = None


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "hardware_baseline.json"
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", r"D:\llama"))
ONEAPI_SETVARS = Path(os.environ.get("ONEAPI_SETVARS", r"D:\Intel\oneAPI\setvars.bat"))


def _ps(query: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (r.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _cmd_with_oneapi(inner: str, timeout: int = 90) -> tuple[str, str, int]:
    """Run a cmd after oneAPI setvars via a temporary .bat (more reliable on Windows)."""
    art = Path(__file__).resolve().parent / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    bat = art / "_oneapi_probe.bat"
    out_file = art / "_oneapi_probe.out"
    err_file = art / "_oneapi_probe.err"
    lines = ["@echo off"]
    if ONEAPI_SETVARS.exists():
        # Do not swallow setvars errors entirely; still continue
        lines.append(f'call "{ONEAPI_SETVARS}"')
    lines.append(f'set "PATH={LLAMA_DIR};%PATH%"')
    lines.append(inner)
    bat.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", errors="replace")
    try:
        with open(out_file, "w", encoding="utf-8", errors="replace") as fo, open(
            err_file, "w", encoding="utf-8", errors="replace"
        ) as fe:
            r = subprocess.run(
                ["cmd.exe", "/c", str(bat)],
                stdout=fo,
                stderr=fe,
                timeout=timeout,
                check=False,
                cwd=str(LLAMA_DIR),
            )
        stdout = out_file.read_text(encoding="utf-8", errors="replace").strip()
        stderr = err_file.read_text(encoding="utf-8", errors="replace").strip()
        return stdout, stderr, r.returncode
    except Exception as exc:  # noqa: BLE001
        return "", f"error: {exc}", 1


def collect_sycl() -> dict:
    sycl_ls = LLAMA_DIR / "sycl-ls.exe"
    llama_cli = LLAMA_DIR / "llama-cli.exe"
    out: dict = {
        "llama_dir": str(LLAMA_DIR),
        "oneapi_setvars": str(ONEAPI_SETVARS),
        "oneapi_setvars_exists": ONEAPI_SETVARS.exists(),
        "sycl_ls_exists": sycl_ls.exists(),
        "llama_cli_exists": llama_cli.exists(),
        "ggml_sycl_dll": (LLAMA_DIR / "ggml-sycl.dll").exists(),
    }
    if sycl_ls.exists():
        stdout, stderr, code = _cmd_with_oneapi(f'"{sycl_ls}"')
        # sycl-ls prints devices on stdout; setvars noise may be mixed
        combined = "\n".join(x for x in (stdout, stderr) if x)
        device_lines = [
            ln.strip()
            for ln in combined.splitlines()
            if ln.strip().startswith("[") and "]" in ln
        ]
        out["sycl_ls"] = {
            "returncode": code,
            "device_lines": device_lines,
            "stdout_tail": stdout[-1500:],
            "stderr_tail": stderr[-800:],
        }
        out["sycl_gpu_present"] = any("level_zero:gpu" in ln or "opencl:gpu" in ln for ln in device_lines)
    if llama_cli.exists():
        stdout, stderr, code = _cmd_with_oneapi(f'"{llama_cli}" --list-devices')
        text = "\n".join(x for x in (stdout, stderr) if x)
        out["llama_list_devices"] = {"returncode": code, "text_tail": text[-2000:]}
        devices = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("available"):
                continue
            # e.g. "SYCL0: Intel(R) Arc(TM) A770 Graphics (...)"
            if line.upper().startswith("SYCL") or line.upper().startswith("GPU"):
                devices.append(line)
        out["devices_parsed"] = devices
        sycl_gpus = [d for d in devices if d.upper().startswith("SYCL") and "CPU" not in d.upper()]
        out["recommended_device"] = (
            "SYCL0" if any(d.upper().startswith("SYCL0") for d in devices) else (sycl_gpus[0].split(":")[0] if sycl_gpus else None)
        )
        out["gpu_offload_ready"] = bool(sycl_gpus) or bool(out.get("sycl_gpu_present"))
        if out["gpu_offload_ready"] and not out.get("recommended_device"):
            out["recommended_device"] = "SYCL0"
    return out


def collect() -> dict:
    screens = []
    if mss is not None:
        sct_cm = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        with sct_cm as sct:
            for i, mon in enumerate(sct.monitors):
                screens.append(
                    {
                        "index": i,
                        "left": mon["left"],
                        "top": mon["top"],
                        "width": mon["width"],
                        "height": mon["height"],
                        "is_virtual_all": i == 0,
                    }
                )

    cpu_freq = psutil.cpu_freq()
    vm = psutil.virtual_memory()
    data = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "cpu": {
            "name": platform.processor(),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "freq_mhz": {
                "current": cpu_freq.current if cpu_freq else None,
                "min": cpu_freq.min if cpu_freq else None,
                "max": cpu_freq.max if cpu_freq else None,
            },
            "wmi_name": _ps("(Get-CimInstance Win32_Processor).Name"),
        },
        "memory": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_percent": vm.percent,
        },
        "gpu": {
            "wmi": _ps(
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM, DriverVersion | Format-List | Out-String"
            ),
            "notes": (
                "Primary discrete GPU: Intel Arc A770. "
                "Inference must use D:\\llama SYCL build (ggml-sycl), not CUDA and not llama-cpp-python CPU wheel. "
                "Initialize with D:\\Intel\\oneAPI\\setvars.bat before llama-server."
            ),
            "sycl": collect_sycl(),
        },
        "screens": screens,
        "conda": {
            "expected_env": "dev",
            "active_hint": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "target_release": {
            "os": "Windows 11 (x64) only for MVP",
            "form": "Local desktop app + independent llama-server inference process (GPU)",
            "macos_linux": "Out of scope for phase 1",
        },
    }
    return data


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = collect()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT}")
    sycl = data.get("gpu", {}).get("sycl", {})
    if not sycl.get("gpu_offload_ready"):
        print("WARNING: no SYCL GPU device detected; Phase A requires GPU.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
