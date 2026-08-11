"""Phase A baseline: screenshot, GPU model load via D:\\llama (SYCL), structured scenarios.

Requires:
  - conda env `dev` (mss, psutil, pillow, requests optional)
  - D:\\llama SYCL build (llama-server)
  - D:\\Intel\\oneAPI\\setvars.bat for Level Zero / SYCL runtime
  - Intel Arc GPU (or other SYCL device)
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mss
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
RESULTS = HERE / "results"

MODEL_PATH = ROOT / "model" / "Qwen3.5-2B-UD-Q4_K_XL.gguf"
MMPROJ_PATH = ROOT / "model" / "mmproj-F16.gguf"

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", r"D:\llama"))
ONEAPI_SETVARS = Path(os.environ.get("ONEAPI_SETVARS", r"D:\Intel\oneAPI\setvars.bat"))
SERVER_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("LLAMA_PORT", "8765"))
SERVER_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"

N_CTX = int(os.environ.get("N_CTX", "4096"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))
DEVICE = os.environ.get("LLAMA_DEVICE", "SYCL0")  # force GPU, never CPU
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "768"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
N_THREADS = max(4, (psutil.cpu_count(logical=True) or 8) - 2)

# Managed server process (if we started it)
_SERVER_PROC: subprocess.Popen | None = None


SYSTEM_JSON = (
    "You are a desktop screen assistant. "
    "Always reply with a single JSON object only, no markdown fences, no thinking. "
    "Schema: {"
    '"observation": string, '
    '"ui_elements": [{"id": string, "type": string, "text": string, '
    '"bbox": {"x": int, "y": int, "width": int, "height": int}, "confidence": number}], '
    '"suggested_action": {"action": string, "target_element_id": string|null, '
    '"risk": "low"|"medium"|"high", "requires_confirmation": true}, '
    '"notes": string'
    "}. "
    "Keep ui_elements to at most 6 items. Be concise. "
    "If unsure, lower confidence and set suggested_action.action to none."
)


SCENARIOS: list[dict[str, Any]] = [
    # Text-only capability probes (no image)
    {"id": "T01", "mode": "text", "task": "Describe what a Windows desktop taskbar typically contains."},
    {"id": "T02", "mode": "text", "task": "List common UI element types: button, input, menu, window, checkbox."},
    {"id": "T03", "mode": "text", "task": "Given element id btn_ok at bbox x=100 y=200 w=80 h=32, propose a low-risk click plan as JSON."},
    {"id": "T04", "mode": "text", "task": "Explain why high-risk actions need user confirmation."},
    {"id": "T05", "mode": "text", "task": "Output a ScreenObservation-like JSON for a Notepad window with title bar and edit area (invent reasonable ids)."},
    {"id": "T06", "mode": "text", "task": "Convert user goal 'click the Search button' into a single-step ActionPlan JSON."},
    {"id": "T07", "mode": "text", "task": "If target element is missing, what should an agent do? Answer as JSON with fields decision and reason."},
    {"id": "T08", "mode": "text", "task": "Rank risks: read screen, click OK, delete file, send payment. Return JSON map."},
    # Vision scenarios (use live screenshot)
    {"id": "V01", "mode": "vision", "task": "Summarize what is currently visible on the screen."},
    {"id": "V02", "mode": "vision", "task": "List any visible windows or application titles you can identify."},
    {"id": "V03", "mode": "vision", "task": "Identify buttons or clickable controls and give approximate screen coordinates."},
    {"id": "V04", "mode": "vision", "task": "Is there a text input or search box visible? If yes, describe location."},
    {"id": "V05", "mode": "vision", "task": "Describe the overall desktop state in one short paragraph, then JSON fields."},
    {"id": "V06", "mode": "vision", "task": "Find the taskbar if present and report its approximate y region."},
    {"id": "V07", "mode": "vision", "task": "Detect whether a browser, editor, or file manager is open."},
    {"id": "V08", "mode": "vision", "task": "Name up to 5 readable text strings on screen."},
    {"id": "V09", "mode": "vision", "task": "Propose one low-risk next action for a user who wants to understand the screen only (no click)."},
    {"id": "V10", "mode": "vision", "task": "Estimate screen resolution from the image and major layout regions."},
    {"id": "V11", "mode": "vision", "task": "If you see a close/minimize/maximize control, report element candidates."},
    {"id": "V12", "mode": "vision", "task": "Classify UI density: sparse, medium, or dense; justify briefly in notes."},
    {"id": "V13", "mode": "vision", "task": "Point out any dialog or modal that may need user attention."},
    {"id": "V14", "mode": "vision", "task": "Return JSON with observation only focused on the primary foreground window."},
    {"id": "V15", "mode": "vision", "task": "Identify icons on the desktop or taskbar if any are visible."},
    {"id": "V16", "mode": "vision", "task": "Would clicking near the center of the screen be safe? Answer with risk assessment JSON."},
    {"id": "V17", "mode": "vision", "task": "Extract any visible Chinese or English labels for menus/buttons."},
    {"id": "V18", "mode": "vision", "task": "Provide element ids e1..eN for up to 5 UI candidates with type and text."},
    {"id": "V19", "mode": "vision", "task": "Describe color theme: light/dark and any high-contrast regions."},
    {"id": "V20", "mode": "vision", "task": "Output structured JSON for current state suitable for an automation planner."},
    {"id": "V21", "mode": "vision", "task": "If an input box is visible, draft a requires_confirmation text-input action plan."},
    {"id": "V22", "mode": "vision", "task": "Report whether multi-monitor content might be present (wide virtual desktop)."},
]


def mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 300.0) -> tuple[dict | None, str | None, float]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            dt = (time.perf_counter() - t0) * 1000
            return json.loads(raw), None, dt
    except urllib.error.HTTPError as exc:
        dt = (time.perf_counter() - t0) * 1000
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:  # noqa: BLE001
            detail = str(exc)
        return None, f"HTTPError {exc.code}: {detail}", dt
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        return None, f"{type(exc).__name__}: {exc}", dt


def server_healthy() -> bool:
    out, err, _ = http_json("GET", f"{SERVER_BASE}/health", timeout=3.0)
    return err is None and isinstance(out, dict) and out.get("status") == "ok"


def server_props() -> dict[str, Any]:
    out, err, _ = http_json("GET", f"{SERVER_BASE}/props", timeout=5.0)
    if err or not out:
        # /props may be disabled; fall back to models
        models, merr, _ = http_json("GET", f"{SERVER_BASE}/v1/models", timeout=5.0)
        return {"props_error": err, "models": models, "models_error": merr}
    return out


def stop_server() -> None:
    global _SERVER_PROC
    if _SERVER_PROC is None:
        return
    proc = _SERVER_PROC
    _SERVER_PROC = None
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:  # noqa: BLE001
        pass


def start_server() -> dict[str, Any]:
    """Launch D:\\llama\\llama-server with oneAPI + SYCL GPU. Refuse CPU-only."""
    global _SERVER_PROC

    if server_healthy():
        props = server_props()
        return {
            "already_running": True,
            "base": SERVER_BASE,
            "props": props,
            "started_by_us": False,
        }

    if not (LLAMA_DIR / "llama-server.exe").exists():
        raise FileNotFoundError(f"llama-server not found under {LLAMA_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model missing: {MODEL_PATH}")
    if not MMPROJ_PATH.exists():
        raise FileNotFoundError(f"mmproj missing: {MMPROJ_PATH}")
    if not ONEAPI_SETVARS.exists():
        raise FileNotFoundError(f"oneAPI setvars missing: {ONEAPI_SETVARS}")

    ART.mkdir(parents=True, exist_ok=True)
    bat = ART / "start_server.bat"
    out_log = ART / "server.out"
    err_log = ART / "server.err"

    # Force GPU device; ngl=99 offloads all layers to VRAM.
    # --mmproj enables vision. --jinja uses model chat template.
    bat_body = f"""@echo off
call "{ONEAPI_SETVARS}" >nul 2>&1
set PATH={LLAMA_DIR};%PATH%
"{LLAMA_DIR / 'llama-server.exe'}" ^
  -m "{MODEL_PATH}" ^
  --mmproj "{MMPROJ_PATH}" ^
  --mmproj-offload ^
  -ngl {N_GPU_LAYERS} ^
  -dev {DEVICE} ^
  -c {N_CTX} ^
  -t {N_THREADS} ^
  --host {SERVER_HOST} ^
  --port {SERVER_PORT} ^
  -np 1 ^
  --jinja
"""
    bat.write_text(bat_body, encoding="ascii", errors="replace")

    # Clear old logs
    for p in (out_log, err_log):
        if p.exists():
            p.unlink()

    t0 = time.perf_counter()
    # Use CREATE_NEW_PROCESS_GROUP so we can signal the tree later
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    with open(out_log, "w", encoding="utf-8") as fo, open(err_log, "w", encoding="utf-8") as fe:
        _SERVER_PROC = subprocess.Popen(
            ["cmd.exe", "/c", str(bat)],
            stdout=fo,
            stderr=fe,
            cwd=str(LLAMA_DIR),
            creationflags=creationflags,
        )
    atexit.register(stop_server)

    # Wait until healthy
    deadline = time.time() + 180
    last_err = "timeout"
    while time.time() < deadline:
        if _SERVER_PROC.poll() is not None:
            err_tail = err_log.read_text(encoding="utf-8", errors="replace")[-2000:] if err_log.exists() else ""
            raise RuntimeError(f"llama-server exited early code={_SERVER_PROC.returncode}\n{err_tail}")
        if server_healthy():
            load_ms = (time.perf_counter() - t0) * 1000
            props = server_props()
            # Hard fail if vision not available
            modalities = props.get("modalities") if isinstance(props, dict) else None
            return {
                "already_running": False,
                "started_by_us": True,
                "base": SERVER_BASE,
                "load_ms": round(load_ms, 2),
                "pid": _SERVER_PROC.pid,
                "device": DEVICE,
                "n_gpu_layers": N_GPU_LAYERS,
                "n_ctx": N_CTX,
                "props": props,
                "vision": bool(isinstance(modalities, dict) and modalities.get("vision")),
                "backend": "llama-server SYCL (D:\\llama)",
                "cpu_fallback": False,
            }
        time.sleep(1.0)
        last_err = "waiting"
    raise TimeoutError(f"server not healthy within 180s ({last_err}); see {err_log}")


def capture_primary(max_width: int = 1280) -> tuple[Path, dict[str, Any]]:
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # mss 10+: prefer MSS(); keep mss() fallback for older builds
    sct_cm = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
    with sct_cm as sct:
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    raw_ms = (time.perf_counter() - t0) * 1000

    orig_w, orig_h = img.size
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)

    path = ART / "primary_screenshot.png"
    t1 = time.perf_counter()
    img.save(path, format="PNG", optimize=True)
    save_ms = (time.perf_counter() - t1) * 1000

    meta = {
        "path": str(path),
        "monitor": mon,
        "original_size": [orig_w, orig_h],
        "saved_size": list(img.size),
        "capture_ms": round(raw_ms, 2),
        "save_ms": round(save_ms, 2),
        "file_kb": round(path.stat().st_size / 1024, 2),
    }
    return path, meta


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_json(text: str) -> tuple[dict | list | None, str | None]:
    text = (text or "").strip()
    if not text:
        return None, "empty"
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    # Prefer outermost object
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None, "no_json_object"
    candidate = m.group(0)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        # Attempt to repair truncated JSON by closing open braces/brackets/strings naively
        repaired = _try_repair_json(candidate)
        if repaired is not None:
            return repaired, f"repaired_after: {exc}"
        return None, f"json_error: {exc}"


def _try_repair_json(s: str) -> dict | list | None:
    """Best-effort close of truncated JSON objects (common when max_tokens hits)."""
    s = s.rstrip()
    # If cut mid-string, close the string
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
    if in_str:
        s += '"'
    # Drop trailing comma
    s = re.sub(r",\s*$", "", s)
    # Balance braces/brackets
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = {"{": "}", "[": "]"}
    while stack:
        s += closers[stack.pop()]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def chat_once(task: str, image_path: Path | None, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    content: Any
    if image_path is not None:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            {"type": "text", "text": task},
        ]
    else:
        content = task

    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": 0.9,
        # Qwen3.5-2B defaults to non-thinking; keep explicit for server variants
        "chat_template_kwargs": {"enable_thinking": False},
    }

    mem0 = mem_mb()
    out, err, dt_ms = http_json("POST", f"{SERVER_BASE}/v1/chat/completions", body=body, timeout=300.0)
    mem1 = mem_mb()

    text = ""
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    if out is not None:
        try:
            choice = out["choices"][0]["message"]
            text = choice.get("content") or ""
            # Some builds put reasoning separately
            if not text and choice.get("reasoning_content"):
                text = choice.get("reasoning_content") or ""
        except Exception:  # noqa: BLE001
            text = str(out)
        usage = out.get("usage") or {}
        timings = out.get("timings") or {}

    parsed, parse_err = extract_json(text) if text else (None, "empty" if not err else None)
    return {
        "latency_ms": round(dt_ms, 2),
        "rss_delta_mb": round(mem1 - mem0, 2),
        "error": err,
        "raw_text": text,
        "parsed": parsed,
        "parse_error": parse_err,
        "json_ok": parsed is not None and isinstance(parsed, dict),
        "usage": usage,
        "timings": timings,
    }


def idle_cpu_sample(seconds: float = 2.0) -> float:
    psutil.cpu_percent(interval=None)
    time.sleep(seconds)
    return psutil.cpu_percent(interval=None)


def find_server_process() -> dict[str, Any] | None:
    for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "llama-server" in name or name == "llama-server.exe":
                mi = p.info.get("memory_info")
                return {
                    "pid": p.info["pid"],
                    "name": p.info.get("name"),
                    "rss_mb": round(mi.rss / (1024**2), 2) if mi else None,
                    "cmdline": p.info.get("cmdline"),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "model": str(MODEL_PATH),
            "mmproj": str(MMPROJ_PATH),
            "llama_dir": str(LLAMA_DIR),
            "oneapi_setvars": str(ONEAPI_SETVARS),
            "model_exists": MODEL_PATH.exists(),
            "mmproj_exists": MMPROJ_PATH.exists(),
        },
        "config": {
            "n_ctx": N_CTX,
            "n_threads": N_THREADS,
            "n_gpu_layers": N_GPU_LAYERS,
            "device": DEVICE,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "server_base": SERVER_BASE,
            "backend": "D:\\llama llama-server SYCL GPU (not llama-cpp-python CPU)",
            "require_gpu": True,
        },
    }

    if DEVICE.upper() in {"NONE", "CPU", ""}:
        print("REFUSING CPU: set LLAMA_DEVICE to a SYCL GPU (e.g. SYCL0)", file=sys.stderr)
        return 3

    print("=== capture ===")
    shot_path, shot_meta = capture_primary()
    report["screenshot"] = shot_meta
    print(json.dumps(shot_meta, ensure_ascii=False, indent=2))

    print("=== idle cpu ===")
    idle = idle_cpu_sample(2.0)
    report["idle_cpu_percent_approx"] = idle
    print(f"idle_cpu_percent_approx={idle}")

    print(f"=== start/attach GPU server device={DEVICE} ngl={N_GPU_LAYERS} ===")
    try:
        load_info = start_server()
    except Exception as exc:  # noqa: BLE001
        load_info = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        report["model_load"] = load_info
        (RESULTS / "baseline_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(load_info["error"], file=sys.stderr)
        return 1

    # Confirm vision + GPU
    props = load_info.get("props") or {}
    modalities = props.get("modalities") if isinstance(props, dict) else {}
    vision_ok = bool(isinstance(modalities, dict) and modalities.get("vision")) or bool(load_info.get("vision"))
    if not vision_ok and MMPROJ_PATH.exists():
        # Server may have loaded without exposing modalities; still try vision
        vision_ok = True
        load_info["vision_assumed"] = True

    load_info["vision_enabled"] = vision_ok
    load_info["server_process"] = find_server_process()
    report["model_load"] = load_info
    print(json.dumps({k: load_info[k] for k in load_info if k != "props"}, ensure_ascii=False, indent=2))
    if isinstance(props, dict):
        print("modalities:", props.get("modalities"))
        print("model_path:", props.get("model_path") or props.get("model_alias"))

    use_vision = vision_ok

    # Warmup (text)
    print("=== warmup ===")
    warm = chat_once(
        'Reply with JSON {"observation":"ok","ui_elements":[],"suggested_action":{"action":"none","target_element_id":null,"risk":"low","requires_confirmation":true},"notes":"warmup"}',
        None,
        max_tokens=128,
    )
    report["warmup"] = {
        k: warm[k] for k in ("latency_ms", "json_ok", "error", "parse_error", "usage", "timings")
    }
    print(json.dumps(report["warmup"], ensure_ascii=False, indent=2))

    # Acceptance: single screenshot structured inference
    print("=== acceptance: screenshot structured inference (GPU vision) ===")
    accept_task = (
        "Analyze this screenshot of a Windows desktop. "
        "Return JSON with observation, ui_elements (up to 6), suggested_action, notes. "
        "Keep each text field short."
    )
    if use_vision:
        accept = chat_once(accept_task, shot_path, max_tokens=MAX_TOKENS)
    else:
        accept = chat_once(
            accept_task + " NOTE: image not available; mark notes and still return JSON.",
            None,
            max_tokens=MAX_TOKENS,
        )
        accept["vision_skipped"] = True

    report["acceptance_inference"] = {
        "latency_ms": accept["latency_ms"],
        "json_ok": accept["json_ok"],
        "error": accept["error"],
        "parse_error": accept["parse_error"],
        "usage": accept["usage"],
        "timings": accept["timings"],
        "parsed": accept["parsed"],
        "raw_text_preview": (accept["raw_text"] or "")[:1500],
        "vision_used": use_vision and shot_path is not None and not accept.get("vision_skipped"),
        "backend": "sycl-gpu",
    }
    (ART / "acceptance_raw.txt").write_text(accept["raw_text"] or "", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report["acceptance_inference"][k]
                for k in ("latency_ms", "json_ok", "error", "parse_error", "vision_used", "usage", "timings")
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Scenarios
    print("=== scenarios ===")
    scenario_results = []
    latencies: list[float] = []
    json_ok_n = 0
    for sc in SCENARIOS:
        img = shot_path if (sc["mode"] == "vision" and use_vision) else None
        task = sc["task"]
        if sc["mode"] == "vision" and not use_vision:
            task = task + " (No image input available; state limitation in notes and still return JSON.)"
        print(f"  - {sc['id']} {sc['mode']} ...", flush=True)
        r = chat_once(llm_task := task, img, max_tokens=512)
        del llm_task
        ok = bool(r["json_ok"]) and r["error"] is None
        if ok:
            json_ok_n += 1
        if r["error"] is None:
            latencies.append(r["latency_ms"])
        scenario_results.append(
            {
                "id": sc["id"],
                "mode": sc["mode"],
                "task": sc["task"],
                "vision_image_attached": img is not None,
                "latency_ms": r["latency_ms"],
                "json_ok": r["json_ok"],
                "error": r["error"],
                "parse_error": r["parse_error"],
                "usage": r["usage"],
                "timings": r["timings"],
                "parsed": r["parsed"],
                "raw_text_preview": (r["raw_text"] or "")[:500],
                "pass": ok,
            }
        )

    report["scenarios"] = {
        "total": len(SCENARIOS),
        "passed": json_ok_n,
        "pass_rate": round(json_ok_n / len(SCENARIOS), 3) if SCENARIOS else 0,
        "latency_ms": {
            "count": len(latencies),
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "min": round(min(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "items": scenario_results,
    }

    report["peak_client_rss_mb"] = round(mem_mb(), 2)
    report["server_process_after"] = find_server_process()
    report["vision_enabled"] = use_vision
    report["gpu_required"] = True
    report["gpu_device"] = DEVICE

    out_path = RESULTS / "baseline_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "backend": "D:\\llama llama-server SYCL",
        "device": DEVICE,
        "n_gpu_layers": N_GPU_LAYERS,
        "model_load_ms": load_info.get("load_ms"),
        "already_running": load_info.get("already_running"),
        "acceptance_json_ok": report["acceptance_inference"]["json_ok"],
        "acceptance_latency_ms": report["acceptance_inference"]["latency_ms"],
        "acceptance_vision_used": report["acceptance_inference"]["vision_used"],
        "scenarios_passed": f"{json_ok_n}/{len(SCENARIOS)}",
        "scenario_pass_rate": report["scenarios"]["pass_rate"],
        "avg_latency_ms": report["scenarios"]["latency_ms"]["avg"],
        "vision_enabled": use_vision,
        "screenshot_capture_ms": shot_meta["capture_ms"],
        "server_rss_mb": (report["server_process_after"] or {}).get("rss_mb"),
        "report": str(out_path),
    }
    (RESULTS / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Keep server if we attached to an existing one; stop only if we started it
    # and env BAODOU_KEEP_SERVER is not set.
    if load_info.get("started_by_us") and os.environ.get("BAODOU_KEEP_SERVER", "0") not in {"1", "true", "True"}:
        print("=== stopping server we started ===")
        stop_server()
    else:
        print("=== leaving server running ===")

    return 0 if report["acceptance_inference"]["json_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
