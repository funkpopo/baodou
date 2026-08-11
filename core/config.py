"""Load and validate application configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from core.errors import ConfigError

# Project root: parent of this package's parent? core/ is at repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class AppSection(BaseModel):
    name: str = "baodou"
    env: str = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    log_dir: str = "logs"
    graceful_shutdown_sec: float = 5.0


class PathsSection(BaseModel):
    project_root: Path | None = None
    model_gguf: Path = Path("model/Qwen3.5-2B-UD-Q4_K_XL.gguf")
    mmproj_gguf: Path = Path("model/mmproj-F16.gguf")
    llama_dir: Path = Path(r"D:/llama")
    oneapi_setvars: Path = Path(r"D:/Intel/oneAPI/setvars.bat")

    def resolve(self, root: Path) -> PathsSection:
        data = self.model_dump()
        data["project_root"] = root
        for key in ("model_gguf", "mmproj_gguf"):
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = (root / p).resolve()
            else:
                data[key] = p.resolve()
        data["llama_dir"] = Path(data["llama_dir"]).resolve()
        data["oneapi_setvars"] = Path(data["oneapi_setvars"]).resolve()
        return PathsSection(**data)


class InferenceSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    n_ctx: int = 4096
    n_gpu_layers: int = 99
    device: str = "SYCL0"
    n_threads: int = 0
    max_tokens: int = 768
    temperature: float = 0.3
    enable_thinking: bool = False
    request_timeout_sec: float = 60.0
    health_timeout_sec: float = 5.0
    mmproj_offload: bool = True
    backend: Literal["mock", "http"] = "mock"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class StreamProfile(BaseModel):
    """Per-kind capture quality / rate (0 fps = event-driven only)."""

    fps: float = 0.0
    max_width: int = 1280
    max_height: int = 720
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = 85
    only_on_change: bool = False


class PrivacySection(BaseModel):
    enabled: bool = True
    mask_color: list[int] = Field(default_factory=lambda: [0, 0, 0])
    # Manual masks in virtual-desktop physical pixels: {x,y,width,height,reason?}
    manual_masks: list[dict[str, Any]] = Field(default_factory=list)
    # Title substrings → full-window mask when capturing that window / if visible.
    privacy_window_titles: list[str] = Field(
        default_factory=lambda: ["密码", "Password", "Credential", "银行", "Bank"]
    )
    mask_password_class_names: list[str] = Field(
        default_factory=lambda: ["Edit", "PasswordBox"]  # heuristic; refined in later phases
    )


class CaptureSection(BaseModel):
    mode: Literal["primary", "all", "window", "region"] = "primary"
    # mss monitor index: 0 = virtual all, 1..N = physical (1 usually primary)
    monitor_index: int = 1
    target_fps: float = 5.0
    max_width: int = 1280
    max_height: int = 720
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = 85
    color_mode: Literal["RGB", "L"] = "RGB"
    queue_size: int = 4
    drop_policy: Literal["newest", "oldest"] = "newest"
    backend: Literal["mock", "mss"] = "mss"
    # Region mode (physical virtual-desktop pixels)
    region: dict[str, int] | None = None
    # Window mode
    window_title: str | None = None
    window_hwnd: int | None = None
    # Change detection
    change_threshold: float = 0.015
    change_sample_size: int = 64
    force_on_user_request: bool = True
    force_on_verify: bool = True
    # Save optional artifacts under project
    save_frames: bool = False
    save_dir: str = "benchmarks/phase_c/artifacts"
    include_b64: bool = False
    privacy: PrivacySection = Field(default_factory=PrivacySection)
    streams: dict[str, StreamProfile] = Field(
        default_factory=lambda: {
            "preview": StreamProfile(
                fps=10.0,
                max_width=960,
                max_height=540,
                image_format="jpeg",
                jpeg_quality=70,
                only_on_change=False,
            ),
            "vision": StreamProfile(
                fps=8.0,
                max_width=1280,
                max_height=720,
                image_format="png",
                only_on_change=True,
            ),
            "model": StreamProfile(
                fps=0.0,
                max_width=1280,
                max_height=720,
                image_format="png",
                only_on_change=True,
            ),
            "verify": StreamProfile(
                fps=0.0,
                max_width=1280,
                max_height=720,
                image_format="png",
                only_on_change=False,
            ),
        }
    )


class UIVisionSection(BaseModel):
    confidence_threshold: float = 0.5
    max_elements: int = 64
    timeout_ms: int = 1500
    # Sources for composite backend: uia | ocr | rules | mock
    sources: list[str] = Field(default_factory=lambda: ["uia", "rules"])
    # mock | composite | uia | ocr | rules
    backend: Literal["mock", "composite", "uia", "ocr", "rules"] = "composite"
    fuse_iou_threshold: float = 0.45
    uia_max_depth: int = 12
    ocr_enabled: bool = True
    ocr_lang: str = "chi_sim+eng"
    ocr_min_confidence: float = 0.4
    rules_enabled: bool = True
    # Compact context for the model (Phase D3)
    context_max_elements: int = 32
    annotate_boxes: bool = True


class AgentSection(BaseModel):
    max_steps: int = 8
    default_risk_auto: bool = False
    plan_timeout_sec: float = 30.0
    backend: Literal["mock"] = "mock"


class ActuatorSection(BaseModel):
    backend: Literal["mock"] = "mock"
    dry_run: bool = True
    max_actions_per_minute: int = 30


class SafetySection(BaseModel):
    default_mode: Literal["read_only", "confirm_all", "allow_low"] = "read_only"
    require_confirmation_below: Literal["low", "medium", "high"] = "medium"
    block_high_risk: bool = True
    sensitive_keywords: list[str] = Field(default_factory=list)
    audit_enabled: bool = True


class FrontendSection(BaseModel):
    mode: Literal["cli"] = "cli"
    show_trace: bool = True


class AppConfig(BaseModel):
    schema_version: str = "1.0.0"
    app: AppSection = Field(default_factory=AppSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    inference: InferenceSection = Field(default_factory=InferenceSection)
    capture: CaptureSection = Field(default_factory=CaptureSection)
    ui_vision: UIVisionSection = Field(default_factory=UIVisionSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    actuator: ActuatorSection = Field(default_factory=ActuatorSection)
    safety: SafetySection = Field(default_factory=SafetySection)
    frontend: FrontendSection = Field(default_factory=FrontendSection)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not v:
            raise ValueError("schema_version required")
        return v

    @property
    def project_root(self) -> Path:
        return self.paths.project_root or PROJECT_ROOT


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Lightweight env overrides for common knobs."""
    mapping = {
        "BAODOU_LOG_LEVEL": ("app", "log_level"),
        "BAODOU_INFERENCE": ("inference", "backend"),
        "BAODOU_CAPTURE": ("capture", "backend"),
        "BAODOU_CAPTURE_MODE": ("capture", "mode"),
        "BAODOU_MONITOR_INDEX": ("capture", "monitor_index"),
        "BAODOU_LLAMA_HOST": ("inference", "host"),
        "BAODOU_LLAMA_PORT": ("inference", "port"),
        "BAODOU_N_CTX": ("inference", "n_ctx"),
        "BAODOU_N_GPU_LAYERS": ("inference", "n_gpu_layers"),
        "BAODOU_DEVICE": ("inference", "device"),
        "BAODOU_UI_VISION": ("ui_vision", "backend"),
    }
    for env_key, path in mapping.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        section, key = path
        bucket = data.setdefault(section, {})
        if key in ("port", "n_ctx", "n_gpu_layers", "monitor_index"):
            bucket[key] = int(raw)
        else:
            bucket[key] = raw
    return data


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config, merge env overrides, resolve paths."""
    cfg_path = Path(path) if path else Path(os.environ.get("BAODOU_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise ConfigError(f"配置文件不存在: {cfg_path}", details={"path": str(cfg_path)})

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败: {exc}", cause=exc) from exc

    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须是 mapping")

    raw = _apply_env_overrides(raw)
    try:
        cfg = AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"配置校验失败: {exc}", cause=exc) from exc

    root = cfg.paths.project_root or PROJECT_ROOT
    cfg.paths = cfg.paths.resolve(root)
    return cfg


def config_to_safe_dict(cfg: AppConfig) -> dict[str, Any]:
    """Dict suitable for logging (no secrets today; keep hook)."""
    return cfg.model_dump(mode="json")
