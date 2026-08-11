"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from core.config import load_config
from core.errors import ConfigError, ErrorCode


def test_load_default_config() -> None:
    cfg = load_config()
    assert cfg.schema_version == "1.0.0"
    assert cfg.inference.device == "SYCL0"
    assert cfg.inference.n_gpu_layers == 99
    assert cfg.actuator.dry_run is True
    assert cfg.paths.project_root is not None
    assert cfg.paths.model_gguf.is_absolute()


def test_env_override_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAODOU_LOG_LEVEL", "DEBUG")
    cfg = load_config()
    assert cfg.app.log_level == "DEBUG"


def test_env_override_inference_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAODOU_INFERENCE", "http")
    cfg = load_config()
    assert cfg.inference.backend == "http"


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as ei:
        load_config(tmp_path / "nope.yaml")
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("app: [\n  broken", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_custom_config_merge(tmp_path: Path) -> None:
    # Minimal override file based on defaults structure.
    data = {
        "schema_version": "1.0.0",
        "app": {"log_level": "WARNING", "log_json": False, "log_dir": ""},
        "inference": {"backend": "mock", "port": 9999},
        "capture": {"backend": "mock", "target_fps": 3},
        "ui_vision": {"backend": "mock"},
        "agent": {"backend": "mock"},
        "actuator": {"backend": "mock", "dry_run": True},
        "safety": {"default_mode": "read_only"},
        "frontend": {"mode": "cli"},
        "paths": {
            "model_gguf": "model/Qwen3.5-2B-UD-Q4_K_XL.gguf",
            "mmproj_gguf": "model/mmproj-F16.gguf",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    # project_root resolution uses core.PROJECT_ROOT for relative model paths.
    cfg = load_config(path)
    assert cfg.app.log_level == "WARNING"
    assert cfg.inference.port == 9999
