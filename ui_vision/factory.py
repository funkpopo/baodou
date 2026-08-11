"""UI vision backend factory."""

from __future__ import annotations

from core.config import AppConfig
from core.errors import ConfigError

from ui_vision.base import UIRecognizer, UIVisionBackend


def create_recognizer(name: str, config: AppConfig) -> UIRecognizer:
    key = (name or "").strip().lower()
    if key == "mock":
        from ui_vision.mock import MockRecognizer

        return MockRecognizer(config)
    if key == "uia":
        from ui_vision.uia import UiaRecognizer

        return UiaRecognizer(config)
    if key == "ocr":
        from ui_vision.ocr import OcrRecognizer

        return OcrRecognizer(config)
    if key == "rules":
        from ui_vision.rules import RulesRecognizer

        return RulesRecognizer(config)
    raise ConfigError(f"未知 UI 识别源: {name}", details={"source": name})


def create_ui_vision(config: AppConfig) -> UIVisionBackend:
    backend = (config.ui_vision.backend or "mock").lower()
    if backend == "mock":
        from ui_vision.mock import MockUIVision

        return MockUIVision(config)

    if backend in ("composite", "pipeline", "multi"):
        sources = list(config.ui_vision.sources or [])
        if not sources:
            sources = ["uia", "rules"]
        recognizers = [create_recognizer(s, config) for s in sources]
        from ui_vision.pipeline import CompositeUIVision

        return CompositeUIVision(config, recognizers)

    # Single-source backends for debugging
    if backend in ("uia", "ocr", "rules"):
        from ui_vision.pipeline import CompositeUIVision

        return CompositeUIVision(config, [create_recognizer(backend, config)])

    raise ConfigError(f"未知 ui_vision.backend: {backend}", details={"backend": backend})
