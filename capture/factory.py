"""Capture backend factory."""

from __future__ import annotations

from core.config import AppConfig

from capture.base import CaptureBackend


def create_capture(config: AppConfig) -> CaptureBackend:
    backend = (config.capture.backend or "mss").lower()
    if backend == "mock":
        from capture.mock import MockCapture

        return MockCapture(config)
    if backend == "mss":
        from capture.mss_backend import MssCapture

        return MssCapture(config)
    raise ValueError(f"unknown capture backend: {backend}")
