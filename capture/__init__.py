"""Screen capture module."""

from capture.base import CaptureBackend
from capture.mock import MockCapture

__all__ = ["CaptureBackend", "MockCapture"]
