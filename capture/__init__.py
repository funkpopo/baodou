"""Screen capture module (Phase C: real pipeline + geometry)."""

from capture.base import CaptureBackend
from capture.factory import create_capture
from capture.frame import FramePacket
from capture.mock import MockCapture
from capture.pipeline import CapturePipeline

__all__ = [
    "CaptureBackend",
    "CapturePipeline",
    "FramePacket",
    "MockCapture",
    "create_capture",
]
