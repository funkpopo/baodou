"""UI element recognition framework (Phase D)."""

from ui_vision.base import UIRecognizer, UIVisionBackend
from ui_vision.factory import create_ui_vision
from ui_vision.mock import MockUIVision

__all__ = [
    "MockUIVision",
    "UIRecognizer",
    "UIVisionBackend",
    "create_ui_vision",
]
