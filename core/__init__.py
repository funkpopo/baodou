"""Shared infrastructure: models, config, logging, errors, cancel."""

from core.cancel import CancellationToken, get_global_token, install_signal_handlers
from core.config import AppConfig, load_config
from core.errors import BaodouError, ErrorCode
from core.logging import get_logger, new_trace_id, set_trace_id
from core.models import PROTOCOL_VERSION

__all__ = [
    "PROTOCOL_VERSION",
    "AppConfig",
    "BaodouError",
    "CancellationToken",
    "ErrorCode",
    "get_global_token",
    "get_logger",
    "install_signal_handlers",
    "load_config",
    "new_trace_id",
    "set_trace_id",
]

__version__ = "0.1.0"
