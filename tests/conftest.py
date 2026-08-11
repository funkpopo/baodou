"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cancel import get_global_token  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging import setup_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cancel_token() -> None:
    tok = get_global_token()
    tok.reset()
    yield
    tok.reset()


@pytest.fixture
def config():
    setup_logging(level="WARNING", json_logs=False, log_dir=None)
    return load_config()
