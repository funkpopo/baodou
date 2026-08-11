"""Cancellation token tests."""

from __future__ import annotations

import threading

import pytest
from core.cancel import CancellationToken
from core.errors import CancelledError, ErrorCode


def test_cancel_and_check() -> None:
    tok = CancellationToken("t")
    tok.check()  # no raise
    tok.cancel("stop")
    assert tok.is_cancelled
    assert tok.reason == "stop"
    with pytest.raises(CancelledError) as ei:
        tok.check()
    assert ei.value.code == ErrorCode.CANCELLED


def test_on_cancel_callback() -> None:
    tok = CancellationToken("cb")
    seen: list[str] = []
    tok.on_cancel(lambda: seen.append("fired"))
    tok.cancel("x")
    assert seen == ["fired"]
    # Late registration fires immediately if already cancelled.
    tok.on_cancel(lambda: seen.append("late"))
    assert seen == ["fired", "late"]


def test_interruptible_sleep() -> None:
    tok = CancellationToken("sleep")

    def _cancel_soon() -> None:
        tok.cancel("from_thread")

    threading.Timer(0.05, _cancel_soon).start()
    with pytest.raises(CancelledError):
        tok.sleep(2.0, slice_sec=0.01)


def test_reset() -> None:
    tok = CancellationToken("r")
    tok.cancel("a")
    tok.reset()
    assert not tok.is_cancelled
    tok.check()
