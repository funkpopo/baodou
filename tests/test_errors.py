"""Error taxonomy tests."""

from __future__ import annotations

from core.errors import BaodouError, CancelledError, ErrorCode, InferenceError


def test_error_to_dict() -> None:
    err = BaodouError(ErrorCode.CAPTURE_FAILED, "boom", details={"monitor": 0})
    d = err.to_dict()
    assert d["error_code"] == "capture_failed"
    assert d["error_message"] == "boom"
    assert d["details"]["monitor"] == 0


def test_cancelled_error_code() -> None:
    err = CancelledError()
    assert err.code == ErrorCode.CANCELLED
    assert "取消" in str(err) or err.code.value in str(err)


def test_inference_error_default() -> None:
    err = InferenceError()
    assert err.code == ErrorCode.INFERENCE_FAILED
