"""Typed error taxonomy for the whole pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # Infrastructure
    CONFIG_INVALID = "config_invalid"
    CANCELLED = "cancelled"
    INTERNAL = "internal"
    NOT_IMPLEMENTED = "not_implemented"

    # Model / inference
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_TIMEOUT = "inference_timeout"
    INFERENCE_FAILED = "inference_failed"
    OUTPUT_PARSE_FAILED = "output_parse_failed"
    OUTPUT_SCHEMA_INVALID = "output_schema_invalid"

    # Capture
    CAPTURE_FAILED = "capture_failed"
    CAPTURE_TIMEOUT = "capture_timeout"
    MONITOR_NOT_FOUND = "monitor_not_found"

    # UI vision
    VISION_TIMEOUT = "vision_timeout"
    VISION_FAILED = "vision_failed"
    ELEMENT_NOT_FOUND = "element_not_found"
    TARGET_STALE = "target_stale"

    # Agent / safety / actuator
    PERMISSION_DENIED = "permission_denied"
    CONFIRMATION_REQUIRED = "confirmation_required"
    RISK_BLOCKED = "risk_blocked"
    ACTION_FAILED = "action_failed"
    ACTION_TIMEOUT = "action_timeout"
    VERIFICATION_FAILED = "verification_failed"

    # Target validity
    TARGET_INVALID = "target_invalid"
    COORDINATE_OUT_OF_BOUNDS = "coordinate_out_of_bounds"


# Human-readable default messages (Chinese + stable code).
_DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.CONFIG_INVALID: "配置无效",
    ErrorCode.CANCELLED: "操作已取消",
    ErrorCode.INTERNAL: "内部错误",
    ErrorCode.NOT_IMPLEMENTED: "功能尚未实现",
    ErrorCode.MODEL_LOAD_FAILED: "模型加载失败",
    ErrorCode.MODEL_UNAVAILABLE: "推理服务不可用",
    ErrorCode.INFERENCE_TIMEOUT: "模型推理超时",
    ErrorCode.INFERENCE_FAILED: "模型推理失败",
    ErrorCode.OUTPUT_PARSE_FAILED: "模型输出解析失败",
    ErrorCode.OUTPUT_SCHEMA_INVALID: "模型输出不符合协议",
    ErrorCode.CAPTURE_FAILED: "截图失败",
    ErrorCode.CAPTURE_TIMEOUT: "截图超时",
    ErrorCode.MONITOR_NOT_FOUND: "显示器未找到",
    ErrorCode.VISION_TIMEOUT: "UI 识别超时",
    ErrorCode.VISION_FAILED: "UI 识别失败",
    ErrorCode.ELEMENT_NOT_FOUND: "目标元素未找到",
    ErrorCode.TARGET_STALE: "目标元素已失效",
    ErrorCode.PERMISSION_DENIED: "权限不足",
    ErrorCode.CONFIRMATION_REQUIRED: "需要用户确认",
    ErrorCode.RISK_BLOCKED: "高风险操作被拦截",
    ErrorCode.ACTION_FAILED: "操作执行失败",
    ErrorCode.ACTION_TIMEOUT: "操作执行超时",
    ErrorCode.VERIFICATION_FAILED: "动作后验证失败",
    ErrorCode.TARGET_INVALID: "目标无效",
    ErrorCode.COORDINATE_OUT_OF_BOUNDS: "坐标越界",
}


class BaodouError(Exception):
    """Base application error with stable code for logs and UI."""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.message = message or _DEFAULT_MESSAGES.get(code, code.value)
        self.details = details or {}
        self.cause = cause
        super().__init__(f"[{code.value}] {self.message}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "error_code": self.code.value,
            "error_message": self.message,
        }
        if self.details:
            out["details"] = self.details
        if self.cause is not None:
            out["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return out


class ConfigError(BaodouError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(ErrorCode.CONFIG_INVALID, message, **kwargs)


class CancelledError(BaodouError):
    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        super().__init__(ErrorCode.CANCELLED, message, **kwargs)


class CaptureError(BaodouError):
    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode = ErrorCode.CAPTURE_FAILED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, **kwargs)


class VisionError(BaodouError):
    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode = ErrorCode.VISION_FAILED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, **kwargs)


class InferenceError(BaodouError):
    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode = ErrorCode.INFERENCE_FAILED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, **kwargs)


class SafetyError(BaodouError):
    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode = ErrorCode.PERMISSION_DENIED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, **kwargs)


class ActuatorError(BaodouError):
    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode = ErrorCode.ACTION_FAILED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, **kwargs)


class VerificationError(BaodouError):
    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        super().__init__(ErrorCode.VERIFICATION_FAILED, message, **kwargs)
