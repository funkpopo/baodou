"""Unified protocol models (versioned). Modules must exchange these, not free text."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# Bump when wire format changes incompatibly.
PROTOCOL_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ElementType(StrEnum):
    WINDOW = "window"
    DIALOG = "dialog"
    BUTTON = "button"
    INPUT = "input"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TAB = "tab"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    LINK = "link"
    LIST = "list"
    TABLE = "table"
    ICON = "icon"
    TEXT = "text"
    IMAGE = "image"
    OTHER = "other"


class ActionType(StrEnum):
    NONE = "none"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MOVE = "move"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    HOTKEY = "hotkey"
    WAIT = "wait"
    REIDENTIFY = "reidentify"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskState(StrEnum):
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class CaptureMode(StrEnum):
    PRIMARY = "primary"
    ALL = "all"
    WINDOW = "window"
    REGION = "region"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned box in screen physical pixels."""

    x: int
    y: int
    width: int
    height: int

    @field_validator("width", "height")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("width/height must be >= 0")
        return v

    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


class Point(BaseModel):
    x: int
    y: int


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class ScreenFrame(BaseModel):
    """One captured screen frame with metadata (no raw pixels in logs)."""

    protocol_version: str = PROTOCOL_VERSION
    frame_id: str = Field(default_factory=lambda: _new_id("frame"))
    trace_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
    mode: CaptureMode = CaptureMode.PRIMARY
    monitor_index: int = 0
    width: int
    height: int
    dpi_scale: float = 1.0
    image_format: str = "png"
    # Optional path or base64; keep out of structured logs by convention.
    image_path: str | None = None
    image_b64: str | None = None
    capture_ms: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def log_summary(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "trace_id": self.trace_id,
            "width": self.width,
            "height": self.height,
            "dpi_scale": self.dpi_scale,
            "monitor_index": self.monitor_index,
            "mode": self.mode.value,
            "capture_ms": self.capture_ms,
            "has_image": bool(self.image_path or self.image_b64),
        }


# ---------------------------------------------------------------------------
# UI Vision
# ---------------------------------------------------------------------------


class UIElement(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    element_id: str
    type: ElementType = ElementType.OTHER
    role: str = ""
    text: str = ""
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    visible: bool = True
    enabled: bool = True
    clickable: bool = False
    editable: bool = False
    source: list[str] = Field(default_factory=list)
    frame_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)

    @property
    def center(self) -> Point:
        cx, cy = self.bbox.center()
        return Point(x=cx, y=cy)


class UIVisionResult(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    result_id: str = Field(default_factory=lambda: _new_id("vision"))
    trace_id: str = ""
    frame_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    elements: list[UIElement] = Field(default_factory=list)
    latency_ms: float | None = None
    source: str = "mock"
    notes: str = ""

    def log_summary(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "trace_id": self.trace_id,
            "frame_id": self.frame_id,
            "element_count": len(self.elements),
            "latency_ms": self.latency_ms,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Inference / Observation
# ---------------------------------------------------------------------------


class ScreenObservation(BaseModel):
    """Model (or mock) screen understanding — structured only."""

    protocol_version: str = PROTOCOL_VERSION
    observation_id: str = Field(default_factory=lambda: _new_id("obs"))
    trace_id: str = ""
    frame_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
    observation: str = ""
    ui_elements: list[UIElement] = Field(default_factory=list)
    notes: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    model_name: str = ""
    latency_ms: float | None = None
    raw_truncated: bool = False  # true if output was cut off

    def log_summary(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "trace_id": self.trace_id,
            "frame_id": self.frame_id,
            "element_count": len(self.ui_elements),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "raw_truncated": self.raw_truncated,
            "observation_preview": (self.observation[:120] + "…")
            if len(self.observation) > 120
            else self.observation,
        }


class InferenceRequest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str = Field(default_factory=lambda: _new_id("llm"))
    trace_id: str = ""
    frame_id: str = ""
    task: str
    system_prompt: str = ""
    ui_summary: list[dict[str, Any]] = Field(default_factory=list)
    has_image: bool = False
    max_tokens: int = 768
    temperature: float = 0.3


class InferenceResponse(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    trace_id: str = ""
    ok: bool = True
    observation: ScreenObservation | None = None
    plan: ActionPlan | None = None
    raw_text: str = ""
    latency_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None

    def log_summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "has_observation": self.observation is not None,
            "has_plan": self.plan is not None,
        }


# ---------------------------------------------------------------------------
# Agent / Actions
# ---------------------------------------------------------------------------


class ActionStep(BaseModel):
    step_id: str = Field(default_factory=lambda: _new_id("step"))
    action: ActionType = ActionType.NONE
    target_element_id: str | None = None
    # Coordinates only as last resort; prefer element_id.
    target_point: Point | None = None
    text: str | None = None
    keys: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = True
    preconditions: list[str] = Field(default_factory=list)
    expected_change: str = ""
    timeout_ms: int = 5000


class ActionPlan(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    plan_id: str = Field(default_factory=lambda: _new_id("plan"))
    trace_id: str = ""
    goal: str
    steps: list[ActionStep] = Field(default_factory=list)
    stop_if: list[str] = Field(default_factory=list)
    risk_max: RiskLevel = RiskLevel.LOW
    created_at: datetime = Field(default_factory=_utcnow)

    def log_summary(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "trace_id": self.trace_id,
            "goal": self.goal[:80],
            "step_count": len(self.steps),
            "risk_max": self.risk_max.value,
            "actions": [s.action.value for s in self.steps],
        }


class ActionResult(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    result_id: str = Field(default_factory=lambda: _new_id("act"))
    trace_id: str = ""
    step_id: str
    action: ActionType
    success: bool
    dry_run: bool = True
    message: str = ""
    latency_ms: float | None = None
    timestamp: datetime = Field(default_factory=_utcnow)

    def log_summary(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "action": self.action.value,
            "success": self.success,
            "dry_run": self.dry_run,
            "latency_ms": self.latency_ms,
        }


class VerificationResult(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    verification_id: str = Field(default_factory=lambda: _new_id("ver"))
    trace_id: str = ""
    step_id: str
    passed: bool
    expected: str = ""
    actual: str = ""
    message: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)

    def log_summary(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "passed": self.passed,
            "message": self.message[:120],
        }


class TaskContext(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    task_id: str = Field(default_factory=lambda: _new_id("task"))
    trace_id: str = ""
    user_goal: str
    state: TaskState = TaskState.IDLE
    frame_id: str | None = None
    plan_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self, state: TaskState | None = None) -> None:
        if state is not None:
            self.state = state
        self.updated_at = _utcnow()


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class SafetyDecision(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    allowed: bool
    requires_confirmation: bool = True
    risk: RiskLevel = RiskLevel.LOW
    reason: str = ""
    blocked_by: str | None = None

    def log_summary(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_confirmation": self.requires_confirmation,
            "risk": self.risk.value,
            "reason": self.reason[:120],
            "blocked_by": self.blocked_by,
        }


# ---------------------------------------------------------------------------
# Pipeline envelope (trace-correlated chain)
# ---------------------------------------------------------------------------


class PipelineEventKind(StrEnum):
    FRAME = "frame"
    VISION = "vision"
    INFERENCE = "inference"
    PLAN = "plan"
    SAFETY = "safety"
    ACTION = "action"
    VERIFICATION = "verification"
    ERROR = "error"
    CANCEL = "cancel"
    DONE = "done"


class PipelineEvent(BaseModel):
    """One hop in the capture → … → verify chain; always carries trace_id."""

    protocol_version: str = PROTOCOL_VERSION
    event_id: str = Field(default_factory=lambda: _new_id("evt"))
    trace_id: str
    kind: PipelineEventKind
    timestamp: datetime = Field(default_factory=_utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


# Fix forward ref for InferenceResponse.plan
InferenceResponse.model_rebuild()
