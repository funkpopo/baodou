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


class FrameKind(StrEnum):
    """Downstream use of a frame — frequency/quality differ per kind."""

    RAW = "raw"
    PREVIEW = "preview"
    VISION = "vision"
    MODEL = "model"
    VERIFY = "verify"


class DisplayOrientation(StrEnum):
    ROT0 = "0"
    ROT90 = "90"
    ROT180 = "180"
    ROT270 = "270"


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

    def clamp(self, bounds: BBox) -> BBox:
        x1 = max(self.x, bounds.x)
        y1 = max(self.y, bounds.y)
        x2 = min(self.x + self.width, bounds.x + bounds.width)
        y2 = min(self.y + self.height, bounds.y + bounds.height)
        return BBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))


class Point(BaseModel):
    x: int
    y: int


class MonitorInfo(BaseModel):
    """One physical (or virtual-all) display in virtual-desktop physical pixels."""

    index: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool = False
    name: str = ""
    dpi_x: int = 96
    dpi_y: int = 96
    dpi_scale: float = 1.0
    orientation: DisplayOrientation = DisplayOrientation.ROT0

    @property
    def bbox(self) -> BBox:
        return BBox(x=self.left, y=self.top, width=self.width, height=self.height)


class WindowInfo(BaseModel):
    hwnd: int | None = None
    title: str = ""
    class_name: str = ""
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    visible: bool = True

    @property
    def bbox(self) -> BBox:
        return BBox(x=self.left, y=self.top, width=self.width, height=self.height)


class SensitiveRegion(BaseModel):
    """Region to mask before leaving the capture layer (physical screen pixels)."""

    x: int
    y: int
    width: int
    height: int
    reason: str = "manual"  # manual | password | privacy_window
    absolute: bool = True

    def as_bbox(self) -> BBox:
        return BBox(x=self.x, y=self.y, width=self.width, height=self.height)


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
    frame_kind: FrameKind = FrameKind.RAW
    monitor_index: int = 0
    # Stored image size (after scale / ROI).
    width: int
    height: int
    # Source capture region in virtual-desktop physical pixels.
    origin_x: int = 0
    origin_y: int = 0
    physical_width: int | None = None
    physical_height: int | None = None
    # Logical (DIP) size of the source region.
    logical_width: float | None = None
    logical_height: float | None = None
    dpi_scale: float = 1.0
    dpi_x: int = 96
    dpi_y: int = 96
    orientation: DisplayOrientation = DisplayOrientation.ROT0
    # Map stored image pixel → physical screen pixel: phys = origin + img * scale
    scale_x: float = 1.0
    scale_y: float = 1.0
    image_format: str = "png"
    color_mode: str = "RGB"
    # Optional path or base64; keep out of structured logs by convention.
    image_path: str | None = None
    image_b64: str | None = None
    image_bytes: int | None = None
    capture_ms: float | None = None
    preprocess_ms: float | None = None
    changed: bool | None = None
    change_score: float | None = None
    pixel_hash: str | None = None
    window: WindowInfo | None = None
    monitor: MonitorInfo | None = None
    masked_regions: list[SensitiveRegion] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def log_summary(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "trace_id": self.trace_id,
            "frame_kind": self.frame_kind.value,
            "width": self.width,
            "height": self.height,
            "physical_width": self.physical_width,
            "physical_height": self.physical_height,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "dpi_scale": self.dpi_scale,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "monitor_index": self.monitor_index,
            "mode": self.mode.value,
            "capture_ms": self.capture_ms,
            "preprocess_ms": self.preprocess_ms,
            "changed": self.changed,
            "change_score": self.change_score,
            "image_bytes": self.image_bytes,
            "has_image": bool(self.image_path or self.image_b64),
            "masked_count": len(self.masked_regions),
            "window_title": self.window.title if self.window else None,
        }

    def image_to_screen(self, ix: float, iy: float) -> Point:
        """Convert coordinates in the stored image to virtual-desktop physical pixels."""
        return Point(
            x=int(round(self.origin_x + ix * self.scale_x)),
            y=int(round(self.origin_y + iy * self.scale_y)),
        )

    def screen_to_image(self, sx: float, sy: float) -> Point:
        """Convert virtual-desktop physical pixels to stored-image coordinates."""
        if self.scale_x == 0 or self.scale_y == 0:
            return Point(x=0, y=0)
        return Point(
            x=int(round((sx - self.origin_x) / self.scale_x)),
            y=int(round((sy - self.origin_y) / self.scale_y)),
        )


# ---------------------------------------------------------------------------
# UI Vision
# ---------------------------------------------------------------------------


class UIElement(BaseModel):
    """One recognized UI element.

    Coordinate contract (Phase D / multi-resolution):
    - ``bbox`` is always **virtual-desktop physical pixels** (same space as mouse).
    - ``bbox_logical`` is optional DIP/logical pixels when DPI scale is known.
    - ``dpi_scale`` / ``dpi_x`` / ``dpi_y`` document the host monitor scaling.
    - Image-space detections must be converted via ``ScreenFrame.image_to_screen``
      before becoming a ``UIElement``.
    """

    protocol_version: str = PROTOCOL_VERSION
    element_id: str
    type: ElementType = ElementType.OTHER
    role: str = ""
    text: str = ""
    name: str = ""
    bbox: BBox
    bbox_logical: BBox | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    visible: bool = True
    enabled: bool = True
    clickable: bool = False
    editable: bool = False
    source: list[str] = Field(default_factory=list)
    frame_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
    # Hierarchy / identity (Phase D)
    parent_id: str | None = None
    depth: int = 0
    z_order: int = 0
    content_hash: str = ""
    dpi_scale: float = 1.0
    dpi_x: int = 96
    dpi_y: int = 96
    needs_review: bool = False
    conflict: bool = False
    native_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def center(self) -> Point:
        cx, cy = self.bbox.center()
        return Point(x=cx, y=cy)

    def is_stale_for_frame(self, frame_id: str) -> bool:
        return bool(self.frame_id) and self.frame_id != frame_id

    def matches_hash(self, other: UIElement, *, iou_min: float = 0.5) -> bool:
        """Short-term identity: same content_hash or high IoU + type/text."""
        if self.content_hash and other.content_hash and self.content_hash == other.content_hash:
            return True
        if self.type != other.type:
            return False
        if (self.text or "").strip() != (other.text or "").strip():
            return False
        return bbox_iou(self.bbox, other.bbox) >= iou_min


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
    sources_used: list[str] = Field(default_factory=list)
    roi: BBox | None = None
    dpi_scale: float = 1.0
    review_count: int = 0

    def by_id(self, element_id: str) -> UIElement | None:
        return next((e for e in self.elements if e.element_id == element_id), None)

    def interactive(self) -> list[UIElement]:
        return [e for e in self.elements if e.clickable or e.editable]

    def log_summary(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "trace_id": self.trace_id,
            "frame_id": self.frame_id,
            "element_count": len(self.elements),
            "latency_ms": self.latency_ms,
            "source": self.source,
            "sources_used": self.sources_used,
            "review_count": self.review_count,
            "dpi_scale": self.dpi_scale,
        }


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union for two physical-pixel boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0:
        return 0.0
    return inter / union


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
