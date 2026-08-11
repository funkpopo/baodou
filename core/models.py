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
    """Protocol risk for action steps (model-facing: low|medium|high)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskCategory(StrEnum):
    """Phase G operational risk buckets (policy-facing)."""

    OBSERVE = "observe"  # read-only observation, no OS input
    LOW = "low"  # low-risk interaction (click/move/scroll)
    MEDIUM = "medium"  # data modification (type/drag)
    HIGH = "high"  # external submit / irreversible


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
    # Drag end / secondary point (physical screen pixels).
    end_point: Point | None = None
    text: str | None = None
    keys: list[str] = Field(default_factory=list)
    # Scroll deltas (positive dy = scroll up on most Windows apps via wheel).
    scroll_dx: int = 0
    scroll_dy: int = 0
    wait_ms: int = 0
    risk: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = True
    # Optional steps may be skipped on recovery instead of failing the task.
    optional: bool = False
    # Explicit opt-in for bare coordinates (still needs secondary confirm by policy).
    allow_coordinate_fallback: bool = False
    preconditions: list[str] = Field(default_factory=list)
    expected_change: str = ""
    timeout_ms: int = 5000
    # Human-readable label for preview UI.
    description: str = ""


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
    # Resolved physical click/type point actually used (after relocate).
    resolved_point: Point | None = None
    resolved_element_id: str | None = None
    relocated: bool = False
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
            "resolved_element_id": self.resolved_element_id,
            "relocated": self.relocated,
            "resolved_point": self.resolved_point.model_dump() if self.resolved_point else None,
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
    # Optional signals used by agent recovery.
    change_score: float | None = None
    target_still_present: bool | None = None
    frame_id_before: str | None = None
    frame_id_after: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)

    def log_summary(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "passed": self.passed,
            "message": self.message[:120],
            "change_score": self.change_score,
            "target_still_present": self.target_still_present,
        }


class ActionPreview(BaseModel):
    """User-visible preview: what will run, where, and expected impact."""

    protocol_version: str = PROTOCOL_VERSION
    preview_id: str = Field(default_factory=lambda: _new_id("prev"))
    step_id: str
    action: ActionType
    summary: str
    target_element_id: str | None = None
    target_text: str = ""
    target_type: str = ""
    target_bbox: BBox | None = None
    target_point: Point | None = None
    risk: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = True
    expected_impact: str = ""
    preconditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    uses_coordinates: bool = False

    def log_summary(self) -> dict[str, Any]:
        return {
            "preview_id": self.preview_id,
            "step_id": self.step_id,
            "action": self.action.value,
            "summary": self.summary[:120],
            "target_element_id": self.target_element_id,
            "risk": self.risk.value,
            "requires_confirmation": self.requires_confirmation,
            "uses_coordinates": self.uses_coordinates,
            "warnings": self.warnings[:5],
        }


class RecoveryAction(StrEnum):
    NONE = "none"
    REIDENTIFY = "reidentify"
    RETRY_STEP = "retry_step"
    SKIP_STEP = "skip_step"
    GO_BACK = "go_back"
    PAUSE = "pause"
    FAIL = "fail"


class StepRecord(BaseModel):
    """One executed (or attempted) step with full audit trail."""

    protocol_version: str = PROTOCOL_VERSION
    step_id: str
    index: int
    action: ActionType
    state: str = "pending"  # pending|preview|confirmed|executed|verified|skipped|failed|paused
    preview: ActionPreview | None = None
    safety: SafetyDecision | None = None
    action_result: ActionResult | None = None
    verification: VerificationResult | None = None
    recovery: RecoveryAction = RecoveryAction.NONE
    recovery_note: str = ""
    frame_id_before: str | None = None
    frame_id_after: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class CorrectionKind(StrEnum):
    """User target corrections for the current task (Phase H)."""

    REJECT_ELEMENT = "reject_element"  # "不是这个按钮"
    PREFER_ELEMENT = "prefer_element"  # "点击这个"
    CLICK_HERE = "click_here"  # explicit point
    IGNORE_REGION = "ignore_region"  # mask / skip area
    NOTE = "note"  # free-form guidance for planner context


class UserCorrection(BaseModel):
    """One user correction applied as task context (not free-form OS input)."""

    protocol_version: str = PROTOCOL_VERSION
    correction_id: str = Field(default_factory=lambda: _new_id("corr"))
    kind: CorrectionKind
    element_id: str | None = None
    point: Point | None = None
    region: BBox | None = None
    note: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)

    def log_summary(self) -> dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "kind": self.kind.value,
            "element_id": self.element_id,
            "point": self.point.model_dump() if self.point else None,
            "region": self.region.model_dump() if self.region else None,
            "note": self.note[:80],
        }


class ActivityPhase(StrEnum):
    """Visible privacy / activity indicator for the UI (Phase H)."""

    IDLE = "idle"
    CAPTURING = "capturing"
    RECOGNIZING = "recognizing"
    INFERRING = "inferring"
    AWAITING_CONFIRM = "awaiting_confirm"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MetricsSnapshot(BaseModel):
    """Latency + resource snapshot for the observability panel."""

    protocol_version: str = PROTOCOL_VERSION
    timestamp: datetime = Field(default_factory=_utcnow)
    capture_latency_ms: float | None = None
    vision_latency_ms: float | None = None
    model_latency_ms: float | None = None
    end_to_end_ms: float | None = None
    queue_length: int = 0
    queue_dropped: int = 0
    cpu_percent: float | None = None
    memory_rss_mb: float | None = None
    memory_percent: float | None = None
    gpu_name: str = ""
    gpu_util_percent: float | None = None
    gpu_mem_mb: float | None = None
    recent_errors: list[str] = Field(default_factory=list)

    def log_summary(self) -> dict[str, Any]:
        return {
            "capture_ms": self.capture_latency_ms,
            "vision_ms": self.vision_latency_ms,
            "model_ms": self.model_latency_ms,
            "e2e_ms": self.end_to_end_ms,
            "queue": self.queue_length,
            "cpu": self.cpu_percent,
            "rss_mb": self.memory_rss_mb,
            "gpu": self.gpu_name or None,
            "errors": self.recent_errors[:5],
        }


class ActivityStatus(BaseModel):
    """What the assistant is doing right now (user-visible privacy cue)."""

    protocol_version: str = PROTOCOL_VERSION
    phase: ActivityPhase = ActivityPhase.IDLE
    capturing: bool = False
    recognizing: bool = False
    inferring: bool = False
    about_to_act: bool = False
    executing: bool = False
    message: str = ""
    control_state: str = "running"  # running|paused|emergency_stop

    def log_summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "capturing": self.capturing,
            "recognizing": self.recognizing,
            "inferring": self.inferring,
            "about_to_act": self.about_to_act,
            "executing": self.executing,
            "message": self.message[:120],
            "control_state": self.control_state,
        }


class TaskContext(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    task_id: str = Field(default_factory=lambda: _new_id("task"))
    trace_id: str = ""
    user_goal: str
    state: TaskState = TaskState.IDLE
    frame_id: str | None = None
    plan_id: str | None = None
    step_index: int = 0
    steps_done: int = 0
    steps_skipped: int = 0
    steps_failed: int = 0
    recovery_attempts: int = 0
    pause_reason: str = ""
    last_error: str = ""
    confirmed: bool = False
    auto_confirmed: bool = False
    step_records: list[StepRecord] = Field(default_factory=list)
    # Phase H: user corrections become structured task context
    corrections: list[UserCorrection] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self, state: TaskState | None = None) -> None:
        if state is not None:
            self.state = state
        self.updated_at = _utcnow()

    def log_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "user_goal": self.user_goal[:80],
            "state": self.state.value,
            "plan_id": self.plan_id,
            "frame_id": self.frame_id,
            "step_index": self.step_index,
            "steps_done": self.steps_done,
            "steps_skipped": self.steps_skipped,
            "steps_failed": self.steps_failed,
            "recovery_attempts": self.recovery_attempts,
            "pause_reason": self.pause_reason[:120] if self.pause_reason else "",
            "confirmed": self.confirmed,
            "auto_confirmed": self.auto_confirmed,
            "correction_count": len(self.corrections),
        }


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
    # Phase G extensions
    category: RiskCategory = RiskCategory.LOW
    rules_hit: list[str] = Field(default_factory=list)
    auto_executable: bool = False
    confirmed_by_user: bool = False

    def log_summary(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_confirmation": self.requires_confirmation,
            "risk": self.risk.value,
            "category": self.category.value,
            "reason": self.reason[:120],
            "blocked_by": self.blocked_by,
            "rules_hit": self.rules_hit[:8],
            "auto_executable": self.auto_executable,
        }


class AuditEventKind(StrEnum):
    TASK_START = "task_start"
    TASK_END = "task_end"
    OBSERVE = "observe"
    PLAN = "plan"
    CONFIRM = "confirm"
    SAFETY = "safety"
    ACTION = "action"
    VERIFY = "verify"
    PAUSE = "pause"
    EMERGENCY_STOP = "emergency_stop"
    REDACT = "redact"
    CLEANUP = "cleanup"
    THREAT = "threat"


class AuditRecord(BaseModel):
    """Non-repudiable local audit entry (Phase G). No raw pixels / secrets."""

    protocol_version: str = PROTOCOL_VERSION
    audit_id: str = Field(default_factory=lambda: _new_id("audit"))
    kind: AuditEventKind
    trace_id: str = ""
    task_id: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
    summary: str = ""
    # Structured but redacted payload
    payload: dict[str, Any] = Field(default_factory=dict)
    model_version: str = ""
    prompt_version: str = ""

    def log_summary(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "kind": self.kind.value,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "summary": self.summary[:160],
            "model_version": self.model_version,
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


# Fix forward refs (plan / safety defined later in file for some models)
InferenceResponse.model_rebuild()
StepRecord.model_rebuild()
