"""Headless UI session controller (Phase H).

Owns task lifecycle for the main window: start / pause / stop / confirm /
correct / metrics / diagnostics. GUI and CLI both drive this layer so logic
stays unit-testable without Tk.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.runtime import TaskAgent, TaskRunResult
from core.cancel import CancellationToken, get_global_token
from core.config import AppConfig
from core.errors import BaodouError, ErrorCode
from core.logging import get_logger, log_event, new_trace_id
from core.models import (
    ActionPreview,
    ActionStep,
    ActivityPhase,
    ActivityStatus,
    BBox,
    FrameKind,
    MetricsSnapshot,
    SafetyDecision,
    ScreenFrame,
    TaskContext,
    TaskState,
    UIElement,
    UIVisionResult,
    UserCorrection,
)
from safety.control import SafetyControl, get_safety_control

from frontend.corrections import (
    CorrectionStore,
    apply_corrections_to_goal,
    apply_corrections_to_vision,
)
from frontend.diagnostics import DiagnosticBundle, build_diagnostics
from frontend.metrics import MetricsCollector

_log = get_logger("frontend.session")

StateListener = Callable[[dict[str, Any]], None]


@dataclass
class SessionSnapshot:
    """Immutable-ish view of UI session state for rendering."""

    activity: ActivityStatus
    metrics: MetricsSnapshot
    task: TaskContext | None = None
    goal: str = ""
    observation_text: str = ""
    elements: list[dict[str, Any]] = field(default_factory=list)
    plan_summary: dict[str, Any] | None = None
    previews: list[dict[str, Any]] = field(default_factory=list)
    pending_preview: dict[str, Any] | None = None
    pending_step: dict[str, Any] | None = None
    pending_safety: dict[str, Any] | None = None
    corrections: list[dict[str, Any]] = field(default_factory=list)
    last_result_ok: bool | None = None
    last_error: str = ""
    busy: bool = False
    dry_run: bool = True
    mock: bool = True
    control_state: str = "running"
    highlight_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity.log_summary(),
            "metrics": self.metrics.log_summary(),
            "task": self.task.log_summary() if self.task else None,
            "goal": self.goal,
            "observation_text": self.observation_text[:400],
            "elements": self.elements[:48],
            "plan_summary": self.plan_summary,
            "previews": self.previews,
            "pending_preview": self.pending_preview,
            "pending_step": self.pending_step,
            "pending_safety": self.pending_safety,
            "corrections": self.corrections,
            "last_result_ok": self.last_result_ok,
            "last_error": self.last_error,
            "busy": self.busy,
            "dry_run": self.dry_run,
            "mock": self.mock,
            "control_state": self.control_state,
            "highlight_ids": self.highlight_ids,
        }


class UISession:
    """Application session used by the main window and headless CLI."""

    def __init__(
        self,
        config: AppConfig,
        *,
        mock: bool | None = None,
        token: CancellationToken | None = None,
        control: SafetyControl | None = None,
        listener: StateListener | None = None,
    ) -> None:
        self.config = config
        self.token = token or get_global_token()
        self.control = control or get_safety_control()
        self.listener = listener
        self.mock = bool(config.frontend.default_mock) if mock is None else bool(mock)
        if self.mock:
            self._force_mock_backends()
        self.metrics = MetricsCollector(recent_errors_max=config.frontend.recent_errors_max)
        self.corrections = CorrectionStore()
        self._lock = threading.RLock()
        self._busy = False
        self._goal = ""
        self._task: TaskContext | None = None
        self._last_result: TaskRunResult | None = None
        self._last_frame: ScreenFrame | None = None
        self._last_image: Any = None  # PIL.Image when available
        self._last_vision: UIVisionResult | None = None
        self._activity = ActivityStatus(phase=ActivityPhase.IDLE, message="空闲")
        self._pending_preview: ActionPreview | None = None
        self._pending_step: ActionStep | None = None
        self._pending_decision: SafetyDecision | None = None
        self._confirm_event = threading.Event()
        self._confirm_answer = False
        self._confirm_waiting = False
        self._worker: threading.Thread | None = None
        self._prompt_version = ""
        self._model_version = str(getattr(config.inference, "llama_build", "") or "")
        try:
            from inference.prompts import PROMPT_VERSION

            self._prompt_version = str(PROMPT_VERSION)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ setup
    def _force_mock_backends(self) -> None:
        self.config.capture.backend = "mock"
        self.config.ui_vision.backend = "mock"
        self.config.inference.backend = "mock"
        self.config.actuator.backend = "mock"
        self.config.agent.backend = "mock"
        self.config.actuator.dry_run = True

    def set_mock(self, mock: bool) -> None:
        with self._lock:
            self.mock = bool(mock)
            if self.mock:
                self._force_mock_backends()

    def set_listener(self, listener: StateListener | None) -> None:
        self.listener = listener

    def _notify(self) -> None:
        if self.listener is None:
            return
        try:
            self.listener(self.snapshot().to_dict())
        except Exception as exc:  # noqa: BLE001
            log_event(_log, "session.listener_error", error=str(exc)[:120])

    def _set_activity(
        self,
        phase: ActivityPhase,
        *,
        message: str = "",
        **flags: bool,
    ) -> None:
        ctrl = self.control.status().get("state", "running")
        st = ActivityStatus(
            phase=phase,
            capturing=flags.get("capturing", phase == ActivityPhase.CAPTURING),
            recognizing=flags.get("recognizing", phase == ActivityPhase.RECOGNIZING),
            inferring=flags.get("inferring", phase == ActivityPhase.INFERRING),
            about_to_act=flags.get("about_to_act", phase == ActivityPhase.AWAITING_CONFIRM),
            executing=flags.get(
                "executing",
                phase in (ActivityPhase.EXECUTING, ActivityPhase.VERIFYING),
            ),
            message=message or phase.value,
            control_state=str(ctrl),
        )
        with self._lock:
            self._activity = st
        self._notify()

    # ------------------------------------------------------------------ confirm bridge
    def _confirm_callback(
        self,
        preview: ActionPreview,
        step: ActionStep,
        decision: SafetyDecision,
    ) -> bool:
        """Block worker thread until UI confirms or rejects (or auto)."""
        with self._lock:
            self._pending_preview = preview
            self._pending_step = step
            self._pending_decision = decision
            self._confirm_waiting = True
            self._confirm_answer = False
            self._confirm_event.clear()
        self._set_activity(
            ActivityPhase.AWAITING_CONFIRM,
            message=f"待确认: {preview.summary}",
            about_to_act=True,
        )
        # Wait until confirm/reject or stop
        while not self._confirm_event.wait(timeout=0.25):
            if self.token.is_cancelled or self.control.is_stopped():
                with self._lock:
                    self._confirm_waiting = False
                return False
            if self.control.is_paused():
                continue
        with self._lock:
            self._confirm_waiting = False
            ans = self._confirm_answer
            self._pending_preview = None
            self._pending_step = None
            self._pending_decision = None
        return ans

    def confirm_pending(self, accept: bool = True) -> bool:
        """UI: accept or reject the pending action preview."""
        with self._lock:
            if not self._confirm_waiting:
                return False
            self._confirm_answer = bool(accept)
            self._confirm_event.set()
        return True

    def has_pending_confirmation(self) -> bool:
        with self._lock:
            return self._confirm_waiting

    # ------------------------------------------------------------------ public controls
    def pause(self, reason: str = "ui_pause") -> None:
        self.control.request_pause(reason=reason)
        self._set_activity(ActivityPhase.PAUSED, message=f"已暂停: {reason}")

    def resume(self, reason: str = "ui_resume") -> None:
        try:
            self.control.request_resume(reason=reason)
            self._set_activity(ActivityPhase.IDLE, message="已继续")
        except BaodouError as exc:
            self.metrics.push_error(exc.message)
            self._set_activity(ActivityPhase.STOPPED, message=exc.message)

    def emergency_stop(self, reason: str = "ui_emergency_stop") -> None:
        self.control.request_stop(reason=reason, token=self.token)
        # Unblock any waiting confirm
        with self._lock:
            self._confirm_answer = False
            self._confirm_event.set()
        self._set_activity(ActivityPhase.STOPPED, message=f"紧急停止: {reason}")

    def reset_stop(self, reason: str = "ui_reset") -> None:
        self.control.reset_stop(reason=reason)
        self.token.reset()
        self._set_activity(ActivityPhase.IDLE, message="已复位")

    # ------------------------------------------------------------------ corrections
    def reject_element(self, element_id: str, note: str = "") -> UserCorrection:
        c = self.corrections.reject_element(element_id, note=note)
        self._notify()
        return c

    def prefer_element(self, element_id: str, note: str = "") -> UserCorrection:
        c = self.corrections.prefer_element(element_id, note=note)
        self._notify()
        return c

    def click_here(self, x: int, y: int, note: str = "") -> UserCorrection:
        c = self.corrections.click_here(x, y, note=note)
        self._notify()
        return c

    def ignore_region(
        self, x: int, y: int, width: int, height: int, note: str = ""
    ) -> UserCorrection:
        c = self.corrections.ignore_region(x, y, width, height, note=note)
        self._notify()
        return c

    def add_note(self, text: str) -> UserCorrection:
        c = self.corrections.note(text)
        self._notify()
        return c

    def clear_corrections(self) -> None:
        self.corrections.clear()
        self._notify()

    # ------------------------------------------------------------------ observe / run
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def start_task(
        self,
        goal: str,
        *,
        execute: bool = True,
        auto_confirm: bool = False,
        background: bool = True,
    ) -> TaskRunResult | None:
        """Start observe→plan→(confirm→act) for ``goal``.

        When ``background`` is True (GUI default), returns None immediately and
        runs on a worker thread. Headless tests use ``background=False``.
        """
        goal = (goal or "").strip()
        if not goal:
            raise BaodouError(ErrorCode.CONFIG_INVALID, "任务目标不能为空")
        with self._lock:
            if self._busy:
                raise BaodouError(ErrorCode.INTERNAL, "已有任务在运行")
            self._busy = True
            self._goal = goal
            self._last_result = None

        if background:
            self._worker = threading.Thread(
                target=self._run_worker,
                kwargs={
                    "goal": goal,
                    "execute": execute,
                    "auto_confirm": auto_confirm,
                },
                name="baodou-ui-session",
                daemon=True,
            )
            self._worker.start()
            return None
        return self._run_worker(goal=goal, execute=execute, auto_confirm=auto_confirm)

    def preview_task(self, goal: str, *, background: bool = False) -> TaskRunResult | None:
        return self.start_task(goal, execute=False, auto_confirm=False, background=background)

    def _run_worker(
        self,
        *,
        goal: str,
        execute: bool,
        auto_confirm: bool,
    ) -> TaskRunResult:
        t0 = time.perf_counter()
        tid = new_trace_id()
        result: TaskRunResult | None = None
        try:
            self.token.check()
            if self.control.is_stopped():
                raise BaodouError(ErrorCode.CANCELLED, "紧急停止中，请先复位")

            effective_goal = apply_corrections_to_goal(goal, self.corrections.to_list())
            # auto_confirm for low-risk when requested (still cannot bypass high)
            prev_auto = self.config.agent.auto_confirm
            self.config.agent.auto_confirm = bool(auto_confirm)

            self._set_activity(ActivityPhase.CAPTURING, message="采集屏幕…", capturing=True)
            agent = TaskAgent(
                self.config,
                token=self.token,
                control=self.control,
                confirm_callback=self._confirm_callback if execute and not auto_confirm else None,
            )

            # Capture for UI preview image first
            try:
                frame, image = agent._capture_frame(  # noqa: SLF001 — intentional for UI
                    kind=FrameKind.PREVIEW if not execute else FrameKind.MODEL,
                    trace_id=tid,
                    force=True,
                )
                with self._lock:
                    self._last_frame = frame
                    self._last_image = image
                if getattr(frame, "capture_ms", None) is not None:
                    self.metrics.record_capture(frame.capture_ms)
                else:
                    self.metrics.record_capture((time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                self.metrics.push_error(f"capture: {exc}")
                frame, image = None, None

            self._set_activity(ActivityPhase.RECOGNIZING, message="识别 UI…", recognizing=True)
            # Run full agent path
            self._set_activity(
                ActivityPhase.INFERRING if execute else ActivityPhase.RECOGNIZING,
                message="规划中…",
                inferring=True,
            )
            result = agent.run(
                effective_goal,
                trace_id=tid,
                execute=execute,
                confirmed=bool(auto_confirm),
            )
            # Attach corrections to task
            if result.task is not None:
                result.task.corrections = self.corrections.to_list()
                # Keep original user goal (without correction appendix) for display
                if result.task.user_goal != goal:
                    # agent used effective_goal; store original in notes via pause_reason only if empty
                    pass
                result.task.user_goal = goal

            # Apply correction filter for display elements
            if result.vision is not None:
                result.vision = apply_corrections_to_vision(
                    result.vision, self.corrections.to_list()
                )

            with self._lock:
                self._last_result = result
                self._task = result.task
                if result.frame is not None:
                    self._last_frame = result.frame
                if result.vision is not None:
                    self._last_vision = result.vision

            self.metrics.from_task_result(result)
            self.metrics.record_e2e((time.perf_counter() - t0) * 1000)

            # Queue stats best-effort
            cap = agent.capture
            if hasattr(cap, "queue_stats"):
                try:
                    st = cap.queue_stats()  # type: ignore[attr-defined]
                    self.metrics.record_queue(int(st.get("length", 0)), int(st.get("dropped", 0)))
                except Exception:  # noqa: BLE001
                    pass

            self.config.agent.auto_confirm = prev_auto

            if result.error is not None:
                phase = (
                    ActivityPhase.PAUSED
                    if result.task and result.task.state == TaskState.PAUSED
                    else ActivityPhase.ERROR
                )
                self._set_activity(
                    phase,
                    message=result.error.message[:160],
                )
            elif result.ok:
                self._set_activity(
                    ActivityPhase.IDLE,
                    message="完成" if execute else "预览就绪",
                )
            else:
                self._set_activity(ActivityPhase.IDLE, message="结束")
            return result
        except BaodouError as exc:
            self.metrics.push_error(exc.message)
            self._set_activity(ActivityPhase.ERROR, message=exc.message)
            # Fabricate minimal result
            task = TaskContext(user_goal=goal, trace_id=tid, state=TaskState.FAILED)
            task.last_error = exc.message
            result = TaskRunResult(trace_id=tid, task=task, error=exc, ok=False)
            with self._lock:
                self._last_result = result
                self._task = task
            return result
        except Exception as exc:  # noqa: BLE001
            err = BaodouError(ErrorCode.INTERNAL, str(exc), cause=exc)
            self.metrics.push_error(err.message)
            self._set_activity(ActivityPhase.ERROR, message=err.message)
            task = TaskContext(user_goal=goal, trace_id=tid, state=TaskState.FAILED)
            task.last_error = err.message
            result = TaskRunResult(trace_id=tid, task=task, error=err, ok=False)
            with self._lock:
                self._last_result = result
                self._task = task
            return result
        finally:
            with self._lock:
                self._busy = False
            self._notify()

    def refresh_observe(self, goal: str | None = None) -> dict[str, Any]:
        """One-shot capture + vision for live screen view (no full agent plan)."""
        g = (goal or self._goal or "观察当前屏幕").strip()
        self._set_activity(ActivityPhase.CAPTURING, message="刷新预览…", capturing=True)
        agent = TaskAgent(self.config, token=self.token, control=self.control)
        tid = new_trace_id()
        t0 = time.perf_counter()
        frame, image = agent._capture_frame(  # noqa: SLF001
            kind=FrameKind.PREVIEW, trace_id=tid, force=True
        )
        self.metrics.record_capture((time.perf_counter() - t0) * 1000)
        self._set_activity(ActivityPhase.RECOGNIZING, message="识别…", recognizing=True)
        t1 = time.perf_counter()
        vision = agent._recognize(frame, trace_id=tid, image=image, goal=g)  # noqa: SLF001
        self.metrics.record_vision((time.perf_counter() - t1) * 1000)
        vision = apply_corrections_to_vision(vision, self.corrections.to_list())
        with self._lock:
            self._last_frame = frame
            self._last_image = image
            self._last_vision = vision
        self._set_activity(ActivityPhase.IDLE, message="预览已更新")
        return {
            "frame": frame.log_summary(),
            "vision": vision.log_summary(),
            "elements": [
                {
                    "element_id": e.element_id,
                    "type": e.type.value,
                    "text": (e.text or "")[:40],
                    "bbox": e.bbox.model_dump(),
                    "clickable": e.clickable,
                }
                for e in vision.elements[:48]
            ],
        }

    # ------------------------------------------------------------------ snapshots
    def get_last_image(self) -> Any:
        with self._lock:
            return self._last_image

    def get_last_frame(self) -> ScreenFrame | None:
        with self._lock:
            return self._last_frame

    def get_last_vision(self) -> UIVisionResult | None:
        with self._lock:
            return self._last_vision

    def get_last_result(self) -> TaskRunResult | None:
        with self._lock:
            return self._last_result

    def activity(self) -> ActivityStatus:
        with self._lock:
            return self._activity.model_copy(deep=True)

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            result = self._last_result
            vision = self._last_vision
            if vision is None and result is not None:
                vision = result.vision
            elements: list[dict[str, Any]] = []
            if vision is not None:
                for e in vision.elements[:48]:
                    elements.append(
                        {
                            "element_id": e.element_id,
                            "type": e.type.value,
                            "text": (e.text or e.name or "")[:40],
                            "bbox": e.bbox.model_dump(),
                            "confidence": e.confidence,
                            "clickable": e.clickable,
                            "editable": e.editable,
                            "source": e.source,
                        }
                    )
            previews = []
            plan_summary = None
            obs_text = ""
            highlight: list[str] = []
            if result is not None:
                previews = [p.log_summary() for p in result.previews]
                if result.plan is not None:
                    plan_summary = result.plan.log_summary()
                    for s in result.plan.steps:
                        if s.target_element_id:
                            highlight.append(s.target_element_id)
                if result.observation is not None:
                    obs_text = result.observation.observation or ""
            pending_prev = self._pending_preview.log_summary() if self._pending_preview else None
            pending_step = None
            if self._pending_step is not None:
                s = self._pending_step
                pending_step = {
                    "step_id": s.step_id,
                    "action": s.action.value,
                    "target_element_id": s.target_element_id,
                    "risk": s.risk.value,
                    "text": (s.text or "")[:40],
                }
            pending_safety = (
                self._pending_decision.log_summary() if self._pending_decision else None
            )
            if self._pending_preview and self._pending_preview.target_element_id:
                highlight = [self._pending_preview.target_element_id]
            err = ""
            ok: bool | None = None
            if result is not None:
                ok = result.ok
                if result.error is not None:
                    err = result.error.message
            ctrl = self.control.status().get("state", "running")
            return SessionSnapshot(
                activity=self._activity.model_copy(deep=True),
                metrics=self.metrics.snapshot(),
                task=self._task.model_copy(deep=True) if self._task else None,
                goal=self._goal,
                observation_text=obs_text,
                elements=elements,
                plan_summary=plan_summary,
                previews=previews,
                pending_preview=pending_prev,
                pending_step=pending_step,
                pending_safety=pending_safety,
                corrections=self.corrections.log_summary(),
                last_result_ok=ok,
                last_error=err,
                busy=self._busy,
                dry_run=bool(self.config.actuator.dry_run),
                mock=self.mock,
                control_state=str(ctrl),
                highlight_ids=highlight,
            )

    def diagnostics(self) -> DiagnosticBundle:
        with self._lock:
            result = self._last_result
            return build_diagnostics(
                task=self._task,
                frame=self._last_frame,
                vision=self._last_vision or (result.vision if result else None),
                observation=result.observation if result else None,
                plan=result.plan if result else None,
                previews=result.previews if result else None,
                actions=result.actions if result else None,
                verifications=result.verifications if result else None,
                corrections=self.corrections,
                metrics=self.metrics,
                activity=self._activity.log_summary(),
                events=result.events if result else None,
                error=result.error if result else None,
                prompt_version=self._prompt_version,
                model_version=self._model_version,
            )

    def find_element(self, element_id: str) -> UIElement | None:
        vision = self.get_last_vision()
        if vision is None:
            return None
        return vision.by_id(element_id)

    def element_bbox(self, element_id: str) -> BBox | None:
        el = self.find_element(element_id)
        return el.bbox if el else None
