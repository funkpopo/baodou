"""Action frequency, consecutive count, task duration, mouse range (Phase G)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from core.config import SafetySection
from core.errors import ErrorCode, SafetyError
from core.models import ActionStep, Point


@dataclass
class LimitState:
    """Mutable counters for one task session (or shared process guard)."""

    started_at: float = field(default_factory=time.monotonic)
    action_times: deque[float] = field(default_factory=deque)
    consecutive: int = 0
    last_point: Point | None = None
    total_mouse_move_px: float = 0.0
    stopped: bool = False

    def reset_task(self) -> None:
        self.started_at = time.monotonic()
        self.action_times.clear()
        self.consecutive = 0
        self.last_point = None
        self.total_mouse_move_px = 0.0
        self.stopped = False


class SafetyLimits:
    def __init__(self, cfg: SafetySection, state: LimitState | None = None) -> None:
        self.cfg = cfg
        self.state = state or LimitState()

    def reset_task(self) -> None:
        self.state.reset_task()

    def check_pre_action(self, step: ActionStep, *, resolved: Point | None = None) -> None:
        """Raise SafetyError if any hard limit is exceeded."""
        if self.state.stopped:
            raise SafetyError(
                "安全限制：会话已停止",
                code=ErrorCode.PERMISSION_DENIED,
                details={"limit": "stopped"},
            )

        now = time.monotonic()
        elapsed = now - self.state.started_at
        max_dur = float(self.cfg.max_task_duration_sec)
        if max_dur > 0 and elapsed > max_dur:
            raise SafetyError(
                f"任务超时限制: {max_dur:.0f}s",
                code=ErrorCode.ACTION_TIMEOUT,
                details={"limit": "max_task_duration_sec", "elapsed": round(elapsed, 2)},
            )

        # Sliding 60s window
        max_apm = max(1, int(self.cfg.max_actions_per_minute))
        while self.state.action_times and now - self.state.action_times[0] > 60.0:
            self.state.action_times.popleft()
        if len(self.state.action_times) >= max_apm:
            raise SafetyError(
                f"动作频率超限: {max_apm}/min",
                code=ErrorCode.ACTION_FAILED,
                details={
                    "limit": "max_actions_per_minute",
                    "count": len(self.state.action_times),
                },
            )

        max_consec = max(1, int(self.cfg.max_consecutive_actions))
        if self.state.consecutive >= max_consec:
            raise SafetyError(
                f"连续动作数量超限: {max_consec}",
                code=ErrorCode.PERMISSION_DENIED,
                details={
                    "limit": "max_consecutive_actions",
                    "consecutive": self.state.consecutive,
                },
            )

        # Mouse travel budget
        max_move = float(self.cfg.max_mouse_move_px)
        pt = resolved or step.target_point
        if pt is not None and max_move > 0 and self.state.last_point is not None:
            dx = float(pt.x - self.state.last_point.x)
            dy = float(pt.y - self.state.last_point.y)
            dist = (dx * dx + dy * dy) ** 0.5
            projected = self.state.total_mouse_move_px + dist
            if projected > max_move:
                raise SafetyError(
                    f"鼠标移动范围超限: {max_move:.0f}px",
                    code=ErrorCode.COORDINATE_OUT_OF_BOUNDS,
                    details={
                        "limit": "max_mouse_move_px",
                        "total": round(projected, 1),
                        "step_dist": round(dist, 1),
                    },
                )

    def record_action(self, *, resolved: Point | None = None) -> None:
        now = time.monotonic()
        self.state.action_times.append(now)
        self.state.consecutive += 1
        if resolved is not None:
            if self.state.last_point is not None:
                dx = float(resolved.x - self.state.last_point.x)
                dy = float(resolved.y - self.state.last_point.y)
                self.state.total_mouse_move_px += (dx * dx + dy * dy) ** 0.5
            self.state.last_point = resolved

    def snapshot(self) -> dict[str, float | int | bool]:
        now = time.monotonic()
        return {
            "elapsed_sec": round(now - self.state.started_at, 2),
            "actions_last_minute": len(self.state.action_times),
            "consecutive": self.state.consecutive,
            "total_mouse_move_px": round(self.state.total_mouse_move_px, 1),
            "stopped": self.state.stopped,
        }
