"""Task state machine transitions (Phase F).

Legal graph:

    idle → observing → planning → awaiting_confirmation → executing → verifying
         ↘ completed / failed / paused / cancelled (from most active states)
"""

from __future__ import annotations

from core.errors import BaodouError, ErrorCode
from core.models import TaskState

# Allowed transitions: from → frozenset(to)
_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.IDLE: frozenset({TaskState.OBSERVING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.OBSERVING: frozenset(
        {
            TaskState.PLANNING,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.PAUSED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.PLANNING: frozenset(
        {
            TaskState.AWAITING_CONFIRMATION,
            TaskState.EXECUTING,  # auto-confirm / no steps needing confirm
            TaskState.COMPLETED,  # empty plan (observe-only)
            TaskState.FAILED,
            TaskState.PAUSED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.AWAITING_CONFIRMATION: frozenset(
        {
            TaskState.EXECUTING,
            TaskState.PAUSED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.COMPLETED,
        }
    ),
    TaskState.EXECUTING: frozenset(
        {
            TaskState.VERIFYING,
            TaskState.PAUSED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.AWAITING_CONFIRMATION,  # next step needs confirm
            TaskState.COMPLETED,
        }
    ),
    TaskState.VERIFYING: frozenset(
        {
            TaskState.EXECUTING,  # next step
            TaskState.AWAITING_CONFIRMATION,
            TaskState.OBSERVING,  # reidentify recovery
            TaskState.PLANNING,  # replan recovery
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.PAUSED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.PAUSED: frozenset(
        {
            TaskState.AWAITING_CONFIRMATION,
            TaskState.EXECUTING,
            TaskState.OBSERVING,
            TaskState.CANCELLED,
            TaskState.FAILED,
            TaskState.COMPLETED,
        }
    ),
    TaskState.CANCELLED: frozenset(),
}

_TERMINAL = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})


def is_terminal(state: TaskState) -> bool:
    return state in _TERMINAL


def can_transition(current: TaskState, new: TaskState) -> bool:
    if current == new:
        return True
    allowed = _TRANSITIONS.get(current, frozenset())
    return new in allowed


def transition(current: TaskState, new: TaskState) -> TaskState:
    """Return new state or raise if illegal."""
    if current == new:
        return new
    if not can_transition(current, new):
        raise BaodouError(
            ErrorCode.INTERNAL,
            f"非法状态迁移: {current.value} → {new.value}",
            details={"from": current.value, "to": new.value},
        )
    return new


def allowed_targets(current: TaskState) -> list[str]:
    return sorted(s.value for s in _TRANSITIONS.get(current, frozenset()))
