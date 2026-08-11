"""Simple action rate limiter (Phase F)."""

from __future__ import annotations

import time
from collections import deque

from core.errors import ActuatorError, ErrorCode


class ActionRateLimiter:
    def __init__(self, max_per_minute: int = 30) -> None:
        self.max_per_minute = max(1, max_per_minute)
        self._times: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()
        if len(self._times) >= self.max_per_minute:
            raise ActuatorError(
                f"动作频率超限: {self.max_per_minute}/min",
                code=ErrorCode.ACTION_FAILED,
                details={"max_per_minute": self.max_per_minute, "count": len(self._times)},
            )

    def record(self) -> None:
        self._times.append(time.monotonic())

    def reset(self) -> None:
        self._times.clear()
