"""Bounded frame queues with drop policies under backpressure."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueueStats:
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    high_watermark: int = 0
    last_drop_reason: str = ""


@dataclass
class FrameQueue(Generic[T]):
    """
    Thread-safe bounded queue.

    drop_policy:
      - newest: when full, drop the *incoming* frame (keep older work) — rarely wanted
      - oldest: when full, drop oldest and accept newest (default for realtime capture)
    Note: config label `newest` means "prefer newest frames" → drop oldest.
    """

    maxsize: int = 4
    drop_policy: str = "newest"  # prefer newest under pressure
    name: str = "default"
    _q: deque[T] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _not_empty: threading.Condition = field(init=False)
    stats: QueueStats = field(default_factory=QueueStats)

    def __post_init__(self) -> None:
        self._not_empty = threading.Condition(self._lock)
        if self.maxsize < 1:
            self.maxsize = 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def clear(self) -> None:
        with self._lock:
            self._q.clear()

    def put(self, item: T) -> bool:
        """
        Enqueue item. Returns False if the item itself was dropped.
        """
        with self._not_empty:
            if len(self._q) >= self.maxsize:
                # "newest" = prefer newest frames → drop oldest
                # "oldest" = keep older work → drop incoming
                if self.drop_policy in ("oldest", "drop_incoming", "keep_oldest"):
                    self.stats.dropped += 1
                    self.stats.last_drop_reason = "drop_incoming"
                    return False
                self._q.popleft()
                self.stats.dropped += 1
                self.stats.last_drop_reason = "drop_oldest_for_newest"
            self._q.append(item)
            self.stats.enqueued += 1
            self.stats.high_watermark = max(self.stats.high_watermark, len(self._q))
            self._not_empty.notify()
            return True

    def get(self, timeout: float | None = None) -> T | None:
        with self._not_empty:
            if not self._q:
                if timeout is None:
                    return None
                end = time.monotonic() + timeout
                while not self._q:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)
            if not self._q:
                return None
            item = self._q.popleft()
            self.stats.dequeued += 1
            return item

    def snapshot_stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "name": self.name,
                "size": len(self._q),
                "maxsize": self.maxsize,
                "enqueued": self.stats.enqueued,
                "dequeued": self.stats.dequeued,
                "dropped": self.stats.dropped,
                "high_watermark": self.stats.high_watermark,
                "last_drop_reason": self.stats.last_drop_reason,
                "drop_policy": self.drop_policy,
            }
