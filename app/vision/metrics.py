"""
Lightweight rolling performance counters — Phase 2 (V2).

Same rolling-window technique `Camera._rolling_fps` already uses
internally, extracted here because Phase 2 needs it in three NEW places
that didn't exist before (camera worker rate, CV worker rate, game loop
rate) — `Camera`'s own FPS counter is untouched, this doesn't replace it.

`scripts/benchmark.py` (Phase 3) builds on these same two classes for its
more detailed per-component profiling rather than reinventing rate/latency
tracking a third time.
"""

from __future__ import annotations

import time
from collections import deque


class RollingRate:
    """Rate (events/second) averaged over the last `window` `tick()` calls."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def rate(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return round((len(self._times) - 1) / span, 1) if span > 0 else 0.0


class RollingAverage:
    """Average of the last `window` `add()`ed values — used for latency (ms)."""

    def __init__(self, window: int = 30):
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def average(self) -> float:
        return round(sum(self._values) / len(self._values), 1) if self._values else 0.0

    @property
    def latest(self) -> float:
        return round(self._values[-1], 1) if self._values else 0.0
