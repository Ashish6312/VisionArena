"""
Thread synchronization primitive — Phase 2 (V2).

WHY ONE CLASS INSTEAD OF `queue.Queue`, `Condition`, etc.:

The whole point of "latest frame/state wins" is that there is never a
backlog to synchronize access *to* — there's exactly one slot, always
holding the newest published item. `queue.Queue` models a FIFO backlog
(exactly what we're trying to avoid: a slow consumer falling behind a fast
producer, then working through increasingly stale items). A
`Condition`/`Lock`-only approach needs the caller to hand-roll the
wait/notify dance. `LatestSlot` bundles the minimum needed for both access
patterns this project actually has:

  - a producer that always overwrites (`put`) — never blocks, backlog
    literally cannot form because there's only one slot to overwrite.
  - a consumer that wants the freshest value RIGHT NOW, never waiting
    (`get`) — used by the 60 FPS game loop, which must never stall behind
    CV inference.
  - a consumer that wants to sleep efficiently until something NEW exists,
    instead of busy-polling `get()` in a tight loop (`wait_for_update`) —
    used by the CV worker waiting on fresh camera frames.

SINGLE-CONSUMER CAVEAT for `wait_for_update`: the internal `Event` is
cleared by whichever call to `wait_for_update` wakes first. With more than
one thread calling `wait_for_update` on the same slot, a second waiter can
miss a wakeup the first one already consumed. Every use of
`wait_for_update` in this codebase has exactly one waiter — documented
here so that constraint isn't silently violated later.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Holds at most one item: the newest one published via `put`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self._ready = threading.Event()

    def put(self, item: T) -> None:
        """Overwrite the slot. Never blocks — there's nothing to wait for."""
        with self._lock:
            self._item = item
        self._ready.set()

    def get(self) -> T | None:
        """Return the current item immediately (or None if `put` has never
        been called). Never blocks — safe to call every frame of a
        real-time render loop."""
        with self._lock:
            return self._item

    def wait_for_update(self, timeout: float | None = None) -> bool:
        """Block until `put` has been called since the last call to this
        method (or `timeout` seconds elapse). Returns True on a genuine
        update, False on timeout — a timeout is the normal way a consumer
        periodically re-checks its own shutdown flag, not an error."""
        woke = self._ready.wait(timeout=timeout)
        if woke:
            self._ready.clear()
        return woke
