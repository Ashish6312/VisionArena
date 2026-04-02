"""
Reaction-time measurement — Phase 1 (V2), upgraded to stimulus IDs in V2.0
Part C.

STIMULUS -> RESPONSE, measured from real, already-computed game signals —
not a fabricated delay:

    an enemy's FSM transitions into ATTACK (the same signal
    EnemyController already produces every tick to drive enemy contact
    damage — see GameRunner._run)
         |
         v
    a UNIQUE stimulus_id is recorded, e.g. "enemy_3_attack_02"
    (ReactionTracker.record_stimulus)
         |
    ... player does something, or nothing ...
         |
         v
    the player's next SHOOT or SHIELD GameEvent is the "response"
         |
         v
    reaction_time = response.timestamp - stimulus.timestamp

WHY STIMULUS IDS INSTEAD OF A SHARED FIFO QUEUE (V2.0 Part C): the old
version pooled every enemy's attack into one shared queue and paired
whichever response arrived next against whichever stimulus was oldest,
with no way to tell them apart, expire a stale one, or reject a duplicate
close on the same stimulus. `record_stimulus`/`close_stimulus` now take an
explicit, unique ID — supporting real per-enemy tracking, safe expiry of
stimuli nobody ever responded to, and duplicate-response rejection
(closing the same ID twice returns `None` the second time, it doesn't
double-count).

WHY THE DEFAULT RESOLUTION IS STILL "CLOSEST BY ARRIVAL, NOT BY ENEMY"
— stated honestly rather than oversold: a keyboard SPACE press or a raised
hand carries no information about *which* enemy the player means to answer
— there's no aiming system feeding into a plain `SHOOT` `GameEvent`.
`observe_event()` is the FIFO convenience built on top of the ID-based
core for exactly that case: close whichever stimulus is oldest. Once a
real target is known (V2.0 Part D's aim system — see
`GameRunner._resolve_reaction_stimulus`), the caller uses
`close_stimulus(specific_id, ...)` directly instead, which is genuine
explicit association, not FIFO wearing an ID's clothing. Multiple
outstanding stimuli from different enemies are correctly tracked
independently either way — only the *choice of which one a given response
answers* falls back to arrival order when no better signal exists.
"""

from __future__ import annotations

import logging
import statistics
from collections import OrderedDict
from dataclasses import dataclass

from ..events import GameEvent, GameEventType

logger = logging.getLogger(__name__)

# Only these event types count as a "response" to a threat stimulus.
RESPONSE_EVENT_TYPES = frozenset({GameEventType.SHOOT, GameEventType.SHIELD})

DEFAULT_EXPIRE_SECONDS = 5.0  # a stimulus nobody answered within this long is dropped, not held forever


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    timestamp: float


@dataclass
class ReactionStats:
    """Empty-safe: an empty/all-discarded sample set reports zeros, same
    convention as the rest of analytics/metrics.py — never a fabricated
    number standing in for "no data"."""

    count: int
    average_s: float
    minimum_s: float
    maximum_s: float
    median_s: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "average_s": self.average_s,
            "minimum_s": self.minimum_s,
            "maximum_s": self.maximum_s,
            "median_s": self.median_s,
        }


def compute_reaction_stats(values: list[float]) -> ReactionStats:
    if not values:
        return ReactionStats(count=0, average_s=0.0, minimum_s=0.0, maximum_s=0.0, median_s=0.0)
    return ReactionStats(
        count=len(values),
        average_s=round(sum(values) / len(values), 3),
        minimum_s=round(min(values), 3),
        maximum_s=round(max(values), 3),
        median_s=round(statistics.median(values), 3),
    )


class ReactionTracker:
    """One instance per session (GameRunner creates one in `start()`).

    Usage (once per tick):
        tracker.record_stimulus("enemy_3_attack_02", timestamp)
        tracker.expire_stale(timestamp)                    # optional housekeeping
        rt = tracker.close_stimulus("enemy_3_attack_02", response_timestamp)  # explicit
        rt = tracker.observe_event(event)                   # FIFO fallback
        ...
        stats = tracker.stats()
    """

    def __init__(self):
        # OrderedDict, not plain dict: `observe_event`'s FIFO fallback needs
        # "oldest still-pending stimulus" in O(1), and insertion order IS
        # arrival order here since stimuli are always recorded as they happen.
        self._pending: OrderedDict[str, Stimulus] = OrderedDict()
        self.reaction_times: list[float] = []
        self.expired_count = 0

    def record_stimulus(self, stimulus_id: str, timestamp: float) -> None:
        """`stimulus_id` must be unique for this stimulus (e.g.
        f"enemy_{enemy_id}_attack_{sequence}") — recording the same ID
        twice overwrites the first, it does not queue two entries."""
        self._pending[stimulus_id] = Stimulus(stimulus_id, timestamp)

    def close_stimulus(self, stimulus_id: str, response_timestamp: float) -> float | None:
        """Explicit association: the caller states exactly which stimulus
        this response answers. Returns the measured reaction time, or
        `None` if:
        - the ID was never recorded, or already closed (duplicate-response
          guard — closing the same stimulus twice never double-counts), or
        - the computed reaction time would be negative (clock skew;
          discarded, not recorded as data).
        """
        stimulus = self._pending.pop(stimulus_id, None)
        if stimulus is None:
            return None
        reaction_time = response_timestamp - stimulus.timestamp
        if reaction_time < 0:
            logger.debug("Discarded negative reaction time (%.3fs) for %s", reaction_time, stimulus_id)
            return None
        self.reaction_times.append(reaction_time)
        return reaction_time

    def observe_event(self, event: GameEvent) -> float | None:
        """FIFO fallback for when no explicit target is known (see module
        docstring): closes the OLDEST outstanding stimulus, regardless of
        which one it is. Returns `None` for irrelevant event types or when
        nothing is outstanding — never fabricates a response."""
        if event.type not in RESPONSE_EVENT_TYPES:
            return None
        if not self._pending:
            return None
        oldest_id = next(iter(self._pending))
        return self.close_stimulus(oldest_id, event.timestamp)

    def expire_stale(self, now: float, max_age_seconds: float = DEFAULT_EXPIRE_SECONDS) -> int:
        """Drops stimuli older than `max_age_seconds` that nobody ever
        responded to — an enemy attack from 5+ seconds ago shouldn't stay
        eligible to be "answered" by an unrelated action much later.
        Returns how many were expired (also tracked in `expired_count`)."""
        stale_ids = [sid for sid, s in self._pending.items() if now - s.timestamp > max_age_seconds]
        for sid in stale_ids:
            del self._pending[sid]
        self.expired_count += len(stale_ids)
        return len(stale_ids)

    def pending_ids(self) -> list[str]:
        return list(self._pending.keys())

    def stats(self) -> ReactionStats:
        return compute_reaction_stats(self.reaction_times)
