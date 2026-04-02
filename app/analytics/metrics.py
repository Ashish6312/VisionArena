"""Small pure calculations shared by performance.py and reports.py — kept
separate so each formula (accuracy, average-of-a-list) is defined and
tested exactly once."""

from __future__ import annotations

from collections.abc import Sequence


def accuracy(hits: int, shots: int) -> float:
    return round(hits / shots * 100, 1) if shots else 0.0


def average(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def zone_time(arena, position_log: Sequence[tuple[float, float, float]]) -> dict[str, float]:
    """`position_log` is a time-ordered sequence of (timestamp, x, y)
    samples of the player's in-game position. Returns seconds spent per
    named zone (samples outside every zone are attributed to "Open Arena")."""
    totals: dict[str, float] = {}
    for i in range(1, len(position_log)):
        t0, _, _ = position_log[i - 1]
        t1, x1, y1 = position_log[i]
        dt = t1 - t0
        if dt <= 0:
            continue
        zone = arena.zone_at(x1, y1)
        name = zone.name if zone else "Open Arena"
        totals[name] = totals.get(name, 0.0) + dt
    return {name: round(seconds, 2) for name, seconds in totals.items()}
