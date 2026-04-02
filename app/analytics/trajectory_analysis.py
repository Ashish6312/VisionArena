"""Adapts a Week 3 `TrackState` (physical player, camera-space pixels) into
the "Movement Analysis" fields the session report wants. Kept separate
from vision/trajectory.py because that module owns live per-frame
tracking; this one owns turning a finished session's TrackState into
report-shaped numbers — a reporting concern, not a tracking one."""

from __future__ import annotations

from dataclasses import dataclass

from ..vision.trajectory import TrackState


@dataclass
class MovementSummary:
    distance_traveled_px: float
    average_speed_px_s: float
    max_speed_px_s: float
    direction_changes: int
    stationary_seconds: float


def summarize_movement(track: TrackState | None) -> MovementSummary:
    """Returns all-zero stats if there's no CV track (e.g. a keyboard-only
    session with no camera) — a missing physical-movement signal shouldn't
    crash a report that still has real game telemetry to show."""
    if track is None:
        return MovementSummary(0.0, 0.0, 0.0, 0, 0.0)
    return MovementSummary(
        distance_traveled_px=track.distance_traveled,
        average_speed_px_s=track.average_speed,
        max_speed_px_s=track.max_speed,
        direction_changes=track.direction_changes,
        stationary_seconds=track.stationary_seconds,
    )
