"""
VisionState — Phase 2 (V2): one CV pipeline pass's output.

An immutable snapshot published by `CVWorker` once per completed frame —
detections, tracks, gestures, and the `GameEvent`s already derived from
them (so `GameRunner`/the game loop never re-derives anything, just reads
what the CV worker already computed). Deliberately does NOT carry the full
session-long trajectory history — that stays in `CVWorker.track_store`
(one `TrajectoryStore`, alive for the whole session) so a per-tick
snapshot stays cheap to publish. `primary_track_id` is enough for the game
loop / a future HUD to know *which* track is "the player"; the full
`TrackState` (distance/speed/etc.) is queried from `track_store` once, at
session end, by `GameRunner._finish` — exactly how it already worked
before Phase 2, just re-homed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..events import GameEvent
from ..gestures.classifier import GestureResult
from .detector import Detection
from .tracker import TrackedObject


@dataclass(frozen=True)
class VisionState:
    frame_id: int
    timestamp: float  # time.time() when this VisionState was published
    frame_timestamp: float  # time.time() of the SOURCE camera frame (cv_timestamp)
    detections: list[Detection] = field(default_factory=list)
    tracked_objects: list[TrackedObject] = field(default_factory=list)
    primary_track_id: int | None = None
    gestures: list[GestureResult] = field(default_factory=list)
    game_events: list[GameEvent] = field(default_factory=list)
    processing_latency_ms: float = 0.0  # detect + track + gesture time, excludes camera read

    @property
    def age_seconds(self) -> float:
        """How long ago this state was published, from "now"."""
        return max(0.0, time.time() - self.timestamp)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "age_seconds": round(self.age_seconds, 3),
            "detections": len(self.detections),
            "tracked_objects": len(self.tracked_objects),
            "primary_track_id": self.primary_track_id,
            "gestures": [g.gesture.value for g in self.gestures],
            "processing_latency_ms": self.processing_latency_ms,
        }


def classify_cv_status(
    vision_state: VisionState | None, worker_error: str | None, stale_seconds: float
) -> str:
    """One of disabled/connecting/connected/stale/unavailable — shared by
    `GameRunner._cv_status()` and `scripts/run_full_system.py`'s HUD so
    "what does CONNECTED vs STALE mean" is defined exactly once. Callers
    handle "disabled" themselves (this function assumes CV was at least
    requested — there's no worker to ask about otherwise)."""
    if worker_error:
        return "unavailable"
    if vision_state is None:
        return "connecting"
    if vision_state.age_seconds > stale_seconds:
        return "stale"
    return "connected"
