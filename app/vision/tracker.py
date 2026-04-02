"""
Multi-object tracking with ByteTrack — Week 3.

DETECTION vs TRACKING: detection answers "where are the objects in THIS
frame?" with no memory. Tracking adds memory: it matches new detections to
existing tracks (by predicted position + box overlap) and assigns a
persistent integer ID that stays the same across frames — "Player 1" in
frame 40 is still "Player 1" in frame 41.

WHY BYTETRACK: a fast, well-established tracking-by-detection algorithm.
Its key idea is matching using BOTH high- and low-confidence detections
(most trackers discard low-confidence boxes, losing an object during
partial occlusion). Not reimplemented here — this wraps the maintained
`supervision.ByteTrack` implementation used throughout the CV community.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Detection


@dataclass
class TrackedObject:
    """A detection matched to a persistent track ID."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Tracker:
    """Wraps supervision's ByteTrack.

    Usage:
        tracker = Tracker()
        tracked = tracker.update(detections)   # once per frame
    """

    def __init__(self, frame_rate: int = 30):
        try:
            import supervision as sv
        except ImportError as e:
            raise ImportError("supervision is not installed. Run: pip install supervision") from e

        self._sv = sv
        self._tracker = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Match this frame's detections to existing tracks (or start new ones)."""
        if not detections:
            # Still update with an empty batch so ByteTrack can age-out lost tracks.
            self._tracker.update_with_detections(self._sv.Detections.empty())
            return []

        xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_detections)
        class_names = {d.class_id: d.class_name for d in detections}

        if tracked.tracker_id is None:
            return []

        results: list[TrackedObject] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            cid = int(tracked.class_id[i])
            results.append(
                TrackedObject(
                    track_id=int(tracked.tracker_id[i]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(tracked.confidence[i]),
                    class_id=cid,
                    class_name=class_names.get(cid, str(cid)),
                )
            )
        return results

    def reset(self) -> None:
        """Clear all tracks — call at the start of a new session."""
        self._tracker.reset()
