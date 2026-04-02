"""Overlay drawing — turns diagnostics/detections/tracks into pixels on a
frame. Kept separate from the numeric pipeline (camera/detector/tracker) so
none of them need to import cv2's drawing API or know about colors/fonts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .camera import FrameDiagnostics

_TEXT_COLOR = (255, 255, 255)
_BOX_COLOR = (60, 220, 60)


def draw_diagnostics(frame: np.ndarray, diag: FrameDiagnostics, recording: bool = False) -> np.ndarray:
    lines = [
        f"Resolution: {diag.resolution[0]}x{diag.resolution[1]}",
        f"FPS: {diag.fps:.1f}",
        f"Frames: {diag.frame_index}",
        f"Latency: {diag.latency_ms:.1f} ms",
    ]
    if recording:
        lines.append("REC")
    y = 24
    for line in lines:
        color = (60, 60, 255) if line == "REC" else _TEXT_COLOR
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24
    return frame


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in (d.x1, d.y1, d.x2, d.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        pos = (x1, max(12, y1 - 6))
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_tracks(frame: np.ndarray, tracked_objects) -> np.ndarray:
    for t in tracked_objects:
        x1, y1, x2, y2 = (int(v) for v in (t.x1, t.y1, t.x2, t.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"ID {t.track_id} | {t.direction}"
        pos = (x1, max(12, y1 - 6))
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_trail(frame: np.ndarray, points: list[tuple[float, float]], color=(80, 180, 255)) -> np.ndarray:
    """Draws a fading movement trail from a track's trajectory history."""
    for i in range(1, len(points)):
        p1 = tuple(int(v) for v in points[i - 1])
        p2 = tuple(int(v) for v in points[i])
        cv2.line(frame, p1, p2, color, 2)
    return frame
