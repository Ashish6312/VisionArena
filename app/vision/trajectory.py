"""
Trajectory / movement analysis — Week 3.

Turns a stream of tracked bounding boxes into per-track movement state:
velocity, direction, distance traveled, speed stats, and a position
history for drawing a trail. Everything here is IMAGE-SPACE pixels, not
real-world units — converting to meters needs camera calibration (focal
length, height, angle, a known reference size), which is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tracker import TrackedObject

_STATIONARY_SPEED_PX_S = 15.0  # below this, a track counts as "not moving" (absorbs jitter)
_TRAIL_LENGTH = 60  # points kept for on-screen trail drawing


def estimate_direction(dx: float, dy: float, min_movement: float = 3.0) -> str:
    """Classify a displacement into one of 5 labels. `min_movement` is a
    dead zone so per-frame detector jitter doesn't read as constant motion."""
    if abs(dx) < min_movement and abs(dy) < min_movement:
        return "STATIONARY"
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class TrackState:
    """Everything known about one tracked object's movement this session."""

    track_id: int
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)  # px/second
    speed: float = 0.0  # px/second, magnitude of velocity
    direction: str = "STATIONARY"
    distance_traveled: float = 0.0  # cumulative path length, pixels
    max_speed: float = 0.0
    direction_changes: int = 0
    stationary_seconds: float = 0.0
    frames_visible: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def average_speed(self) -> float:
        elapsed = self.last_seen - self.first_seen
        return self.distance_traveled / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "center": [round(v, 1) for v in self.center],
            "velocity": [round(v, 1) for v in self.velocity],
            "speed": round(self.speed, 1),
            "direction": self.direction,
            "distance_traveled": round(self.distance_traveled, 1),
            "average_speed": round(self.average_speed, 1),
            "max_speed": round(self.max_speed, 1),
            "direction_changes": self.direction_changes,
            "stationary_seconds": round(self.stationary_seconds, 2),
            "frames_visible": self.frames_visible,
        }


class TrajectoryStore:
    """Maintains a TrackState per track ID across a session.

    Usage (once per processed frame):
        store = TrajectoryStore()
        states = store.update(tracked_objects, timestamp)
    """

    def __init__(self):
        self.tracks: dict[int, TrackState] = {}

    def update(self, tracked_objects: list[TrackedObject], timestamp: float) -> dict[int, TrackState]:
        seen_ids = set()

        for obj in tracked_objects:
            seen_ids.add(obj.track_id)
            cx, cy = obj.center

            if obj.track_id not in self.tracks:
                self.tracks[obj.track_id] = TrackState(
                    track_id=obj.track_id,
                    center=(cx, cy),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    trail=[(cx, cy)],
                )
                continue

            state = self.tracks[obj.track_id]
            dt = timestamp - state.last_seen
            dx, dy = cx - state.center[0], cy - state.center[1]
            distance = (dx**2 + dy**2) ** 0.5

            vx, vy = (dx / dt, dy / dt) if dt > 0 else (0.0, 0.0)
            speed = (vx**2 + vy**2) ** 0.5
            new_direction = estimate_direction(dx, dy)

            if new_direction != state.direction and "STATIONARY" not in (new_direction, state.direction):
                state.direction_changes += 1
            if speed < _STATIONARY_SPEED_PX_S:
                state.stationary_seconds += dt

            state.center = (cx, cy)
            state.velocity = (vx, vy)
            state.speed = speed
            state.direction = new_direction
            state.distance_traveled += distance
            state.max_speed = max(state.max_speed, speed)
            state.frames_visible += 1
            state.last_seen = timestamp
            state.trail.append((cx, cy))
            if len(state.trail) > _TRAIL_LENGTH:
                state.trail.pop(0)

        return {tid: t for tid, t in self.tracks.items() if tid in seen_ids}

    def get(self, track_id: int) -> TrackState | None:
        return self.tracks.get(track_id)

    def all_tracks(self) -> dict[int, TrackState]:
        return dict(self.tracks)
