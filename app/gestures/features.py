"""
Feature extraction — Week 4.

Pure functions turning raw landmark geometry into interpretable booleans/
values. Deliberately NOT a neural network: MediaPipe already did the hard
perception problem (finding 21/33 points reliably); classifying "is this
hand raised" from those points is simple, explainable geometry, and
being interpretable matters when a false SHOOT event is a bug you need to
debug from a demo video, not a black box.

Coordinates are MediaPipe's normalized image space: x, y in [0, 1], origin
top-left, so **smaller y is higher up the frame** — "wrist above shoulder"
means `wrist.y < shoulder.y`.
"""

from __future__ import annotations

from .hands import Landmark
from .pose import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

# Tunable thresholds — heuristics, not physics. See docs/cv_pipeline.md
# "Technical Challenges" for why these are calibration-free and what that costs.
_CROUCH_TORSO_RATIO = 0.20  # normalized shoulder-to-hip distance below this = crouching
_SIDE_ZONE = 0.33  # body center inside the outer third of the frame = MOVE_LEFT/RIGHT

# Hand landmark indices (MediaPipe's documented ordering)
_WRIST = 0
_FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


# ---- pose-based features ----------------------------------------------------


def is_hand_raised(pose: list[Landmark], side: str = "right") -> bool:
    """A wrist above (smaller y than) its own-side shoulder counts as raised."""
    wrist_idx, shoulder_idx = (
        (RIGHT_WRIST, RIGHT_SHOULDER) if side == "right" else (LEFT_WRIST, LEFT_SHOULDER)
    )
    return pose[wrist_idx][1] < pose[shoulder_idx][1]


def both_hands_raised(pose: list[Landmark]) -> bool:
    return is_hand_raised(pose, "left") and is_hand_raised(pose, "right")


def body_center_x(pose: list[Landmark]) -> float:
    """Midpoint between the shoulders — used as the player's horizontal position."""
    return (pose[LEFT_SHOULDER][0] + pose[RIGHT_SHOULDER][0]) / 2.0


def horizontal_zone(pose: list[Landmark]) -> str | None:
    """MOVE_LEFT / MOVE_RIGHT if the player has stepped into an outer third
    of the frame, else None (center zone = no horizontal movement event)."""
    x = body_center_x(pose)
    if x < _SIDE_ZONE:
        return "LEFT"
    if x > 1.0 - _SIDE_ZONE:
        return "RIGHT"
    return None


def is_crouching(pose: list[Landmark]) -> bool:
    """Shoulder-to-hip vertical distance compressed relative to a standing
    torso — a simple, calibration-free crouch proxy (see module docstring)."""
    shoulder_y = (pose[LEFT_SHOULDER][1] + pose[RIGHT_SHOULDER][1]) / 2.0
    hip_y = (pose[LEFT_HIP][1] + pose[RIGHT_HIP][1]) / 2.0
    return (hip_y - shoulder_y) < _CROUCH_TORSO_RATIO


def arm_aim_vector(pose: list[Landmark], side: str = "right") -> tuple[float, float] | None:
    """V2.0 Part D: elbow->wrist direction as a 2D aim vector — a simple,
    interpretable proxy for "which way is the arm pointing", not a
    physically accurate 3D aim model. Image-space and game/screen-space
    agree on sign here (+y is down in both), so no coordinate flip is
    needed — only normalization to a unit vector, done by the caller
    (app/game/aim.py::normalize). Returns None for a degenerate reading
    (elbow and wrist reported at ~the same point), so an unreliable
    landmark never produces a fabricated direction."""
    elbow_idx, wrist_idx = (RIGHT_ELBOW, RIGHT_WRIST) if side == "right" else (LEFT_ELBOW, LEFT_WRIST)
    ex, ey = pose[elbow_idx][0], pose[elbow_idx][1]
    wx, wy = pose[wrist_idx][0], pose[wrist_idx][1]
    dx, dy = wx - ex, wy - ey
    if (dx * dx + dy * dy) ** 0.5 < 1e-6:
        return None
    return (dx, dy)


# ---- hand-based features -----------------------------------------------------


def finger_extended(hand: list[Landmark], finger: str) -> bool:
    """A finger counts as extended if its tip is above (smaller y than) its
    own PIP joint — folded fingers curl the tip back down below the joint."""
    tip_idx, pip_idx = _FINGER_TIPS[finger], _FINGER_PIPS[finger]
    return hand[tip_idx][1] < hand[pip_idx][1]


def extended_fingers(hand: list[Landmark]) -> set[str]:
    return {name for name in _FINGER_TIPS if finger_extended(hand, name)}


def is_pointing(hand: list[Landmark]) -> bool:
    """Index finger out, the rest curled — the AIM gesture."""
    extended = extended_fingers(hand)
    return extended == {"index"}
