"""
Gesture classification — Week 4, aim direction added in V2.0 Part D.

Combines hand + pose landmarks into `GestureResult`s, then converts those
into the same `GameEvent` objects a keyboard press produces (see
app/events.py). This is the file that turns "computer vision" into
"game input" — everything downstream (game/, backend/) only ever sees
`GameEvent`.

NOT IMPLEMENTED: MOVE_FORWARD/MOVE_BACKWARD. A monocular camera can't
measure depth directly; the only proxy available (bounding-box growth
rate) is noisy and was cut for reliability rather than shipped half-working
— see docs/cv_pipeline.md "Technical Challenges". Movement in this project
is left/right only, driven by the player's position in the frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..events import GameEvent, GameEventType
from ..vision.preprocessing import to_rgb
from . import features
from .hands import HandLandmarkerWrapper
from .pose import PoseLandmarkerWrapper

logger = logging.getLogger(__name__)


@dataclass
class GestureResult:
    gesture: GameEventType
    confidence: float
    timestamp: float
    # V2.0 Part D: only ever set on an AIM result — the pose-derived
    # elbow->wrist direction (app/gestures/features.py::arm_aim_vector).
    # None when no pose was available that frame (graceful degradation:
    # the AIM gesture still fires from the hand shape alone, just without
    # a fresh direction — GameEngine keeps the player's last known aim).
    aim_vector: tuple[float, float] | None = None


class GestureRecognizer:
    """Runs hand + pose detection on a frame and classifies the result into
    zero or more game gestures.

    Usage:
        recognizer = GestureRecognizer()
        results = recognizer.process(frame_bgr)          # List[GestureResult]
        events = recognizer.to_game_events(results)       # List[GameEvent]
        recognizer.close()
    """

    def __init__(self):
        self._hands = HandLandmarkerWrapper()
        self._pose = PoseLandmarkerWrapper()

    def process(self, frame_bgr: np.ndarray) -> list[GestureResult]:
        rgb = to_rgb(frame_bgr)
        now = time.time()
        results: list[GestureResult] = []
        pose = self._pose.detect(rgb)

        # AIM first (V2.0 Part D): if the SAME tick also produces a SHOOT
        # (raised hand, from pose, below) the fresh aim direction must
        # already be on the player before GameEngine fires — see
        # GameEngine.apply_events, which processes this tick's events in
        # order and GameEngine._try_shoot, which reads whatever aim
        # direction is current *at the moment it fires*.
        for hand in self._hands.detect(rgb):
            if features.is_pointing(hand.landmarks):
                aim_vector = features.arm_aim_vector(pose, side="right") if pose is not None else None
                results.append(GestureResult(GameEventType.AIM, hand.confidence, now, aim_vector=aim_vector))

        if pose is not None:
            results.extend(self._classify_pose(pose, now))

        return results

    @staticmethod
    def _classify_pose(pose, now: float) -> list[GestureResult]:
        results: list[GestureResult] = []

        if features.both_hands_raised(pose):
            results.append(GestureResult(GameEventType.SHIELD, 1.0, now))
        elif features.is_hand_raised(pose, "right"):
            results.append(GestureResult(GameEventType.SHOOT, 1.0, now))

        if features.is_crouching(pose):
            results.append(GestureResult(GameEventType.CROUCH, 1.0, now))

        zone = features.horizontal_zone(pose)
        if zone == "LEFT":
            results.append(GestureResult(GameEventType.MOVE_LEFT, 1.0, now))
        elif zone == "RIGHT":
            results.append(GestureResult(GameEventType.MOVE_RIGHT, 1.0, now))

        return results

    @staticmethod
    def to_game_events(results: list[GestureResult]) -> list[GameEvent]:
        return [
            GameEvent(
                type=r.gesture,
                source="gesture",
                timestamp=r.timestamp,
                confidence=r.confidence,
                aim_vector=r.aim_vector,
            )
            for r in results
        ]

    def close(self) -> None:
        self._hands.close()
        self._pose.close()

    def __enter__(self) -> GestureRecognizer:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
