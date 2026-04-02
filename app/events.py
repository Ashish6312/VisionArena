"""
The shared vocabulary between "what the player did" and "what the game does".

WHY THIS MODULE SITS AT THE TOP LEVEL, NOT INSIDE game/:
`gestures/` needs to produce these without importing the game engine, and
`game/` needs to consume them without importing MediaPipe. Putting the
contract in its own module with no dependencies on either side is what
makes this true: a keyboard key press and a raised hand both become the
exact same `GameEvent(GameEventType.SHOOT)` before anything game-related
ever sees them. The game engine cannot tell which one sent it — and later,
neither will a Unity client reading the same event off the WebSocket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class GameEventType(str, Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    SHOOT = "SHOOT"
    SHIELD = "SHIELD"
    CROUCH = "CROUCH"
    AIM = "AIM"
    PAUSE = "PAUSE"


@dataclass(frozen=True)
class GameEvent:
    """One game-relevant action, regardless of where it came from.

    `source` is kept only for debugging/analytics ("was this session mostly
    keyboard or gestures?") — the game engine's behavior never branches on it.
    """

    type: GameEventType
    source: str  # "keyboard" | "gesture"
    timestamp: float
    confidence: float = 1.0  # keyboard is always 1.0; gestures carry real confidence
    # V2.0 Part D: only ever set on an AIM event, carrying the 2D direction
    # vector derived from pose elbow->wrist geometry (see
    # app/gestures/features.py::arm_aim_vector). None for every other event
    # type and for keyboard-sourced events — additive, doesn't change the
    # shape of any existing event.
    aim_vector: tuple[float, float] | None = None

    @classmethod
    def now(cls, type: GameEventType, source: str, confidence: float = 1.0) -> GameEvent:
        return cls(type=type, source=source, timestamp=time.time(), confidence=confidence)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
            "aim_vector": list(self.aim_vector) if self.aim_vector else None,
        }
