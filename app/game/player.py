"""Player entity — Week 5, aim direction added in V2.0 Part D, damage-
feedback timestamp added in V2.0 Part F."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .aim import DEFAULT_AIM_VECTOR, AimVector

MOVE_STEP_PX = 6.0  # per-tick displacement from a single MOVE_* event
SHIELD_DAMAGE_REDUCTION = 0.75  # while shielded, incoming damage is cut by this fraction
CROUCH_DAMAGE_REDUCTION = 0.35


@dataclass
class Player:
    x: float
    y: float
    health: int = 100
    max_health: int = 100
    radius: float = 18.0
    score: int = 0

    # transient per-tick status — set by the engine from this tick's events,
    # and cleared every tick (so shield/crouch require a continuous stream
    # of gesture events to stay active, same as holding a key down)
    shielded: bool = False
    crouching: bool = False
    aiming: bool = False

    # V2.0 Part D: NOT reset every tick like the flags above — this is a
    # "last known aim direction", sticky across ticks so a shot fired a
    # moment after the AIM gesture last updated still fires the intended
    # way, not back to a hardcoded default. Defaults to straight up, which
    # is the exact pre-Part-D fixed firing direction — a session with no
    # aim data at all (keyboard mode, or MediaPipe never seeing a pointing
    # hand) behaves identically to before this feature existed.
    last_aim_vector: AimVector = DEFAULT_AIM_VECTOR
    target_enemy_id: int | None = None  # recomputed each tick — which enemy the aim currently covers
    last_damage_time: float = 0.0  # V2.0 Part F — HUD damage-flash feedback, 0.0 = never damaged

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def take_damage(self, amount: int) -> int:
        """Applies status-based damage reduction. Returns actual damage taken."""
        multiplier = 1.0
        if self.shielded:
            multiplier -= SHIELD_DAMAGE_REDUCTION
        elif self.crouching:
            multiplier -= CROUCH_DAMAGE_REDUCTION
        actual = max(0, round(amount * multiplier))
        self.health = max(0, self.health - actual)
        if actual > 0:
            self.last_damage_time = time.time()
        return actual

    def heal(self, amount: int) -> None:
        self.health = min(self.max_health, self.health + amount)

    def reset_tick_status(self) -> None:
        self.shielded = False
        self.crouching = False
        self.aiming = False
        # last_aim_vector/target_enemy_id deliberately NOT reset here — see
        # their field comments above.

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "health": self.health,
            "max_health": self.max_health,
            "score": self.score,
            "shielded": self.shielded,
            "crouching": self.crouching,
            "aiming": self.aiming,
            "aim_vector": list(self.last_aim_vector),
            "target_enemy_id": self.target_enemy_id,
            "last_damage_time": round(self.last_damage_time, 3),
        }
