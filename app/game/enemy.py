"""Enemy entity — Week 5, patrol/FOV/last-known-position added in V2.0
Part E. Movement/behavior *decisions* come from ai/ (Week 7/V2.0 Part E);
this class only holds state and executes whatever it's told — the FSM and
`EnemyController` decide, `Enemy` just moves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Waypoint = tuple[float, float]


@dataclass
class Enemy:
    enemy_id: int
    x: float
    y: float
    health: int
    max_health: int
    speed: float  # px/second
    damage: int
    radius: float = 16.0
    state: str = "PATROL"  # set by ai/state_machine.py each tick
    last_attack_time: float = 0.0

    # V2.0 Part E
    waypoints: list[Waypoint] = field(default_factory=list)
    waypoint_index: int = 0
    facing: tuple[float, float] = (0.0, 1.0)  # unit vector; updated by every movement method below
    last_known_player_pos: tuple[float, float] | None = None  # set only while the player is actually visible

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def take_damage(self, amount: int) -> None:
        self.health = max(0, self.health - amount)

    def step_towards(self, target_x: float, target_y: float, dt: float) -> None:
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        self.facing = (dx / dist, dy / dist)
        step = min(dist, self.speed * dt)
        self.x += self.facing[0] * step
        self.y += self.facing[1] * step

    def patrol_step(self, dt: float) -> None:
        """Cycles through `waypoints` in order, looping back to the first
        once the last is reached — A -> B -> C -> D -> A. An enemy with no
        waypoints simply holds position (same as the pre-Part-E behavior),
        not an error: not every enemy needs a patrol route."""
        if not self.waypoints:
            return
        target = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if math.hypot(target[0] - self.x, target[1] - self.y) < 5.0:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            return
        self.step_towards(target[0], target[1], dt)

    def to_dict(self) -> dict:
        return {
            "enemy_id": self.enemy_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "health": self.health,
            "max_health": self.max_health,
            "state": self.state,
        }
