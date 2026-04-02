"""Projectiles — Week 5. Starts with one weapon type (Laser); EMP/shield
power-ups are a documented future improvement (README), not implemented
here, to keep the first playable version shippable."""

from __future__ import annotations

import math
from dataclasses import dataclass

PROJECTILE_SPEED = 480.0  # px/second
PROJECTILE_RADIUS = 5.0
SHOOT_COOLDOWN_SECONDS = 0.35  # caps fire rate regardless of how often SHOOT events arrive


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    owner: str  # "player" or an enemy_id as a string
    damage: int
    radius: float = PROJECTILE_RADIUS
    alive: bool = True

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

    def out_of_bounds(self, width: int, height: int) -> bool:
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height


def fire_laser(x: float, y: float, direction_deg: float, owner: str, damage: int) -> Projectile:
    """`direction_deg` is 0 = facing right (+x), increasing clockwise (screen
    coordinates, where +y is down) — matches Pygame's coordinate system."""
    rad = math.radians(direction_deg)
    return Projectile(
        x=x,
        y=y,
        vx=math.cos(rad) * PROJECTILE_SPEED,
        vy=math.sin(rad) * PROJECTILE_SPEED,
        owner=owner,
        damage=damage,
    )
