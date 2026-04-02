"""The playfield: bounds plus named rectangular zones — Week 5.

Deliberately the same "rectangle + name + kind" shape as VisionArena's zone
engine, because it's the same problem (is point P inside rectangle Z) with
one more field (a zone here affects gameplay — damage/heal/speed — rather
than scoring)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ZoneKind(str, Enum):
    SAFE = "SAFE"  # slow regen, no enemy damage bonus
    DANGER = "DANGER"  # damage-over-time while standing in it


@dataclass
class Zone:
    name: str
    kind: ZoneKind
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Arena:
    width: int = 1280
    height: int = 720
    zones: list[Zone] = field(default_factory=list)

    def zone_at(self, x: float, y: float) -> Zone | None:
        for zone in self.zones:
            if zone.contains(x, y):
                return zone
        return None

    def clamp(self, x: float, y: float, radius: float = 16.0) -> tuple[float, float]:
        """Keeps a position (and its collision radius) inside the arena bounds."""
        return (
            max(radius, min(self.width - radius, x)),
            max(radius, min(self.height - radius, y)),
        )


def default_arena(width: int = 1280, height: int = 720) -> Arena:
    """A safe corner to spawn in and a danger zone in the middle — enough
    to demonstrate zone-driven gameplay without hand-authoring a full level."""
    return Arena(
        width=width,
        height=height,
        zones=[
            Zone("Safe Corner", ZoneKind.SAFE, 0, height - 160, 220, height),
            Zone("Danger Core", ZoneKind.DANGER, width * 0.35, height * 0.35, width * 0.65, height * 0.65),
        ],
    )
