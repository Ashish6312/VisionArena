"""Collision detection — Week 5. Every entity in this game (player, enemy,
projectile) is a circle, so one function covers all pairs; no separate
rect-based physics needed."""

from __future__ import annotations


def circles_collide(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> bool:
    dx, dy = x1 - x2, y1 - y2
    distance_sq = dx * dx + dy * dy
    radius_sum = r1 + r2
    return distance_sq <= radius_sum * radius_sum
