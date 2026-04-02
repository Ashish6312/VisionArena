"""
Aim-direction geometry — V2.0 Part D.

A 2D approximation, not physically accurate 3D aiming — stated plainly,
not oversold. A direction vector `(dx, dy)` in the SAME coordinate space
as the arena (screen space: +x right, +y down), derived upstream from
MediaPipe pose elbow->wrist geometry
(`app/gestures/features.py::arm_aim_vector`) or left at the default
straight-up direction in keyboard mode / when no pose is available. This
module has zero MediaPipe dependency — only vector math — which is what
makes it testable without a camera (see tests/test_aim.py).
"""

from __future__ import annotations

import math

AimVector = tuple[float, float]

DEFAULT_AIM_VECTOR: AimVector = (0.0, -1.0)  # straight up — the pre-Part-D fixed firing direction


def direction_degrees(vector: AimVector) -> float:
    """Converts a direction vector to degrees in the convention
    `app/game/weapons.py::fire_laser` expects: 0 = facing +x (right),
    increasing clockwise (screen coordinates, +y is down)."""
    dx, dy = vector
    return math.degrees(math.atan2(dy, dx))


def normalize(vector: AimVector) -> AimVector | None:
    """None for a degenerate (near-zero-length) vector — e.g. elbow and
    wrist reported at the same point — rather than dividing by ~0."""
    dx, dy = vector
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return (dx / length, dy / length)


def find_aim_target(
    origin: tuple[float, float],
    vector: AimVector,
    targets: list[tuple[int, float, float, float]],
    max_distance: float = 2000.0,
    cone_half_angle_deg: float = 12.0,
) -> int | None:
    """Which target (if any) the aim direction currently points at: a
    narrow cone from `origin` along `vector`, nearest qualifying target
    wins. `targets` is `[(id, x, y, radius), ...]`.

    This answers "what is the reticle over", for the HUD's target-
    intersection indicator — it does not fire anything or deal damage.
    The actual hit/miss mechanic stays the existing projectile-travel
    collision system (`GameEngine._resolve_collisions`); this function
    only decides which direction the projectile leaves in
    (`GameEngine._try_shoot`) and what the HUD highlights.
    """
    unit = normalize(vector)
    if unit is None:
        return None
    ux, uy = unit

    best_id: int | None = None
    best_dist: float | None = None
    for target_id, tx, ty, radius in targets:
        dx, dy = tx - origin[0], ty - origin[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist > max_distance + radius:
            continue

        target_ux, target_uy = dx / dist, dy / dist
        cos_angle = max(-1.0, min(1.0, ux * target_ux + uy * target_uy))
        angle_deg = math.degrees(math.acos(cos_angle))

        # Widen the effective cone for closer/larger targets, so a
        # dead-center aim at a nearby enemy isn't rejected by float noise
        # or a strict angle threshold that ignores target size.
        effective_half_angle = cone_half_angle_deg + math.degrees(math.atan2(radius, max(dist, 1.0)))
        if angle_deg <= effective_half_angle and (best_dist is None or dist < best_dist):
            best_id, best_dist = target_id, dist

    return best_id
