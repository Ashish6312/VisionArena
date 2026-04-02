"""
Enemy finite state machine — Week 7, field-of-view + last-known-position
added in V2.0 Part E.

Deterministic, explainable decisions — no randomness. Every transition is
a plain comparison against distance/health/visibility, so "why did the
enemy attack right then" always has a one-sentence answer. This is the
baseline the RL-ready interface in policies.py is measured against, not a
placeholder for it — a rule-based FSM is often the *better* choice for a
small, fully-specified game like this one, not just the easier one.

STATES
    PATROL   default; player not visible and not recently seen
    SEARCH   player was visible recently (this state or CHASE/ATTACK) but
             isn't right now — moves toward EnemyController's last-known
             position, not the player's true live position (see
             app/game/enemy.py::last_known_player_pos)
    CHASE    player currently visible, not yet in attack range
    ATTACK   player currently visible and within attack range
    RETREAT  own health below the flee threshold — overrides all of the above
    DEAD     health <= 0 (terminal)

FIELD OF VIEW (V2.0 Part E): "visible" is not just "within
DETECTION_RANGE_PX" — `EnemyPerception.player_visible` is computed by
`EnemyController` from BOTH distance and whether the player falls within
a forward-facing cone (`FOV_HALF_ANGLE_DEG`) of the enemy's current facing
direction, with a small `SURPRISE_RANGE_PX` bypass (a player standing
right next to an enemy is noticed regardless of which way it's facing —
matches how most games treat point-blank proximity, and avoids the
absurdity of an enemy the player is stepping on failing to react).
"""

from __future__ import annotations

from dataclasses import dataclass

DETECTION_RANGE_PX = 260.0
SEARCH_BAND_PX = 80.0  # how far beyond detection range a SEARCH is still allowed to continue
ATTACK_RANGE_PX = 60.0
RETREAT_HEALTH_FRACTION = 0.25

FOV_HALF_ANGLE_DEG = 100.0  # cone half-angle the enemy can "see" within, centered on its facing direction
SURPRISE_RANGE_PX = 50.0  # regardless of facing, a player this close is always noticed

_ACTIVE_PURSUIT_STATES = ("CHASE", "ATTACK", "SEARCH")


@dataclass(frozen=True)
class EnemyPerception:
    """What the FSM is allowed to base a decision on — kept explicit and
    narrow so "what does the AI know" is answerable by reading one
    dataclass. `player_visible` (not just distance) is what makes field of
    view a real constraint instead of omniscient detection."""

    distance_to_player: float
    enemy_health: int
    enemy_max_health: int
    player_alive: bool
    player_visible: bool = True  # default True preserves pre-Part-E (omniscient-detection) behavior


def next_state(current_state: str, perception: EnemyPerception) -> str:
    if perception.enemy_health <= 0:
        return "DEAD"
    if current_state == "DEAD":
        return "DEAD"  # terminal — no resurrection
    if not perception.player_alive:
        return "PATROL"

    health_fraction = perception.enemy_health / perception.enemy_max_health
    if health_fraction < RETREAT_HEALTH_FRACTION:
        return "RETREAT"

    d = perception.distance_to_player

    if perception.player_visible:
        if d <= ATTACK_RANGE_PX:
            return "ATTACK"
        if d <= DETECTION_RANGE_PX:
            return "CHASE"
        # Visible but far (FOV caught them at range) — pretend the FSM
        # hasn't detected them yet unless already actively engaged; a
        # distant, barely-in-cone glimpse shouldn't yank a patrolling
        # enemy into a chase from arbitrary range.
        if current_state in _ACTIVE_PURSUIT_STATES:
            return "SEARCH"
        return "PATROL"

    # Not currently visible (out of FOV, or too far even accounting for
    # the surprise-range bypass): only keep searching if there's an active
    # pursuit to lose — a PATROL enemy that never saw the player has
    # nothing to "search" for and just keeps patrolling.
    if current_state in _ACTIVE_PURSUIT_STATES and d <= DETECTION_RANGE_PX + SEARCH_BAND_PX:
        return "SEARCH"
    return "PATROL"
