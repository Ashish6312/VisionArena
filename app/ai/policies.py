"""
RL-ready interface — Week 7 (deliberately NOT implemented).

Per the project's build rules: don't implement reinforcement learning
until the rule-based FSM works, and keep RL experimental and isolated so
it can never silently become a dependency of the shipping game. This
module is the seam a future RL policy would plug into — same State/Action
shape the rule-based `EnemyController` already works with — without
committing to training one now.

WHY DEFINE THE INTERFACE NOW ANYWAY: it's evidence the FSM's inputs
(distance, health, player state) were chosen deliberately as "the state a
policy would need," not as whatever happened to be convenient for if/else
logic. `RuleBasedPolicy` below proves the interface fits the system that
already ships; it wraps `state_machine.next_state` with zero new logic.

RULE-BASED vs REINFORCEMENT LEARNING, the actual trade-off:
  Rule-based (shipped): every decision is a one-line, human-checkable
  reason. No training data, no training time, correct on day one. Ceiling
  is whatever the designer thought to encode.
  RL (not shipped): can discover strategies no one wrote by hand, and
  adapts to a specific player's habits — but needs a reward function that
  actually captures "good play," thousands of episodes to converge, and
  produces decisions that are hard to explain when one looks wrong. For a
  4-state enemy in a small arena, the ceiling RL buys isn't worth that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..game.enemy import Enemy
from ..game.player import Player
from .state_machine import EnemyPerception, next_state


class Action(str, Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    ATTACK = "ATTACK"
    RETREAT = "RETREAT"
    HIDE = "HIDE"


@dataclass(frozen=True)
class RLState:
    """The observation an RL policy would receive — deliberately the same
    facts `EnemyPerception` already exposes to the rule-based FSM, plus
    position (a policy choosing MOVE_* needs coordinates; next_state() only
    ever needed distance)."""

    player_x: float
    player_y: float
    enemy_x: float
    enemy_y: float
    distance: float
    enemy_health: int
    player_health: int


class EnemyPolicy(Protocol):
    """Anything that can answer "what should this enemy do" satisfies this
    — the rule-based FSM and a future trained policy are interchangeable
    behind it."""

    def act(self, state: RLState) -> Action: ...


class RuleBasedPolicy:
    """Wraps state_machine.next_state in the Action interface — proves the
    interface fits without introducing any new decision logic."""

    def act(self, state: RLState) -> Action:
        fsm_state = next_state(
            "PATROL",
            EnemyPerception(
                distance_to_player=state.distance,
                enemy_health=state.enemy_health,
                enemy_max_health=max(state.enemy_health, 1),
                player_alive=state.player_health > 0,
            ),
        )
        return {"ATTACK": Action.ATTACK, "RETREAT": Action.RETREAT}.get(
            fsm_state, Action.MOVE_LEFT if state.enemy_x > state.player_x else Action.MOVE_RIGHT
        )


def reward(*, damaged_player: bool, took_damage: bool, died: bool, survived_tick: bool) -> float:
    """Reference reward shape for a future training run — not called by
    anything that ships. Positive for landing damage and staying alive,
    negative for taking damage or dying, matching the spec this project
    was scoped against."""
    r = 0.0
    if damaged_player:
        r += 10.0
    if survived_tick:
        r += 1.0
    if took_damage:
        r -= 10.0
    if died:
        r -= 20.0
    return r


def to_rl_state(enemy: Enemy, player: Player) -> RLState:
    """Adapter from live game entities to the observation shape above —
    this is the one piece of real, used code in this module: both the
    reference policy and any future trained one need it."""
    import math

    return RLState(
        player_x=player.x,
        player_y=player.y,
        enemy_x=enemy.x,
        enemy_y=enemy.y,
        distance=math.hypot(enemy.x - player.x, enemy.y - player.y),
        enemy_health=enemy.health,
        player_health=player.health,
    )
