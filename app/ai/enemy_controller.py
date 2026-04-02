"""Bridges the GameEngine's world state to the FSM — Week 7, field-of-view
computation added in V2.0 Part E.

`GameEngine` doesn't know AI exists; it just accepts a `{enemy_id: state}`
dict each tick (see GameEngine.update's `enemy_actions` parameter). This
class is what builds that dict: measuring distance/health/visibility from
the engine's current entities and asking `state_machine.next_state` what
each enemy should do next. It's also the one place `Enemy.last_known_player_pos`
gets written — updated only on ticks where the player is actually visible,
which is what lets SEARCH move toward where the player *was*, not an
omniscient live position (see app/ai/state_machine.py module docstring).
"""

from __future__ import annotations

import math

from ..game.enemy import Enemy
from ..game.player import Player
from .state_machine import FOV_HALF_ANGLE_DEG, SURPRISE_RANGE_PX, EnemyPerception, next_state


class EnemyController:
    """Usage (once per tick, before GameEngine.update):
    actions = controller.decide(engine.enemies, engine.player)
    engine.update(dt, enemy_actions=actions)
    """

    def decide(self, enemies: list[Enemy], player: Player) -> dict[int, str]:
        actions: dict[int, str] = {}
        for enemy in enemies:
            dx, dy = player.x - enemy.x, player.y - enemy.y
            distance = math.hypot(dx, dy)
            visible = player.is_alive and self._can_see(enemy, dx, dy, distance)
            if visible:
                enemy.last_known_player_pos = (player.x, player.y)

            perception = EnemyPerception(
                distance_to_player=distance,
                enemy_health=enemy.health,
                enemy_max_health=enemy.max_health,
                player_alive=player.is_alive,
                player_visible=visible,
            )
            actions[enemy.enemy_id] = next_state(enemy.state, perception)
        return actions

    @staticmethod
    def _can_see(enemy: Enemy, dx: float, dy: float, distance: float) -> bool:
        """Distance-and-direction visibility: within `SURPRISE_RANGE_PX`
        regardless of facing, or within the forward-facing FOV cone at any
        distance (the FSM itself still gates by `DETECTION_RANGE_PX`;
        this only answers "is the player inside the enemy's cone of
        vision", not "is that within pursuit range")."""
        if distance <= SURPRISE_RANGE_PX:
            return True
        if distance < 1e-6:
            return True
        fx, fy = enemy.facing
        cos_angle = max(-1.0, min(1.0, (fx * dx + fy * dy) / distance))
        angle_deg = math.degrees(math.acos(cos_angle))
        return angle_deg <= FOV_HALF_ANGLE_DEG
