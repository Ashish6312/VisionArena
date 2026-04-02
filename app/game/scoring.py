"""Scoring rules, isolated from engine.py so point values live in one place
(the same reasoning as VisionArena's game_engine: business rules shouldn't
be scattered through orchestration code)."""

from __future__ import annotations

from .state import GameState

POINTS_PER_HIT = 10
POINTS_PER_KILL = 100


def register_shot_fired(state: GameState) -> None:
    state.shots_fired += 1


def register_hit(state: GameState, killed: bool) -> None:
    state.shots_hit += 1
    state.score += POINTS_PER_HIT
    if killed:
        state.kills += 1
        state.score += POINTS_PER_KILL
