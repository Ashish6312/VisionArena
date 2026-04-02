"""Game-wide state: status, difficulty, score/timer — Week 5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GameStatus(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    WON = "WON"
    LOST = "LOST"


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


@dataclass
class DifficultyProfile:
    enemy_count: int
    enemy_speed: float
    enemy_health: int
    enemy_damage: int


DIFFICULTY_PROFILES: dict[Difficulty, DifficultyProfile] = {
    Difficulty.EASY: DifficultyProfile(enemy_count=2, enemy_speed=60.0, enemy_health=40, enemy_damage=5),
    Difficulty.MEDIUM: DifficultyProfile(enemy_count=3, enemy_speed=90.0, enemy_health=60, enemy_damage=10),
    Difficulty.HARD: DifficultyProfile(enemy_count=5, enemy_speed=130.0, enemy_health=80, enemy_damage=15),
}


@dataclass
class GameState:
    status: GameStatus = GameStatus.RUNNING
    difficulty: Difficulty = Difficulty.MEDIUM
    score: int = 0
    kills: int = 0
    shots_fired: int = 0
    shots_hit: int = 0
    elapsed_seconds: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.status == GameStatus.RUNNING

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "difficulty": self.difficulty.value,
            "score": self.score,
            "kills": self.kills,
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "accuracy": round(self.shots_hit / self.shots_fired * 100, 1) if self.shots_fired else 0.0,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
