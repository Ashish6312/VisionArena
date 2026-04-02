"""The finished per-session report — Week 7. One dataclass every consumer
(the API's GET /analytics/{id}, the database repository, reports.py's text
formatter) reads from, so the report's shape is defined exactly once."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..game.state import GameState
from .metrics import accuracy
from .reaction import compute_reaction_stats
from .trajectory_analysis import MovementSummary


@dataclass
class PerformanceReport:
    session_id: str
    score: int
    kills: int
    shots_fired: int
    shots_hit: int
    accuracy_pct: float
    damage_taken: int
    survival_seconds: float
    movement: MovementSummary
    # reaction_time_avg_s is kept as its own field (not folded into a nested
    # object) for backward compatibility — it shipped in Week 7 and existing
    # consumers (tests, the API response shape) read it directly. The new
    # min/max/median/sample-count fields are purely additive.
    reaction_time_avg_s: float
    reaction_time_min_s: float = 0.0
    reaction_time_max_s: float = 0.0
    reaction_time_median_s: float = 0.0
    reaction_time_samples: int = 0
    gesture_counts: dict[str, int] = field(default_factory=dict)
    zone_time: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "score": self.score,
            "kills": self.kills,
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "accuracy_pct": self.accuracy_pct,
            "damage_taken": self.damage_taken,
            "survival_seconds": round(self.survival_seconds, 1),
            "movement": {
                "distance_traveled_px": round(self.movement.distance_traveled_px, 1),
                "average_speed_px_s": round(self.movement.average_speed_px_s, 1),
                "max_speed_px_s": round(self.movement.max_speed_px_s, 1),
                "direction_changes": self.movement.direction_changes,
                "stationary_seconds": round(self.movement.stationary_seconds, 1),
            },
            "reaction_time_avg_s": self.reaction_time_avg_s,
            "reaction_time_min_s": self.reaction_time_min_s,
            "reaction_time_max_s": self.reaction_time_max_s,
            "reaction_time_median_s": self.reaction_time_median_s,
            "reaction_time_samples": self.reaction_time_samples,
            "gesture_counts": self.gesture_counts,
            "zone_time": self.zone_time,
        }


def build_report(
    session_id: str,
    state: GameState,
    damage_taken: int,
    movement: MovementSummary,
    reaction_times: list[float],
    gesture_counts: dict[str, int],
    zone_time: dict[str, float],
) -> PerformanceReport:
    reaction = compute_reaction_stats(reaction_times)
    return PerformanceReport(
        session_id=session_id,
        score=state.score,
        kills=state.kills,
        shots_fired=state.shots_fired,
        shots_hit=state.shots_hit,
        accuracy_pct=accuracy(state.shots_hit, state.shots_fired),
        damage_taken=damage_taken,
        survival_seconds=state.elapsed_seconds,
        movement=movement,
        reaction_time_avg_s=reaction.average_s,
        reaction_time_min_s=reaction.minimum_s,
        reaction_time_max_s=reaction.maximum_s,
        reaction_time_median_s=reaction.median_s,
        reaction_time_samples=reaction.count,
        gesture_counts=gesture_counts,
        zone_time=zone_time,
    )
