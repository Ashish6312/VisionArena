"""One row per game session — Week 6. `report_json` holds the full
PerformanceReport so the API can return everything without widening this
table every time analytics grows a new field; the individual columns exist
for the things worth querying/sorting on directly (leaderboards, "recent
sessions")."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class GameSessionRecord(Base):
    __tablename__ = "game_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    difficulty: Mapped[str] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    score: Mapped[int] = mapped_column(Integer, default=0)
    kills: Mapped[int] = mapped_column(Integer, default=0)
    shots_fired: Mapped[int] = mapped_column(Integer, default=0)
    shots_hit: Mapped[int] = mapped_column(Integer, default=0)
    accuracy_pct: Mapped[float] = mapped_column(Float, default=0.0)
    damage_taken: Mapped[int] = mapped_column(Integer, default=0)
    survival_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
