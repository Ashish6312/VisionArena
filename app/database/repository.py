"""Data-access layer — Week 6. Routes and the game runner talk to this,
never to a SQLAlchemy Session directly (engineering rule: no business/data
logic in API routes)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..analytics.performance import PerformanceReport
from .database import SessionLocal
from .models import GameSessionRecord


class SessionRepository:
    def __init__(self, session_factory: Callable[[], Session] = SessionLocal):
        self._session_factory = session_factory

    def create(self, session_id: str, difficulty: str) -> None:
        with self._session_factory() as db:
            db.add(GameSessionRecord(session_id=session_id, difficulty=difficulty))
            db.commit()

    def finish(self, session_id: str, report: PerformanceReport) -> None:
        with self._session_factory() as db:
            record = db.get(GameSessionRecord, session_id)
            if record is None:
                return
            record.ended_at = datetime.now(UTC)
            record.score = report.score
            record.kills = report.kills
            record.shots_fired = report.shots_fired
            record.shots_hit = report.shots_hit
            record.accuracy_pct = report.accuracy_pct
            record.damage_taken = report.damage_taken
            record.survival_seconds = report.survival_seconds
            record.report_json = report.to_dict()
            db.commit()

    def get(self, session_id: str) -> GameSessionRecord | None:
        with self._session_factory() as db:
            return db.get(GameSessionRecord, session_id)

    def list_recent(self, limit: int = 20) -> list[GameSessionRecord]:
        with self._session_factory() as db:
            return (
                db.query(GameSessionRecord).order_by(GameSessionRecord.started_at.desc()).limit(limit).all()
            )
