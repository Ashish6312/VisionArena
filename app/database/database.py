"""SQLAlchemy engine/session setup — Week 6.

SQLite by default (`settings.database_url`, see .env.example). Swapping to
Postgres later is a one-line env var change — nothing else in this module
or in repository.py is SQLite-specific, `check_same_thread` is the only
SQLite-only knob and it's applied conditionally below.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    pass


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from . import models  # noqa: F401 — import registers the ORM models on Base before create_all

    Base.metadata.create_all(bind=engine)
