"""
Logging setup, called once at process startup by each entry point
(scripts/run_*.py). Every module below just does `logging.getLogger(__name__)`
and writes to it — nothing configures logging for itself, so behavior
(format, level, destination) stays consistent across the CLI vision runner,
the game process, and the API server.

RULE (see project engineering rules): log lifecycle events (startup, camera
opened, model loaded, WebSocket connect/disconnect, game start/end,
performance warnings) — never per-frame. A 30 FPS video loop that logs every
frame produces ~1800 lines/minute of noise that hides the one line that
actually matters.
"""

from __future__ import annotations

import logging

from .config import settings


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
