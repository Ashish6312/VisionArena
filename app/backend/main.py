"""
FastAPI application factory — Week 6.

Ties together every other week: /game/* drives the GameRunner (engine +
AI + optionally the full CV/gesture pipeline, Weeks 1-5 and 7), /ws/game
streams its output live, /player and /analytics read the resulting state
and stored session reports.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .. import __version__
from ..database.database import init_db
from ..logging_config import setup_logging
from .game_runner import runner
from .routes.analytics import router as analytics_router
from .routes.game import router as game_router
from .routes.health import router as health_router
from .routes.player import router as player_router
from .websocket_manager import manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Record the server's running event loop so GameRunner's background
    # thread can safely schedule WebSocket broadcasts onto it — a worker
    # thread has no event loop of its own (see game_runner.py docstring).
    runner.bind_loop(asyncio.get_running_loop())
    init_db()
    logger.info("VisionStrike API started (version %s)", __version__)
    yield
    logger.info("VisionStrike API shutting down")


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="VisionStrike API", version=__version__, lifespan=_lifespan)
    app.include_router(health_router)
    app.include_router(game_router)
    app.include_router(player_router)
    app.include_router(analytics_router)

    @app.websocket("/ws/game")
    async def websocket_game(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()  # keep-alive; the dashboard doesn't send commands here
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app


app = create_app()
