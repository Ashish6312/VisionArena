"""Pydantic request/response models — Week 6. Validated at the boundary so
a malformed request 422s here, not three layers deep in the game engine."""

from __future__ import annotations

from pydantic import BaseModel

from ..game.state import Difficulty


class GameStartRequest(BaseModel):
    difficulty: Difficulty = Difficulty.MEDIUM
    use_camera: bool = False


class GameStartResponse(BaseModel):
    session_id: str
    message: str
