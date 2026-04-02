"""Game lifecycle + live state — Week 6. Routes stay thin: they translate
HTTP <-> GameRunner calls and turn RuntimeError into the right HTTP status,
nothing else (engineering rule: no business logic in routes)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..game_runner import runner
from ..schemas import GameStartRequest, GameStartResponse

router = APIRouter(prefix="/game", tags=["game"])


@router.post("/start", response_model=GameStartResponse)
def start_game(req: GameStartRequest) -> GameStartResponse:
    try:
        session_id = runner.start(req.difficulty, req.use_camera)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return GameStartResponse(session_id=session_id, message="Game starting")


@router.post("/stop")
def stop_game() -> dict:
    if not runner.running:
        raise HTTPException(status_code=400, detail="No game session is running.")
    runner.stop()
    return {"message": "Stop signal sent"}


@router.get("/state")
def get_state() -> dict:
    return runner.state()
