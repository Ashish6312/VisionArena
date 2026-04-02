"""Live player state — Week 6. Single-player game: player_id 1 is always
the (only) player; anything else 404s rather than silently returning
nothing, so a typo'd ID fails loudly."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..game_runner import runner

router = APIRouter(prefix="/player", tags=["player"])


@router.get("/{player_id}")
def get_player(player_id: int) -> dict:
    if player_id != 1:
        raise HTTPException(status_code=404, detail=f"No player with id {player_id}.")
    if runner.engine is None:
        raise HTTPException(status_code=404, detail="No game session has been started yet.")
    return runner.engine.player.to_dict()
