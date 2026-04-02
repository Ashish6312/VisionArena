"""Finished-session analytics — Week 6. Reads from the database, not from
GameRunner's live state — a session's report only exists once it has
actually ended (see GameRunner._finish, which is what writes it)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...database.repository import SessionRepository

router = APIRouter(prefix="/analytics", tags=["analytics"])
_repository = SessionRepository()


@router.get("/{session_id}")
def get_analytics(session_id: str) -> dict:
    record = _repository.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No session found with id {session_id}.")
    if record.report_json is not None:
        return record.report_json
    return {"session_id": session_id, "message": "Session is still in progress."}
