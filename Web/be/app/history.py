from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from starlette.requests import Request

from app.conversation_runtime import conversation_store
from app.schemas import SessionHistorySummary


router = APIRouter(prefix="/api/sessions", tags=["history"])


@router.get("/{session_id}/history-summary", response_model=SessionHistorySummary)
def get_history_summary(session_id: str, request: Request, response: Response) -> dict[str, object]:
    try:
        messages = conversation_store(request, response).messages(session_id)
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "recent_messages": messages[-20:],
        "has_activity": bool(messages),
    }
