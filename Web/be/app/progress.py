from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from starlette.requests import Request

from app.conversation_runtime import conversation_store
from app.schemas import ProgressCurrentRisk, ProgressEntry, ProgressSummary


router = APIRouter(prefix="/api/progress", tags=["progress"])


@dataclass
class RiskSample:
    score: float
    label: str
    note: str


def _now_label() -> str:
    return datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M")


def _normalize_score(score: float) -> float:
    return max(0.5, min(10.0, round(score, 1)))


def _label_for_score(score: float) -> str:
    if score < 4:
        return "Ổn định"
    if score < 7:
        return "Theo dõi"
    return "Cao"


def _trend_for_scores(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "Chờ dữ liệu"
    delta = current - previous
    if delta > 0.4:
        return "Tăng"
    if delta < -0.4:
        return "Giảm"
    return "Ổn định"


def _build_entries(messages: list[dict[str, Any]]) -> tuple[list[float], list[ProgressEntry]]:
    scores: list[float] = []
    entries: list[ProgressEntry] = []

    for message in messages:
        card = message.get("assistant_card")
        if message.get("role") != "assistant" or not isinstance(card, dict):
            continue
        if card.get("model_used") != "stroke_model_clinical.joblib":
            continue
        score = _normalize_score(float(card.get("risk_score_10", 0.0)))
        label = str(card.get("risk_label") or _label_for_score(score))
        scores.append(score)
        entries.append(
            ProgressEntry(
                source="model",
                label=label,
                note=str(card.get("summary") or message.get("content", ""))[:160],
                score=score,
                time=str(message.get("time") or _now_label()),
            )
        )

    return scores, list(reversed(entries))


def build_progress_summary(session_id: str, messages: list[dict[str, Any]] | None = None) -> ProgressSummary:
    chart_points, entries = _build_entries(messages or [])
    latest_score = chart_points[-1] if chart_points else None
    previous_score = chart_points[-2] if len(chart_points) > 1 else None

    current_risk = ProgressCurrentRisk(
        label=entries[0].label if entries else "Chờ dữ liệu",
        score=latest_score,
        trend=_trend_for_scores(latest_score, previous_score),
        updated_at=entries[0].time if entries else None,
    )

    return ProgressSummary(
        session_id=session_id,
        current_risk=current_risk,
        chart_points=chart_points,
        entries=entries,
    )


@router.get("/{session_id}", response_model=ProgressSummary)
def get_progress(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    try:
        store = conversation_store(request, response)
        messages = store.progress_messages(session_id) if hasattr(store, "progress_messages") else store.messages(session_id)
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return build_progress_summary(session_id, messages).model_dump()
