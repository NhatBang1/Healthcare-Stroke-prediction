from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .repositories import (
    ChatRepository,
    OwnerIdentity,
    PredictionRepository,
    ProfileRepository,
    ToolAuditRepository,
)
from app.extraction.schemas import ScreeningProfileUpdate


PROFILE_FIELDS = {
    "age",
    "gender",
    "systolic_bp",
    "diastolic_bp",
    "blood_pressure_category",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "pregnant",
    "smoke_flag",
    "alco_flag",
}


class _BoundConnectionPool:
    def __init__(self, connection) -> None:
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


class PostgresConversationStore:
    def __init__(
        self,
        pool,
        anonymous_token: str | None = None,
        *,
        owner: OwnerIdentity | None = None,
    ) -> None:
        if owner is None and anonymous_token is None:
            raise ValueError("An authenticated or anonymous owner is required")
        self.owner = owner or OwnerIdentity.anonymous(str(anonymous_token))
        self.chat = ChatRepository(pool)
        self.profiles = ProfileRepository(pool)
        self.predictions = PredictionRepository(pool)

    @staticmethod
    def new_session_id() -> str:
        return str(uuid4())

    @staticmethod
    def _id(session_id: str) -> UUID:
        return UUID(session_id)

    def ensure_session(self, session_id: str, first_message: str) -> None:
        title = " ".join(first_message.split())[:60] or "Cuộc hội thoại"
        if not self.chat.create_session(self._id(session_id), self.owner, title=title):
            raise PermissionError("Session is unavailable for this owner")

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        parsed_id = self._id(session_id)
        if not self.chat.owned_session_exists(parsed_id, self.owner):
            raise PermissionError("Session is unavailable for this owner")
        return [self._serialize_message(row) for row in self.chat.messages(parsed_id, self.owner)]

    def sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": str(row["session_id"]),
                "message_count": row["message_count"],
                "title": row["title"],
            }
            for row in self.chat.sessions(self.owner)
        ]

    def progress_messages(self, session_id: str) -> list[dict[str, Any]]:
        parsed_id = self._id(session_id)
        if not self.chat.owned_session_exists(parsed_id, self.owner):
            raise PermissionError("Session is unavailable for this owner")
        messages: list[dict[str, Any]] = []
        for row in self.predictions.list_for_session(parsed_id, self.owner):
            probability = float(row["risk_probability"])
            label = {
                "low_risk": "Thấp",
                "medium_risk": "Trung bình",
                "high_risk": "Cao",
            }.get(row["risk_label"], str(row["risk_label"]))
            result = row.get("result_payload") or {}
            messages.append(
                {
                    "role": "assistant",
                    "content": result.get("disclaimer", "Kết quả sàng lọc"),
                    "time": row["created_at"].astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M"),
                    "assistant_card": {
                        "model_used": row["model_name"],
                        "risk_score_10": min(10.0, max(0.0, probability * 100)),
                        "risk_label": label,
                        "summary": result.get("disclaimer", "Kết quả sàng lọc"),
                    },
                }
            )
        return messages

    def profile(self, session_id: str) -> dict[str, Any]:
        row = self.profiles.get(self._id(session_id), self.owner)
        return {field: row[field] for field in PROFILE_FIELDS if row.get(field) is not None}

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        result: dict[str, Any],
        *,
        reply_context: dict[str, Any] | None = None,
        media: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        parsed_id = self._id(session_id)
        with self.chat.pool.connection() as connection:
            bound_pool = _BoundConnectionPool(connection)
            chat = ChatRepository(bound_pool)
            profiles = ProfileRepository(bound_pool)
            predictions = PredictionRepository(bound_pool)
            audits = ToolAuditRepository(bound_pool)

            user_row = chat.add_message(
                parsed_id,
                self.owner,
                "user",
                user_message,
                metadata={"reply_context": reply_context} if reply_context is not None else None,
            )
            profile = {
                key: value
                for key, value in result.get("current_profile", {}).items()
                if key in PROFILE_FIELDS
            }
            raw_update = result.get("extracted_update")
            update = ScreeningProfileUpdate.model_validate(raw_update) if raw_update is not None else None
            profile_row = None
            if profile or (update is not None and update.explicit_unknown_fields):
                evidence = update.evidence_by_field() if update is not None else {}
                clear_fields = set(update.explicit_unknown_fields) if update is not None else set()
                profile_row = profiles.merge(
                    parsed_id,
                    self.owner,
                    profile,
                    evidence,
                    clear_fields=clear_fields,
                )
            prediction = result.get("prediction_result")
            if prediction:
                predictions.save(
                    parsed_id,
                    self.owner,
                    prediction,
                    profile_id=profile_row["id"] if profile_row else None,
                    source_message_id=user_row["id"],
                )
            if raw_update is not None or result.get("extraction_error"):
                audits.record(
                    session_id=parsed_id,
                    message_id=user_row["id"],
                    tool_name="extract_screening_profile",
                    status=(
                        result.get("extraction_error_type", "provider_error")
                        if result.get("extraction_error")
                        else "success"
                    ),
                    output_metadata={
                        "fields": sorted(update.profile_updates()) if update is not None else [],
                    },
                )
            chat.add_message(
                parsed_id,
                self.owner,
                "assistant",
                result["reply"],
                assistant_card=result.get("assistant_card"),
                metadata={"media": media} if media else None,
            )
        return self.messages(session_id)

    def rename(self, session_id: str, title: str) -> None:
        if not self.chat.rename(self._id(session_id), self.owner, title):
            raise PermissionError("Session is unavailable for this owner")

    def reset(self, session_id: str) -> None:
        parsed_id = self._id(session_id)
        if not self.chat.soft_delete(parsed_id, self.owner):
            raise PermissionError("Session is unavailable for this owner")
        self._delete_checkpoints(str(parsed_id))

    def _delete_checkpoints(self, thread_id: str) -> None:
        with self.chat.pool.connection() as connection:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                exists = connection.execute(
                    "SELECT to_regclass(%s) AS relation",
                    (f"public.{table}",),
                ).fetchone()["relation"]
                if exists is not None:
                    connection.execute(
                        f"DELETE FROM public.{table} WHERE thread_id = %s",
                        (thread_id,),
                    )

    @staticmethod
    def _serialize_message(row: dict[str, Any]) -> dict[str, Any]:
        created = row.get("created_at")
        if isinstance(created, datetime):
            timestamp = created.astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M")
        else:
            timestamp = str(created or "")
        message = {
            "id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "time": timestamp,
        }
        if row.get("assistant_card") is not None:
            message["assistant_card"] = row["assistant_card"]
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("reply_context"), dict):
            message["reply_context"] = metadata["reply_context"]
        if isinstance(metadata, dict) and isinstance(metadata.get("media"), list):
            message["media"] = metadata["media"]
        return message
