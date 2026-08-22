from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class OwnerIdentity:
    user_id: UUID | None = None
    anonymous_owner_hash: str | None = None

    def __post_init__(self) -> None:
        if bool(self.user_id) == bool(self.anonymous_owner_hash):
            raise ValueError("OwnerIdentity requires exactly one owner value")

    @classmethod
    def anonymous(cls, token: str) -> "OwnerIdentity":
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return cls(anonymous_owner_hash=digest)


class _OwnedRepository:
    def __init__(self, pool) -> None:
        self.pool = pool

    @staticmethod
    def _owner_params(owner: OwnerIdentity) -> tuple[Any, Any, Any, Any]:
        return (
            owner.user_id,
            owner.user_id,
            owner.anonymous_owner_hash,
            owner.anonymous_owner_hash,
        )

    @staticmethod
    def _owner_clause(alias: str = "s") -> str:
        return (
            f"((%s::uuid IS NOT NULL AND {alias}.user_id = %s::uuid) "
            f"OR (%s::text IS NOT NULL AND {alias}.anonymous_owner_hash = %s::text))"
        )


class AuthRepository:
    def __init__(self, pool) -> None:
        self.pool = pool

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM strokeguard.users WHERE email = %s",
                    (email.strip().lower(),),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def create_password_user(
        self,
        *,
        email: str,
        full_name: str,
        password_salt: str,
        password_hash: str,
        role: str = "user",
    ) -> dict[str, Any]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO strokeguard.users
                        (email, full_name, password_salt, password_hash, role)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (email.strip().lower(), full_name, password_salt, password_hash, role),
                )
                return dict(cursor.fetchone())

    def upsert_oauth_user(
        self,
        *,
        email: str,
        full_name: str,
        provider: str,
        provider_id: str,
        role: str = "user",
    ) -> dict[str, Any]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO strokeguard.users
                        (email, full_name, provider, provider_id, role)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                        full_name = EXCLUDED.full_name,
                        provider = COALESCE(strokeguard.users.provider, EXCLUDED.provider),
                        provider_id = COALESCE(strokeguard.users.provider_id, EXCLUDED.provider_id),
                        role = CASE WHEN strokeguard.users.role = 'admin' THEN 'admin' ELSE EXCLUDED.role END,
                        updated_at = now()
                    RETURNING *
                    """,
                    (email.strip().lower(), full_name, provider, provider_id, role),
                )
                return dict(cursor.fetchone())

    def create_auth_session(self, user_id: UUID, token: str, expires_at: datetime) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO strokeguard.auth_sessions (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, token_hash, expires_at),
            )

    def user_by_token(self, token: str) -> dict[str, Any] | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT u.*
                    FROM strokeguard.auth_sessions a
                    JOIN strokeguard.users u ON u.id = a.user_id
                    WHERE a.token_hash = %s AND a.expires_at > now()
                    """,
                    (token_hash,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def delete_auth_session(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.pool.connection() as connection:
            connection.execute(
                "DELETE FROM strokeguard.auth_sessions WHERE token_hash = %s",
                (token_hash,),
            )

class ChatRepository(_OwnedRepository):
    def owned_session_exists(self, session_id: UUID, owner: OwnerIdentity) -> bool:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT 1 FROM strokeguard.chat_sessions s
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                return cursor.fetchone() is not None

    def create_session(
        self,
        session_id: UUID,
        owner: OwnerIdentity,
        *,
        title: str = "Cuộc hội thoại",
    ) -> bool:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO strokeguard.chat_sessions
                        (id, user_id, anonymous_owner_hash, title)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    (session_id, owner.user_id, owner.anonymous_owner_hash, title),
                )
                inserted = cursor.fetchone()
                if inserted:
                    return True
                cursor.execute(
                    f"""
                    SELECT id FROM strokeguard.chat_sessions s
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                return cursor.fetchone() is not None

    def add_message(
        self,
        session_id: UUID,
        owner: OwnerIdentity,
        role: str,
        content: str,
        *,
        assistant_card: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO strokeguard.chat_messages
                        (session_id, role, content, assistant_card, metadata, created_at)
                    SELECT s.id, %s, %s, %s, %s, clock_timestamp()
                    FROM strokeguard.chat_sessions s
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    RETURNING id, session_id, role, content, assistant_card,
                              metadata, created_at
                    """,
                    (
                        role,
                        content,
                        Jsonb(assistant_card) if assistant_card is not None else None,
                        Jsonb(metadata or {}),
                        session_id,
                        *self._owner_params(owner),
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise PermissionError("Session is unavailable for this owner")
                cursor.execute(
                    """
                    UPDATE strokeguard.chat_sessions
                    SET updated_at = now()
                    WHERE id = %s
                    """,
                    (session_id,),
                )
                return dict(row)

    def messages(self, session_id: UUID, owner: OwnerIdentity) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT m.id, m.session_id, m.role, m.content,
                           m.assistant_card, m.metadata, m.created_at
                    FROM strokeguard.chat_messages m
                    JOIN strokeguard.chat_sessions s ON s.id = m.session_id
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    ORDER BY m.created_at, m.id
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                return [dict(row) for row in cursor.fetchall()]

    def sessions(self, owner: OwnerIdentity) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT s.id AS session_id, s.title, s.updated_at,
                           count(m.id)::integer AS message_count
                    FROM strokeguard.chat_sessions s
                    LEFT JOIN strokeguard.chat_messages m ON m.session_id = s.id
                    WHERE s.status <> 'deleted' AND {self._owner_clause()}
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    """,
                    self._owner_params(owner),
                )
                return [dict(row) for row in cursor.fetchall()]

    def rename(self, session_id: UUID, owner: OwnerIdentity, title: str) -> bool:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE strokeguard.chat_sessions s
                    SET title = %s, updated_at = now()
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (title, session_id, *self._owner_params(owner)),
                )
                return cursor.rowcount == 1

    def soft_delete(self, session_id: UUID, owner: OwnerIdentity) -> bool:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE strokeguard.chat_sessions s
                    SET status = 'deleted', deleted_at = now(), updated_at = now()
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                return cursor.rowcount == 1


class ProfileRepository(_OwnedRepository):
    _PROFILE_COLUMNS = (
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
        "avg_resting_bp",
        "avg_max_hr",
        "avg_oldpeak",
        "heart_disease_rate",
        "cardio_cholesterol",
        "cardio_gluc",
        "cardio_rate",
        "smoke_flag",
        "alco_flag",
        "ever_married",
        "work_type",
        "residence_type",
        "smoking_status",
    )

    def get(self, session_id: UUID, owner: OwnerIdentity) -> dict[str, Any]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT p.*
                    FROM strokeguard.patient_profiles p
                    JOIN strokeguard.chat_sessions s ON s.id = p.session_id
                    WHERE p.session_id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                row = cursor.fetchone()
                return dict(row) if row else {}

    def merge(
        self,
        session_id: UUID,
        owner: OwnerIdentity,
        updates: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        clear_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        clear_fields = clear_fields or set()
        unknown = (set(updates) | clear_fields) - set(self._PROFILE_COLUMNS)
        if unknown:
            raise ValueError(f"Unsupported profile fields: {sorted(unknown)}")

        current = self.get(session_id, owner)
        merged = {column: current.get(column) for column in self._PROFILE_COLUMNS}
        merged.update({key: value for key, value in updates.items() if value is not None})
        for field in clear_fields:
            merged[field] = None
        merged_evidence = dict(current.get("source_evidence") or {})
        merged_evidence.update(evidence or {})

        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT s.id FROM strokeguard.chat_sessions s
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                if cursor.fetchone() is None:
                    raise PermissionError("Session is unavailable for this owner")

                columns = ", ".join(self._PROFILE_COLUMNS)
                placeholders = ", ".join(["%s"] * len(self._PROFILE_COLUMNS))
                assignments = ", ".join(
                    f"{column} = EXCLUDED.{column}" for column in self._PROFILE_COLUMNS
                )
                cursor.execute(
                    f"""
                    INSERT INTO strokeguard.patient_profiles
                        (session_id, {columns}, source_evidence)
                    VALUES (%s, {placeholders}, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        {assignments},
                        source_evidence = EXCLUDED.source_evidence,
                        updated_at = now()
                    RETURNING *
                    """,
                    (
                        session_id,
                        *[merged[column] for column in self._PROFILE_COLUMNS],
                        Jsonb(merged_evidence),
                    ),
                )
                return dict(cursor.fetchone())


class PredictionRepository(_OwnedRepository):
    def save(
        self,
        session_id: UUID,
        owner: OwnerIdentity,
        prediction: dict[str, Any],
        *,
        profile_id: UUID | None = None,
        source_message_id: UUID | None = None,
    ) -> dict[str, Any]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO strokeguard.stroke_predictions
                        (session_id, profile_id, source_message_id, model_name,
                         model_version, risk_probability, risk_label, threshold,
                         input_features, filled_features, result_payload)
                    SELECT s.id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    FROM strokeguard.chat_sessions s
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    RETURNING *
                    """,
                    (
                        profile_id,
                        source_message_id,
                        prediction["model_used"],
                        prediction["model_version"],
                        prediction["risk_probability"],
                        prediction["risk_label"],
                        prediction["threshold"],
                        Jsonb(prediction.get("detected_features") or {}),
                        Jsonb(prediction.get("filled_features") or {}),
                        Jsonb(prediction),
                        session_id,
                        *self._owner_params(owner),
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise PermissionError("Session is unavailable for this owner")
                return dict(row)

    def list_for_session(
        self,
        session_id: UUID,
        owner: OwnerIdentity,
    ) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT p.*
                    FROM strokeguard.stroke_predictions p
                    JOIN strokeguard.chat_sessions s ON s.id = p.session_id
                    WHERE s.id = %s AND s.status <> 'deleted'
                      AND {self._owner_clause()}
                    ORDER BY p.created_at
                    """,
                    (session_id, *self._owner_params(owner)),
                )
                return [dict(row) for row in cursor.fetchall()]


class ToolAuditRepository(_OwnedRepository):
    def record(
        self,
        *,
        session_id: UUID,
        message_id: UUID | None,
        tool_name: str,
        status: str,
        input_metadata: dict[str, Any] | None = None,
        output_metadata: dict[str, Any] | None = None,
        error_detail: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO strokeguard.tool_audit_logs
                    (session_id, message_id, tool_name, status, input_metadata,
                     output_metadata, error_detail, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    message_id,
                    tool_name,
                    status,
                    Jsonb(input_metadata or {}),
                    Jsonb(output_metadata or {}),
                    Jsonb(error_detail or {}),
                    latency_ms,
                ),
            )
