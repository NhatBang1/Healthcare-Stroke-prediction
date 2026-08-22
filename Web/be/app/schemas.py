from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ReplyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1, max_length=4000)

    @field_validator("message_id", "quote")
    @classmethod
    def normalize_reply_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reply context values cannot be empty")
        return cleaned


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    reply_context: ReplyContext | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not _SESSION_ID_PATTERN.fullmatch(cleaned):
            raise ValueError("session_id must contain only letters, digits, underscore, or hyphen")
        return cleaned


class ChatMessage(BaseModel):
    id: str | None = None
    role: str
    content: str
    time: str | None = None
    assistant_card: dict[str, Any] | None = None
    reply_context: ReplyContext | None = None
    media: list[dict[str, Any]] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float | None = None
    gender: str | int | float | None = None
    hypertension: int | bool | None = None
    heart_disease: int | bool | None = None
    avg_glucose_level: float | None = None
    bmi: float | None = None
    avg_RestingBP: float | None = None
    avg_MaxHR: float | None = None
    avg_Oldpeak: float | None = None
    heart_disease_rate: float | None = None
    cardio_cholesterol: float | None = None
    cardio_gluc: float | None = None
    cardio_rate: float | None = None
    smoke_flag: int | bool | None = None
    alco_flag: int | bool | None = None
    ever_married: str | None = None
    work_type: str | None = None
    Residence_type: str | None = None
    smoking_status: str | None = None

    model_config = {"populate_by_name": True}

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0 <= float(value) <= 100:
            raise ValueError("age must be between 0 and 100")
        return value

    @field_validator("avg_glucose_level")
    @classmethod
    def validate_avg_glucose_level(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 40 <= float(value) <= 400:
            raise ValueError("avg_glucose_level must be between 40 and 400")
        return value

    @field_validator("bmi")
    @classmethod
    def validate_bmi(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 10 <= float(value) <= 80:
            raise ValueError("bmi must be between 10 and 80")
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str | int | float | None) -> str | int | float | None:
        if value is None:
            return value
        if isinstance(value, (int, float, bool)):
            if float(value) not in {0.0, 1.0}:
                raise ValueError("numeric gender must be 0 or 1")
            return value
        cleaned = " ".join(str(value).split()).lower()
        allowed = {
            "male",
            "female",
            "nam",
            "nữ",
            "nu",
            "m",
            "f",
        }
        if cleaned not in allowed:
            raise ValueError("gender must be nam or nữ")
        return cleaned

    @field_validator("hypertension", "heart_disease")
    @classmethod
    def validate_binary_fields(cls, value: int | bool | None) -> int | bool | None:
        if value is not None and int(value) not in {0, 1}:
            raise ValueError("binary fields must be 0, 1, true, or false")
        return value


class PredictionResponse(BaseModel):
    model_used: str
    model_status: str
    model_version: str
    risk_probability: float
    risk_label: str
    threshold: float
    detected_features: dict[str, Any]
    filled_features: dict[str, Any]
    estimated_calories: float
    missing_user_fields: list[str]
    disclaimer: str


class AuthRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=120)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(cleaned):
            raise ValueError("invalid email format")
        return cleaned

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("full_name cannot be empty")
        return cleaned


class AuthLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(cleaned):
            raise ValueError("invalid email format")
        return cleaned

    @field_validator("password")
    @classmethod
    def normalize_password(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("password cannot be empty")
        return cleaned


class AdminLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("username cannot be empty")
        return cleaned

    @field_validator("password")
    @classmethod
    def normalize_admin_password(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("password cannot be empty")
        return cleaned


class AuthUser(BaseModel):
    email: str
    full_name: str
    is_admin: bool = False


class AuthResponse(AuthUser):
    access_token: str
    token_type: str = "bearer"


class AdminUser(BaseModel):
    username: str
    full_name: str
    is_admin: bool = True


class AdminAuthResponse(AdminUser):
    access_token: str
    token_type: str = "bearer"


class OAuthUrlResponse(BaseModel):
    url: str


class RagSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        return cleaned


class ProgressEntry(BaseModel):
    source: str
    label: str
    note: str
    score: float
    time: str


class ProgressCurrentRisk(BaseModel):
    label: str
    score: float | None = None
    trend: str = "Chờ dữ liệu"
    updated_at: str | None = None


class ProgressSummary(BaseModel):
    session_id: str
    current_risk: ProgressCurrentRisk
    chart_points: list[float]
    entries: list[ProgressEntry]


class AssistantCard(BaseModel):
    model_used: str
    model_status: str
    model_version: str
    verified: bool = True
    alert_level: str = "neutral"
    risk_probability_percent: float
    risk_score_10: float
    risk_label: str
    confidence_percent: float | None = None
    missing_fields: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    feature_cards: list[dict[str, str]] = Field(default_factory=list)
    alert_message: str | None = None
    disclaimer: str
    summary: str
    interpretation: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    messages: list[ChatMessage]
    assistant_card: AssistantCard | None = None


class SessionHistorySummary(BaseModel):
    session_id: str
    message_count: int
    recent_messages: list[ChatMessage]
    has_activity: bool


class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    title: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class RenameSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=80)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("title cannot be empty")
        return cleaned
