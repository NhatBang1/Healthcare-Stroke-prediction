from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Web").exists() and (parent / "Origin_code").exists():
            return parent
    # Docker image only copies the backend app, so fall back to the container root.
    return current.parents[2] if len(current.parents) > 2 else current.parent


REPO_ROOT = _resolve_repo_root()
WEB_ROOT = REPO_ROOT / "Web"
ORIGIN_ROOT = REPO_ROOT / "Origin_code"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=WEB_ROOT / "be" / ".env", env_file_encoding="utf-8")

    app_env: str = "development"
    app_name: str = "StrokeGuard AI Backend"
    static_dir: Path = WEB_ROOT / "fe" / "dist"
    artifact_dir: Path = ORIGIN_ROOT / "artifacts"
    frontend_public_url: str = "http://localhost:8000"
    api_public_url: str = "http://localhost:8000/api"
    google_client_id: str | None = None
    google_client_secret: str | None = None
    database_url: str | None = None
    redis_url: str | None = None
    chat_memory_ttl_seconds: int = 7 * 24 * 60 * 60
    admin_emails_raw: str = ""
    auth_token_cookie_name: str = "strokeguard_token"
    anonymous_owner_cookie_name: str = "strokeguard_owner"
    auth_token_expire_minutes: int = 24 * 60
    admin_portal_cookie_name: str = "strokeguard_admin_token"
    admin_portal_expire_minutes: int = 24 * 60
    cookie_secure: bool = False
    max_request_body_bytes: int = 32 * 1024
    general_rate_limit_requests: int = 120
    general_rate_limit_window_seconds: int = 60
    auth_rate_limit_requests: int = 5
    auth_rate_limit_window_seconds: int = 15 * 60
    clinical_model_path: Path = ORIGIN_ROOT / "artifacts" / "stroke_model_clinical.joblib"
    edge_model_path: Path = ORIGIN_ROOT / "artifacts" / "stroke_model.joblib"
    calories_model_path: Path = ORIGIN_ROOT / "artifacts" / "calories_model.joblib"
    preprocess_metadata_path: Path = ORIGIN_ROOT / "artifacts" / "preprocess_metadata.json"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = 30.0
    deepseek_strict_tools: bool = False
    tavily_api_key: str | None = None
    langgraph_checkpoint_enabled: bool = True
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:8080",
            "https://verbaa.pp.ua",
            "https://lilkremxxx.strokeg.pp.ua",
            "https://strokeg.lilkremxxx.pp.ua",
        ]
    )

    @field_validator("database_url", "redis_url", mode="before")
    @classmethod
    def empty_connection_url_is_unset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


settings = Settings()
