from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import settings
from app.persistence.database import database_available, get_pool
from app.persistence.repositories import AuthRepository
from app.schemas import (
    AdminAuthResponse,
    AdminLoginRequest,
    AdminUser,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    AuthUser,
    OAuthUrlResponse,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

_PBKDF2_ROUNDS = 120_000
_TOKEN_PREFIX = "sg_"


@dataclass
class UserRecord:
    id: UUID | None
    email: str
    full_name: str
    password_salt: str
    password_hash: str
    is_admin: bool = False
    provider: str | None = None
    provider_id: str | None = None
    created_at: str = ""


@dataclass
class AdminPortalRecord:
    username: str
    full_name: str
    expires_at: datetime


_users: dict[str, UserRecord] = {}
_tokens: dict[str, tuple[str, datetime]] = {}
_admin_tokens: dict[str, AdminPortalRecord] = {}


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=settings.auth_token_expire_minutes)


def _admin_token_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=settings.admin_portal_expire_minutes)


def _admin_emails() -> set[str]:
    return {email.strip().lower() for email in settings.admin_emails_raw.split(",") if email.strip()}


def _is_admin_email(email: str) -> bool:
    return _normalize_email(email) in _admin_emails()


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    _, computed = _hash_password(password, salt_hex)
    return hmac.compare_digest(computed, expected_hash)


def _issue_token(email: str) -> str:
    token = f"{_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    expires_at = _token_expires_at()
    if database_available():
        user = _get_user_by_email(email)
        if not user or user.id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        AuthRepository(get_pool()).create_auth_session(user.id, token, expires_at)
    else:
        _tokens[token] = (_normalize_email(email), expires_at)
    return token


def _user_to_auth_response(user: UserRecord, token: str) -> AuthResponse:
    return AuthResponse(
        email=user.email,
        full_name=user.full_name,
        is_admin=user.is_admin,
        access_token=token,
        token_type="bearer",
    )


def _user_to_auth_user(user: UserRecord) -> AuthUser:
    return AuthUser(email=user.email, full_name=user.full_name, is_admin=user.is_admin)


def _admin_user_to_auth_response(record: AdminPortalRecord, token: str) -> AdminAuthResponse:
    return AdminAuthResponse(
        username=record.username,
        full_name=record.full_name,
        is_admin=True,
        access_token=token,
        token_type="bearer",
    )


def _admin_user_to_user(record: AdminPortalRecord) -> AdminUser:
    return AdminUser(username=record.username, full_name=record.full_name, is_admin=True)


def _get_user_by_email(email: str) -> UserRecord | None:
    normalized = _normalize_email(email)
    if database_available():
        row = AuthRepository(get_pool()).get_user_by_email(normalized)
        return _row_to_user(row) if row else None
    return _users.get(normalized)


def _row_to_user(row: dict[str, Any]) -> UserRecord:
    return UserRecord(
        id=row.get("id"),
        email=str(row["email"]),
        full_name=str(row["full_name"]),
        password_salt=str(row.get("password_salt") or ""),
        password_hash=str(row.get("password_hash") or ""),
        is_admin=row.get("role") == "admin",
        provider=row.get("provider"),
        provider_id=row.get("provider_id"),
        created_at=str(row.get("created_at") or ""),
    )


def _issue_admin_token(username: str) -> str:
    token = f"sg_admin_{secrets.token_urlsafe(32)}"
    _admin_tokens[token] = AdminPortalRecord(
        username=username,
        full_name="Admin",
        expires_at=_admin_token_expires_at(),
    )
    return token


def _get_admin_portal_by_token(token: str) -> AdminPortalRecord:
    record = _admin_tokens.get(token)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")

    if datetime.now(timezone.utc) >= record.expires_at:
        _admin_tokens.pop(token, None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin token expired")
    return record


def _get_current_user_from_request(request: Request) -> UserRecord:
    authorization = request.headers.get("authorization", "")
    token = None
    if authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get(settings.auth_token_cookie_name)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    return _get_user_by_token(token)


def get_optional_current_user(request: Request) -> UserRecord | None:
    try:
        return _get_current_user_from_request(request)
    except HTTPException:
        return None


def get_optional_auth_user(request: Request) -> AuthUser | None:
    user = get_optional_current_user(request)
    return _user_to_auth_user(user) if user else None


def get_optional_admin_portal_user(request: Request) -> AdminUser | None:
    authorization = request.headers.get("authorization", "")
    token = None
    if authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get(settings.admin_portal_cookie_name)
    if not token:
        return None
    record = _get_admin_portal_by_token(token)
    return _admin_user_to_user(record)


def _get_user_by_token(token: str) -> UserRecord:
    if database_available():
        row = AuthRepository(get_pool()).user_by_token(token)
        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        return _row_to_user(row)
    record = _tokens.get(token)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    email, expires_at = record
    if datetime.now(timezone.utc) >= expires_at:
        _tokens.pop(token, None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")

    user = _get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def _get_admin_portal_by_request(request: Request) -> AdminPortalRecord:
    admin_user = get_optional_admin_portal_user(request)
    if admin_user:
        authorization = request.headers.get("authorization", "")
        token = None
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
        if not token:
            token = request.cookies.get(settings.admin_portal_cookie_name)
        if token:
            return _get_admin_portal_by_token(token)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin access required")


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_token_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.auth_token_expire_minutes * 60,
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.auth_token_cookie_name, path="/")


def _set_admin_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.admin_portal_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.admin_portal_expire_minutes * 60,
        path="/",
    )


def _clear_admin_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.admin_portal_cookie_name, path="/")


def _clear_token(token: str | None) -> None:
    if token:
        if database_available():
            AuthRepository(get_pool()).delete_auth_session(token)
        else:
            _tokens.pop(token, None)


def _clear_admin_token(token: str | None) -> None:
    if token:
        _admin_tokens.pop(token, None)


def _frontend_redirect(path: str, **params: str) -> str:
    base = settings.frontend_public_url.rstrip("/")
    query = urlencode({key: value for key, value in params.items() if value})
    return f"{base}{path}" + (f"?{query}" if query else "")


def require_admin_access(request: Request) -> None:
    user = get_optional_current_user(request)
    if user and user.is_admin:
        return
    if get_optional_admin_portal_user(request):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin access required")


def _google_callback_url() -> str:
    return f"{settings.api_public_url.rstrip('/')}/auth/oauth/google/callback"


def _google_authorize_url(mode: str) -> str:
    redirect_uri = _google_callback_url()
    params = {
        "client_id": settings.google_client_id or "",
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": mode,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def _normalize_mode(mode: str | None) -> str:
    return "signup" if (mode or "").strip().lower() == "signup" else "login"


def _ensure_user(email: str, full_name: str, password: str, *, is_admin: bool = False) -> UserRecord:
    normalized_email = _normalize_email(email)
    existing = _get_user_by_email(normalized_email)
    salt_hex, password_hash = _hash_password(password)

    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    role_is_admin = is_admin or _is_admin_email(normalized_email)
    record = UserRecord(
        id=None,
        email=normalized_email,
        full_name=full_name.strip() or normalized_email,
        password_salt=salt_hex,
        password_hash=password_hash,
        is_admin=role_is_admin,
        created_at=_now(),
    )
    if database_available():
        row = AuthRepository(get_pool()).create_password_user(
            email=record.email,
            full_name=record.full_name,
            password_salt=record.password_salt,
            password_hash=record.password_hash,
            role="admin" if role_is_admin else "user",
        )
        record = _row_to_user(row)
    else:
        _users[normalized_email] = record
    return record


def _create_auth_response(user: UserRecord) -> AuthResponse:
    token = _issue_token(user.email)
    return _user_to_auth_response(user, token)


def require_admin_user(request: Request) -> UserRecord:
    user = _get_current_user_from_request(request)
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


@router.post("/register", response_model=AuthResponse)
def register(payload: AuthRegisterRequest, response: Response) -> AuthResponse:
    user = _ensure_user(payload.email, payload.full_name, payload.password)
    auth_response = _create_auth_response(user)
    _set_auth_cookie(response, auth_response.access_token)
    return auth_response


@router.post("/login", response_model=AuthResponse)
def login(payload: AuthLoginRequest, response: Response) -> AuthResponse:
    user = _get_user_by_email(payload.email)
    if not user or not _verify_password(payload.password, user.password_salt, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    user.is_admin = user.is_admin or _is_admin_email(user.email)
    auth_response = _create_auth_response(user)
    _set_auth_cookie(response, auth_response.access_token)
    return auth_response


@router.get("/me", response_model=AuthUser)
def me(request: Request) -> AuthUser:
    user = _get_current_user_from_request(request)
    return _user_to_auth_user(user)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    authorization = request.headers.get("authorization", "")
    token = None
    if authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get(settings.auth_token_cookie_name)
    _clear_token(token)
    _clear_auth_cookie(response)
    return {"ok": True}


@admin_router.post("/login", response_model=AdminAuthResponse)
def admin_login(payload: AdminLoginRequest, response: Response) -> AdminAuthResponse:
    username = payload.username.strip().lower()
    password = payload.password.strip()
    if username != "admin" or password != "admin":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")

    token = _issue_admin_token("admin")
    _set_admin_cookie(response, token)
    return _admin_user_to_auth_response(_admin_tokens[token], token)


@admin_router.get("/me", response_model=AdminUser)
def admin_me(request: Request) -> AdminUser:
    admin_user = get_optional_admin_portal_user(request)
    if not admin_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin access required")
    return admin_user


@admin_router.post("/logout")
def admin_logout(request: Request, response: Response) -> dict[str, bool]:
    authorization = request.headers.get("authorization", "")
    token = None
    if authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get(settings.admin_portal_cookie_name)
    _clear_admin_token(token)
    _clear_admin_cookie(response)
    return {"ok": True}


@router.get("/oauth/google", response_model=OAuthUrlResponse)
def google_oauth_url(mode: str = "login") -> OAuthUrlResponse:
    if not settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google OAuth is not configured")
    return OAuthUrlResponse(url=_google_authorize_url(_normalize_mode(mode)))


async def _exchange_google_code(code: str) -> dict[str, Any]:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google OAuth is not configured")

    data = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": _google_callback_url(),
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_response = await client.post("https://oauth2.googleapis.com/token", data=data)
        token_response.raise_for_status()
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Missing Google access token")

        userinfo_response = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_response.raise_for_status()
        profile = userinfo_response.json()

    return {
        "email": str(profile.get("email") or "").strip(),
        "full_name": str(profile.get("name") or profile.get("email") or "").strip(),
        "provider_id": str(profile.get("sub") or "").strip(),
    }


@router.get("/oauth/google/callback")
async def google_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> RedirectResponse:
    if error:
        return RedirectResponse(_frontend_redirect("/auth.html", error=error), status_code=302)

    if not code:
        return RedirectResponse(_frontend_redirect("/auth.html", error="missing_google_code"), status_code=302)

    try:
        profile = await _exchange_google_code(code)
        if not profile["email"]:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Google profile is missing email")
        user = _get_user_by_email(profile["email"])
        if user is None:
            if database_available():
                row = AuthRepository(get_pool()).upsert_oauth_user(
                    email=profile["email"],
                    full_name=profile["full_name"] or profile["email"],
                    provider="google",
                    provider_id=profile["provider_id"],
                    role="admin" if _is_admin_email(profile["email"]) else "user",
                )
                user = _row_to_user(row)
            else:
                user = UserRecord(
                    id=None,
                    email=_normalize_email(profile["email"]),
                    full_name=profile["full_name"] or profile["email"],
                    password_salt=secrets.token_hex(16),
                    password_hash=secrets.token_hex(32),
                    is_admin=_is_admin_email(profile["email"]),
                    provider="google",
                    provider_id=profile["provider_id"] or None,
                    created_at=_now(),
                )
                _users[user.email] = user
        else:
            user.full_name = profile["full_name"] or user.full_name
            user.provider = user.provider or "google"
            user.provider_id = user.provider_id or profile["provider_id"] or None
            user.is_admin = user.is_admin or _is_admin_email(user.email)

        token = _issue_token(user.email)
        response = RedirectResponse(_frontend_redirect("/", mode=_normalize_mode(state)), status_code=302)
        _set_auth_cookie(response, token)
        return response
    except HTTPException:
        raise
    except Exception:
        return RedirectResponse(_frontend_redirect("/auth.html", error="google_oauth_failed"), status_code=302)
