from __future__ import annotations

import secrets

from fastapi import Response
from starlette.requests import Request

from app.core.config import settings
from app.memory import session_store
from app.persistence.conversation_store import PostgresConversationStore
from app.persistence.database import database_available, get_pool
from app.persistence.repositories import AuthRepository, OwnerIdentity


def conversation_store(request: Request, response: Response):
    if not database_available():
        return session_store
    pool = get_pool()
    authorization = request.headers.get("authorization", "")
    auth_token = authorization.split(" ", 1)[1].strip() if authorization.lower().startswith("bearer ") else None
    auth_token = auth_token or request.cookies.get(settings.auth_token_cookie_name)
    if auth_token:
        user = AuthRepository(pool).user_by_token(auth_token)
        if user:
            return PostgresConversationStore(pool, owner=OwnerIdentity(user_id=user["id"]))

    token = request.cookies.get(settings.anonymous_owner_cookie_name)
    if not token:
        token = secrets.token_urlsafe(32)
        response.set_cookie(
            key=settings.anonymous_owner_cookie_name,
            value=token,
            httponly=True,
            samesite="lax",
            secure=settings.cookie_secure,
            max_age=365 * 24 * 60 * 60,
            path="/",
        )
    return PostgresConversationStore(pool, token)
