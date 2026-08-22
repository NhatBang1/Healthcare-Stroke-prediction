from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from time import time
from uuid import uuid4

from app.core.config import settings
from app.redis_support import RedisConnection


class MemorySessionStore:
    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._session_order: dict[str, float] = {}
        self._session_titles: dict[str, str] = {}

    def new_session_id(self) -> str:
        return str(uuid4())

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        assistant_card: dict[str, object] | None = None,
        reply_context: dict[str, object] | None = None,
        media: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        timestamp = self._now_label()
        user_payload: dict[str, object] = {
            "id": str(uuid4()),
            "role": "user",
            "content": user_message,
            "time": timestamp,
        }
        if reply_context is not None:
            user_payload["reply_context"] = reply_context
        self._messages[session_id].append(user_payload)
        assistant_payload: dict[str, object] = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": assistant_message,
            "time": self._now_label(),
        }
        if assistant_card is not None:
            assistant_payload["assistant_card"] = assistant_card
        if media:
            assistant_payload["media"] = media
        self._messages[session_id].append(assistant_payload)
        self._session_order[session_id] = time()
        self._session_titles.setdefault(session_id, self._session_title(self._messages[session_id]))
        return self.messages(session_id)

    def messages(self, session_id: str) -> list[dict[str, object]]:
        return list(self._messages.get(session_id, []))

    def sessions(self) -> list[dict[str, object]]:
        ordered_sessions = sorted(
            self._messages.items(),
            key=lambda item: self._session_order.get(item[0], 0.0),
            reverse=True,
        )
        return [
            {
                "session_id": session_id,
                "message_count": len(messages),
                "title": self._session_titles.get(session_id) or self._session_title(messages),
            }
            for session_id, messages in ordered_sessions
        ]

    def reset(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._session_order.pop(session_id, None)
        self._session_titles.pop(session_id, None)

    def rename(self, session_id: str, title: str) -> None:
        self._session_titles[session_id] = title

    @staticmethod
    def _now_label() -> str:
        return datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M")

    @staticmethod
    def _session_title(messages: list[dict[str, object]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = " ".join(str(message.get("content", "")).split()).strip()
            if content:
                return content[:60] + ("..." if len(content) > 60 else "")
        return "Cuộc hội thoại"


class RedisSessionStore:
    def __init__(self, redis_url: str, ttl_seconds: int | None = None) -> None:
        self._redis = RedisConnection(redis_url)
        self._ttl_seconds = ttl_seconds or settings.chat_memory_ttl_seconds
        self._session_index_key = "strokeguard:chat:sessions"
        self._session_title_key = "strokeguard:chat:session:{session_id}:title"

    def new_session_id(self) -> str:
        return str(uuid4())

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        assistant_card: dict[str, object] | None = None,
        reply_context: dict[str, object] | None = None,
        media: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        messages_key = self._messages_key(session_id)
        timestamp = time()
        payloads = [
            json.dumps(
                {
                    "id": str(uuid4()),
                    "role": "user",
                    "content": user_message,
                    "time": self._now_label(),
                    **({"reply_context": reply_context} if reply_context is not None else {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                {
                    "id": str(uuid4()),
                    "role": "assistant",
                    "content": assistant_message,
                    "time": self._now_label(),
                    **({"assistant_card": assistant_card} if assistant_card is not None else {}),
                    **({"media": media} if media else {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
        self._redis.execute("RPUSH", messages_key, *payloads)
        self._redis.execute("EXPIRE", messages_key, str(self._ttl_seconds))
        self._redis.execute("ZADD", self._session_index_key, str(timestamp), session_id)
        self._redis.execute("EXPIRE", self._session_index_key, str(self._ttl_seconds))
        if not self._title(session_id):
            self._redis.execute("SET", self._title_key(session_id), self._session_title(self.messages(session_id)))
            self._redis.execute("EXPIRE", self._title_key(session_id), str(self._ttl_seconds))
        return self.messages(session_id)

    def messages(self, session_id: str) -> list[dict[str, object]]:
        raw_messages = self._redis.execute("LRANGE", self._messages_key(session_id), "0", "-1")
        if not raw_messages:
            return []
        messages: list[dict[str, object]] = []
        for index, item in enumerate(raw_messages):
            try:
                message = json.loads(str(item))
            except Exception:
                continue
            if isinstance(message, dict) and "role" in message and "content" in message:
                content = str(message["content"])
                fallback_id = sha256(
                    f"{session_id}:{index}:{message['role']}:{content}".encode("utf-8")
                ).hexdigest()[:32]
                normalized: dict[str, object] = {
                    "id": str(message.get("id") or f"legacy-{fallback_id}"),
                    "role": str(message["role"]),
                    "content": content,
                }
                if "time" in message:
                    normalized["time"] = str(message["time"])
                if isinstance(message.get("assistant_card"), dict):
                    normalized["assistant_card"] = message["assistant_card"]
                if isinstance(message.get("reply_context"), dict):
                    normalized["reply_context"] = message["reply_context"]
                if isinstance(message.get("media"), list):
                    normalized["media"] = message["media"]
                messages.append(normalized)
        return messages

    def sessions(self) -> list[dict[str, object]]:
        raw_sessions = self._redis.execute("ZREVRANGE", self._session_index_key, "0", "-1", "WITHSCORES")
        if not raw_sessions:
            return []

        sessions: list[dict[str, object]] = []
        for index in range(0, len(raw_sessions), 2):
            session_id = str(raw_sessions[index])
            messages = self.messages(session_id)
            message_count = len(messages)
            if message_count == 0:
                self._redis.execute("ZREM", self._session_index_key, session_id)
                self._redis.execute("DEL", self._title_key(session_id))
                continue
            sessions.append(
                {
                    "session_id": session_id,
                    "message_count": message_count,
                    "title": self._title(session_id) or self._session_title(messages),
                }
            )
        return sessions

    def reset(self, session_id: str) -> None:
        self._redis.execute("DEL", self._messages_key(session_id))
        self._redis.execute("ZREM", self._session_index_key, session_id)
        self._redis.execute("DEL", self._title_key(session_id))

    def rename(self, session_id: str, title: str) -> None:
        self._redis.execute("SET", self._title_key(session_id), title)
        self._redis.execute("EXPIRE", self._title_key(session_id), str(self._ttl_seconds))

    @staticmethod
    def _now_label() -> str:
        return datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M")

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"strokeguard:chat:session:{session_id}:messages"

    def _title_key(self, session_id: str) -> str:
        return self._session_title_key.format(session_id=session_id)

    def _title(self, session_id: str) -> str | None:
        title = self._redis.execute("GET", self._title_key(session_id))
        return str(title) if title else None

    @staticmethod
    def _session_title(messages: list[dict[str, object]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = " ".join(str(message.get("content", "")).split()).strip()
            if content:
                return content[:60] + ("..." if len(content) > 60 else "")
        return "Cuộc hội thoại"


def build_session_store(redis_url: str | None) -> MemorySessionStore | RedisSessionStore:
    if redis_url:
        try:
            return RedisSessionStore(redis_url)
        except Exception:
            return MemorySessionStore()
    return MemorySessionStore()


session_store = build_session_store(settings.redis_url)
