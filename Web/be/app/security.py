from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Deque

from fastapi import Request

from app.core.config import settings
from app.redis_support import RedisConnection, parse_redis_url


@dataclass(frozen=True)
class RateLimitRule:
    max_requests: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, rule: RateLimitRule) -> tuple[bool, int, int]:
        now = time()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - rule.window_seconds:
                hits.popleft()

            if len(hits) >= rule.max_requests:
                retry_after = max(1, int(rule.window_seconds - (now - hits[0])))
                return False, 0, retry_after

            hits.append(now)
            remaining = max(0, rule.max_requests - len(hits))
            return True, remaining, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class RedisRateLimiter:
    def __init__(self, redis_url: str, fallback: InMemoryRateLimiter | None = None) -> None:
        self._config = parse_redis_url(redis_url)
        self._redis = RedisConnection(redis_url)
        self._fallback = fallback or InMemoryRateLimiter()

    def allow(self, key: str, rule: RateLimitRule) -> tuple[bool, int, int]:
        bucket = int(time() // rule.window_seconds)
        bucket_key = f"rate:{key}:{bucket}"
        try:
            count = self._incr(bucket_key)
            if count == 1:
                self._expire(bucket_key, rule.window_seconds + 1)
            if count > rule.max_requests:
                return False, 0, max(1, rule.window_seconds - int(time() % rule.window_seconds))
            return True, max(0, rule.max_requests - count), 0
        except Exception:
            return self._fallback.allow(key, rule)

    def reset(self) -> None:
        cursor = "0"
        while True:
            reply = self._redis.execute("SCAN", cursor, "MATCH", "rate:*", "COUNT", "200")
            if not isinstance(reply, list) or len(reply) != 2:
                return
            cursor = str(reply[0])
            keys = reply[1]
            if isinstance(keys, list) and keys:
                self._redis.execute("DEL", *[str(key) for key in keys])
            if cursor == "0":
                return

    def _incr(self, key: str) -> int:
        reply = self._redis.execute("INCR", key)
        if not isinstance(reply, int):
            raise RuntimeError("Unexpected Redis INCR reply")
        return reply

    def _expire(self, key: str, ttl_seconds: int) -> None:
        self._redis.execute("EXPIRE", key, str(ttl_seconds))


def build_rate_limiter(redis_url: str | None) -> InMemoryRateLimiter | RedisRateLimiter:
    if redis_url:
        try:
            return RedisRateLimiter(redis_url, fallback=InMemoryRateLimiter())
        except Exception:
            return InMemoryRateLimiter()
    return InMemoryRateLimiter()


rate_limiter = build_rate_limiter(settings.redis_url)


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded_for:
        return forwarded_for
    host = request.client.host if request.client else ""
    return host or "unknown"
