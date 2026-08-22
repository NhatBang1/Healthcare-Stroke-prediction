from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class RedisConfig:
    scheme: str
    host: str
    port: int
    db: int
    username: str | None
    password: str | None


def parse_redis_url(redis_url: str) -> RedisConfig:
    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", "rediss"}:
        raise ValueError("Unsupported Redis URL scheme")

    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    db = int(parsed.path.lstrip("/") or "0")
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    return RedisConfig(
        scheme=parsed.scheme,
        host=host,
        port=port,
        db=db,
        username=username,
        password=password,
    )


class RedisConnection:
    def __init__(self, redis_url: str) -> None:
        self._config = parse_redis_url(redis_url)

    def execute(self, *parts: object) -> object:
        payload = self._encode(*parts)
        context = ssl.create_default_context() if self._config.scheme == "rediss" else None
        with socket.create_connection((self._config.host, self._config.port), timeout=1.5) as raw_sock:
            sock = context.wrap_socket(raw_sock, server_hostname=self._config.host) if context else raw_sock
            if self._config.password or self._config.username:
                auth_parts = ["AUTH"]
                if self._config.username:
                    auth_parts.append(self._config.username)
                if self._config.password:
                    auth_parts.append(self._config.password)
                sock.sendall(self._encode(*auth_parts))
                self._read_reply(sock)
            if self._config.db:
                sock.sendall(self._encode("SELECT", str(self._config.db)))
                self._read_reply(sock)
            sock.sendall(payload)
            return self._read_reply(sock)

    @staticmethod
    def _encode(*parts: object) -> bytes:
        chunks = [f"*{len(parts)}\r\n".encode("ascii")]
        for part in parts:
            data = str(part).encode("utf-8")
            chunks.append(f"${len(data)}\r\n".encode("ascii"))
            chunks.append(data)
            chunks.append(b"\r\n")
        return b"".join(chunks)

    @classmethod
    def _read_reply(cls, sock) -> object:
        line = cls._readline(sock)
        if not line:
            raise RuntimeError("Empty Redis reply")
        prefix = line[:1]
        body = line[1:]
        if prefix == b"+":
            return body.decode("utf-8")
        if prefix == b":":
            return int(body)
        if prefix == b"$":
            length = int(body)
            if length == -1:
                return None
            data = cls._read_exact(sock, length)
            cls._read_exact(sock, 2)
            return data.decode("utf-8")
        if prefix == b"*":
            count = int(body)
            if count == -1:
                return None
            return [cls._read_reply(sock) for _ in range(count)]
        if prefix == b"-":
            raise RuntimeError(body.decode("utf-8"))
        raise RuntimeError("Unsupported Redis reply")

    @staticmethod
    def _readline(sock) -> bytes:
        chunks = bytearray()
        while True:
            char = sock.recv(1)
            if not char:
                break
            chunks.extend(char)
            if len(chunks) >= 2 and chunks[-2:] == b"\r\n":
                break
        return bytes(chunks[:-2])

    @staticmethod
    def _read_exact(sock, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError("Unexpected end of Redis stream")
            data.extend(chunk)
        return bytes(data)
