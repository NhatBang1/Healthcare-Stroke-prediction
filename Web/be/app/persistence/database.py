from __future__ import annotations

from threading import Lock

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import settings


_pool: ConnectionPool | None = None
_checkpoint_pool: ConnectionPool | None = None
_pool_lock = Lock()


def database_available() -> bool:
    return bool(settings.database_url)


def get_pool() -> ConnectionPool:
    global _pool

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=1,
                max_size=8,
                kwargs={"row_factory": dict_row},
                open=False,
            )
            _pool.open(wait=True, timeout=10)
    return _pool


def get_checkpoint_pool() -> ConnectionPool:
    global _checkpoint_pool

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    if _checkpoint_pool is not None:
        return _checkpoint_pool

    with _pool_lock:
        if _checkpoint_pool is None:
            _checkpoint_pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=1,
                max_size=4,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
                open=False,
            )
            _checkpoint_pool.open(wait=True, timeout=10)
    return _checkpoint_pool


def close_pool() -> None:
    global _pool, _checkpoint_pool

    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        if _checkpoint_pool is not None:
            _checkpoint_pool.close()
            _checkpoint_pool = None
