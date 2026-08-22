from __future__ import annotations

from pathlib import Path

import psycopg

from app.core.config import WEB_ROOT, settings


MIGRATIONS_DIR = WEB_ROOT / "be" / "migrations"


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def run_migrations(database_url: str | None = None) -> list[str]:
    connection_url = database_url or settings.database_url
    if not connection_url:
        raise RuntimeError("DATABASE_URL is not configured")

    applied: list[str] = []
    with psycopg.connect(connection_url) as connection:
        for path in migration_files():
            connection.execute(path.read_text(encoding="utf-8"))
            applied.append(path.stem.split("_", 1)[0])
    return applied
