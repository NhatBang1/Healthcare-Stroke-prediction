from pathlib import Path


MIGRATION_PATH = Path(__file__).resolve().parents[1] / "migrations" / "001_strokeguard_core.sql"


def migration_sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def test_core_migration_contains_durable_chat_contract():
    sql = migration_sql().lower()

    assert "create schema if not exists strokeguard" in sql
    assert "create extension if not exists pgcrypto" in sql
    assert "create extension if not exists citext" in sql
    assert "create extension if not exists vector" in sql
    assert "create table if not exists strokeguard.users" in sql
    assert "create table if not exists strokeguard.auth_sessions" in sql
    assert "create table if not exists strokeguard.chat_sessions" in sql
    assert "create table if not exists strokeguard.chat_messages" in sql
    assert "assistant_card jsonb" in sql
    assert "create table if not exists strokeguard.patient_profiles" in sql
    assert "create table if not exists strokeguard.stroke_predictions" in sql
    assert "create table if not exists strokeguard.tool_audit_logs" in sql
    assert "create table if not exists strokeguard.memory_summaries" in sql
    assert "create table if not exists strokeguard.documents" in sql
    assert "create table if not exists strokeguard.ingestion_jobs" in sql
    assert "create table if not exists strokeguard.document_chunks" in sql
    assert "embedding vector(1024)" in sql


def test_core_migration_is_non_destructive_and_versioned():
    sql = migration_sql().lower()

    assert "pg_advisory_xact_lock" in sql
    assert "strokeguard.schema_migrations" in sql
    assert "'001'" in sql
    assert "drop schema" not in sql
    assert "drop table" not in sql
    assert "password '" not in sql


def test_core_migration_defines_ownership_and_history_indexes():
    sql = migration_sql().lower()

    assert "chat_sessions_owner_required" in sql
    assert "idx_chat_sessions_user_updated" in sql
    assert "idx_chat_sessions_anonymous_updated" in sql
    assert "idx_chat_messages_session_created" in sql
    assert "idx_stroke_predictions_session_created" in sql
    assert "idx_document_chunks_embedding_hnsw" in sql


def test_migration_registry_discovers_core_migration():
    from app.persistence.migrations import migration_files

    files = migration_files()

    assert [path.name for path in files] == [
        "001_strokeguard_core.sql",
        "002_patient_pregnancy_context.sql",
    ]


def test_database_is_disabled_without_connection_url():
    from app.persistence.database import database_available

    assert database_available() is False
