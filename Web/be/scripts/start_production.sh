#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="/home/nhatbang/Samsung_NIC/Healthcare-Stroke-prediction/Web/be"
PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"
UVICORN_BIN="${BACKEND_DIR}/.venv/bin/uvicorn"

if [[ -z "${DATABASE_URL:-}" ]]; then
  DB_PASSWORD="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' postgres \
    | sed -n 's/^POSTGRES_PASSWORD=//p' \
    | head -n 1)"
  if [[ -z "${DB_PASSWORD}" ]]; then
    echo "PostgreSQL credential is unavailable" >&2
    exit 1
  fi
  export DATABASE_URL="postgresql://cloud:${DB_PASSWORD}@127.0.0.1:5432/stroke_ai"
  unset DB_PASSWORD
fi

cd "${BACKEND_DIR}"
PYTHONPATH=. "${PYTHON_BIN}" -c 'from app.persistence.migrations import run_migrations; run_migrations()'
exec "${UVICORN_BIN}" app.main:app --host 127.0.0.1 --port 9001
