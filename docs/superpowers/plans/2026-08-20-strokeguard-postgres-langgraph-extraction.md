# StrokeGuard PostgreSQL and LangGraph Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PostgreSQL `stroke_ai` the durable source of truth for StrokeGuard chat and prediction cards, and replace regex screening extraction with a DeepSeek function-calling tool orchestrated by LangGraph.

**Architecture:** Keep the existing FastAPI/static frontend boundary, add a focused PostgreSQL persistence package, and compile an explicit LangGraph `StateGraph` whose extraction node forces the DeepSeek `extract_screening_profile` tool. Application validators merge profile facts across turns and route to a missing-field response or the existing prediction service; Redis remains an optional cache/rate limiter.

**Tech Stack:** Python 3.11, FastAPI, Pydantic 2, psycopg 3 pool, PostgreSQL 15/pgvector, LangGraph, httpx, DeepSeek Chat Completions function calling, pytest, Vite.

---

### Task 1: Isolate tests and add runtime dependencies

**Files:**
- Modify: `Web/be/tests/conftest.py`
- Modify: `Web/be/requirements.txt`
- Modify: `Web/be/app/core/config.py`
- Create: `Web/be/.env.example`
- Test: `Web/be/tests/test_test_environment.py`

- [ ] **Step 1: Write a failing test that proves test settings cannot use production persistence**

```python
def test_test_environment_disables_external_persistence():
    from app.core.config import settings

    assert settings.app_env == "test"
    assert settings.database_url is None
    assert settings.redis_url is None
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_test_environment.py -q`

Expected: FAIL because `app_env` and `database_url` are not defined and `.env` currently supplies the production Redis URL.

- [ ] **Step 3: Set safe environment values before importing application modules**

At the top of `tests/conftest.py`, set `APP_ENV=test`, `DATABASE_URL=` and `REDIS_URL=` before importing `app.security`. Add nullable `database_url`, `app_env`, graph checkpoint, DeepSeek strict-mode, and anonymous-owner cookie settings to `Settings`.

- [ ] **Step 4: Add pinned dependencies and a secret-free environment template**

Add compatible pinned versions of `psycopg[binary,pool]`, `langgraph`, and `langgraph-checkpoint-postgres`. Document variable names only in `.env.example`; do not copy credentials.

- [ ] **Step 5: Run the focused test and existing contract baseline**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_test_environment.py tests/test_api_contract.py tests/test_auth_progress_contract.py -q
```

Expected: all tests pass without connecting to production Redis/PostgreSQL.

- [ ] **Step 6: Commit**

```bash
git add Web/be/tests/conftest.py Web/be/tests/test_test_environment.py Web/be/requirements.txt Web/be/app/core/config.py Web/be/.env.example
git commit -m "test: isolate StrokeGuard persistence settings"
```

### Task 2: Create the PostgreSQL migration and migration runner

**Files:**
- Create: `Web/be/migrations/001_strokeguard_core.sql`
- Create: `Web/be/app/persistence/__init__.py`
- Create: `Web/be/app/persistence/database.py`
- Create: `Web/be/app/persistence/migrations.py`
- Test: `Web/be/tests/test_database_migration.py`

- [ ] **Step 1: Write failing migration-contract tests**

The tests read the SQL file and assert the presence of the approved schema, extensions, tables, ownership constraints, card JSONB, profile checks, prediction audit, RAG tables, and indexes. They also assert there are no `DROP SCHEMA`, `DROP TABLE`, or plaintext password statements.

```python
def test_core_migration_contains_durable_chat_contract():
    sql = migration_sql().lower()
    assert "create schema if not exists strokeguard" in sql
    assert "assistant_card jsonb" in sql
    assert "create table if not exists strokeguard.patient_profiles" in sql
    assert "create table if not exists strokeguard.stroke_predictions" in sql
```

- [ ] **Step 2: Run the migration test and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_database_migration.py -q`

Expected: FAIL because the migration does not exist.

- [ ] **Step 3: Implement the idempotent migration**

Create the schema and tables from the approved design using `CREATE ... IF NOT EXISTS`, named constraints, and indexes. Use `pg_advisory_xact_lock` and record version `001` in `strokeguard.schema_migrations`.

- [ ] **Step 4: Implement the connection pool and runner**

Expose:

```python
def database_available() -> bool: ...
def get_pool() -> ConnectionPool: ...
def run_migrations() -> list[str]: ...
```

The pool is lazy, uses `dict_row`, and raises a clear error when `DATABASE_URL` is absent outside tests. The runner applies SQL transactionally and returns applied versions.

- [ ] **Step 5: Run tests and inspect the SQL**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_database_migration.py -q
git diff --check -- Web/be/migrations Web/be/app/persistence
```

Expected: PASS and no whitespace errors.

- [ ] **Step 6: Apply the authorized migration to `stroke_ai` and verify objects**

Use the configured PostgreSQL container connection without printing credentials. Run the migration against `stroke_ai`, then query `pg_tables`, `pg_constraint`, `pg_indexes`, and `pg_extension` read-only to verify the result.

- [ ] **Step 7: Commit**

```bash
git add Web/be/migrations/001_strokeguard_core.sql Web/be/app/persistence Web/be/tests/test_database_migration.py
git commit -m "feat: add StrokeGuard PostgreSQL schema"
```

### Task 3: Add durable session, message, card, and profile repositories

**Files:**
- Create: `Web/be/app/persistence/repositories.py`
- Modify: `Web/be/app/memory.py`
- Modify: `Web/be/app/schemas.py`
- Test: `Web/be/tests/test_postgres_repositories.py`
- Test: `Web/be/tests/test_api_contract.py`

- [ ] **Step 1: Write failing repository behavior tests**

Use a small fake pool/connection boundary for unit tests and verify:

- owner filters are passed to list/load/rename/delete operations;
- assistant card JSON round-trips with the assistant message;
- the latest explicit profile update merges without erasing prior fields;
- soft-delete only targets the selected owned session.

```python
def test_message_round_trip_preserves_assistant_card(repository):
    saved = repository.add_message(
        session_id=SESSION_ID,
        owner=OWNER,
        role="assistant",
        content="Kết quả",
        assistant_card={"risk_label": "Thấp"},
    )
    assert saved["assistant_card"]["risk_label"] == "Thấp"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_postgres_repositories.py -q`

Expected: FAIL because the repositories do not exist.

- [ ] **Step 3: Implement focused repositories**

Implement `OwnerIdentity`, `ChatRepository`, `ProfileRepository`, `PredictionRepository`, and `ToolAuditRepository`. All SQL is parameterized. Repository methods require an owner identity for user-facing session operations.

- [ ] **Step 4: Extend the message schema**

Add `id` and `assistant_card` to persisted `ChatMessage`. Keep response compatibility by making both optional/defaulted for legacy in-memory tests.

- [ ] **Step 5: Add a PostgreSQL-backed session-store adapter**

Preserve the existing memory-store interface where practical, but add owner-aware methods used by new API routes. Redis may cache recent messages but cannot be the durable write target when the database is configured.

- [ ] **Step 6: Run focused and existing tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_postgres_repositories.py tests/test_api_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Web/be/app/persistence/repositories.py Web/be/app/memory.py Web/be/app/schemas.py Web/be/tests/test_postgres_repositories.py Web/be/tests/test_api_contract.py
git commit -m "feat: persist StrokeGuard conversations"
```

### Task 4: Implement the DeepSeek extraction tool

**Files:**
- Create: `Web/be/app/extraction/__init__.py`
- Create: `Web/be/app/extraction/schemas.py`
- Create: `Web/be/app/extraction/deepseek_tool.py`
- Modify: `Web/be/app/llm.py`
- Test: `Web/be/tests/test_deepseek_extraction.py`

- [ ] **Step 1: Write failing extraction tests**

Cover tool-call parsing for:

- `Nam, huyết áp 100/70, glucose tầm 70, 20 tuổi, BMI là 25`;
- explicit `không có bệnh tim`;
- `không biết bệnh tim` remaining unknown;
- evidence absent from the newest message being rejected;
- malformed tool arguments and provider errors;
- no regex fallback.

```python
def test_extractor_accepts_natural_vietnamese_tool_arguments(fake_transport):
    result = extractor.extract(new_message=SAMPLE, current_profile={})
    assert result.avg_glucose_level == 70
    assert result.bmi == 25
    assert result.systolic_bp == 100
    assert result.diastolic_bp == 70
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_deepseek_extraction.py -q`

Expected: FAIL because the extractor does not exist.

- [ ] **Step 3: Define Pydantic extraction contracts**

Define `EvidenceItem` and `ScreeningProfileUpdate`. Optional values distinguish absent facts from explicit false values. `explicit_unknown_fields` and `corrections` are enums limited to known screening fields.

- [ ] **Step 4: Implement forced DeepSeek tool calling**

Send the newest message, recent context, and current profile to `/chat/completions` with:

```python
"tools": [{"type": "function", "function": EXTRACT_SCREENING_PROFILE_TOOL}],
"tool_choice": {"type": "function", "function": {"name": "extract_screening_profile"}},
```

Parse only the matching tool call. Validate arguments with Pydantic and verify every non-empty evidence quote occurs in the normalized newest message.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_deepseek_extraction.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Web/be/app/extraction Web/be/app/llm.py Web/be/tests/test_deepseek_extraction.py
git commit -m "feat: extract screening fields with DeepSeek tools"
```

### Task 5: Build the explicit LangGraph screening workflow

**Files:**
- Create: `Web/be/app/chat/__init__.py`
- Create: `Web/be/app/chat/blood_pressure.py`
- Create: `Web/be/app/chat/graph.py`
- Create: `Web/be/app/chat/responses.py`
- Test: `Web/be/tests/test_chat_graph.py`

- [ ] **Step 1: Write failing graph tests**

Use injected extractor, prediction service, and repositories. Verify:

- prior-turn fields merge with the newest tool output;
- prediction is not called with one to five required fields;
- prediction is called exactly once when the sixth field is accepted;
- under-18 or incomplete blood pressure does not infer hypertension;
- adult `130/80` maps to stage 1 and `hypertension=true`;
- `không yếu tay` alone does not route to emergency;
- provider failure returns a controlled retry response.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_chat_graph.py -q`

Expected: FAIL because graph modules do not exist.

- [ ] **Step 3: Implement deterministic blood-pressure classification**

Expose:

```python
def classify_adult_blood_pressure(age, systolic, diastolic, *, pregnant=False) -> BloodPressureResult: ...
```

Return `unknown` for age below 18, pregnancy, or incomplete readings. Use the approved AHA categories for non-pregnant adults.

- [ ] **Step 4: Implement graph state and nodes**

Define the state from the spec and nodes for persistence, triage, extraction, validation/merge, missing-field response, prediction, card creation, and assistant persistence. Conditional edges are explicit; the extraction node has no direct edge to prediction.

- [ ] **Step 5: Compile with injectable checkpointer**

Tests use `InMemorySaver`. Production uses `PostgresSaver` when configured. Use `session_id` as `thread_id` and call `setup()` only through the persistence bootstrap.

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_chat_graph.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Web/be/app/chat Web/be/tests/test_chat_graph.py
git commit -m "feat: orchestrate screening with LangGraph"
```

### Task 6: Wire ownership, graph, and durable history into FastAPI

**Files:**
- Modify: `Web/be/app/main.py`
- Modify: `Web/be/app/auth.py`
- Modify: `Web/be/app/history.py`
- Modify: `Web/be/app/progress.py`
- Modify: `Web/be/app/schemas.py`
- Test: `Web/be/tests/test_api_contract.py`
- Test: `Web/be/tests/test_auth_progress_contract.py`

- [ ] **Step 1: Write failing API regression tests**

Add contract tests for:

- strict missing-field gate;
- multi-turn completion and one prediction;
- history messages returning `assistant_card`;
- cross-owner list/load/rename/delete denial;
- progress derived from persisted predictions;
- new-chat semantics remaining non-destructive.

- [ ] **Step 2: Run focused API tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_api_contract.py tests/test_auth_progress_contract.py -q
```

Expected: new assertions fail under the legacy global Redis API.

- [ ] **Step 3: Add owner resolution**

Resolve authenticated users from the existing cookie/token path. For guests, issue an HttpOnly anonymous-owner cookie and persist only its hash. Every session endpoint passes `OwnerIdentity` to the repository.

- [ ] **Step 4: Replace `_build_chat_reply` screening routing with graph invocation**

Keep direct emergency/general chat behavior behind graph nodes. Remove `_extract_screening_profile` and its regex routing from production. Invoke the graph with `configurable.thread_id=session_id`.

- [ ] **Step 5: Persist auth records and tokens**

When PostgreSQL is configured, register/login/OAuth/token lookup use durable user and auth-session repositories. Tests retain isolated in-memory adapters.

- [ ] **Step 6: Return durable messages/cards and prediction progress**

History endpoints serialize `assistant_card`; progress reads ordered predictions. Preserve existing route shapes where compatibility permits.

- [ ] **Step 7: Run API tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_api_contract.py tests/test_auth_progress_contract.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Web/be/app/main.py Web/be/app/auth.py Web/be/app/history.py Web/be/app/progress.py Web/be/app/schemas.py Web/be/tests/test_api_contract.py Web/be/tests/test_auth_progress_contract.py
git commit -m "feat: serve owner-scoped durable StrokeGuard chat"
```

### Task 7: Restore prediction cards from history in the frontend

**Files:**
- Modify: `Web/fe/index.html`
- Modify: `Web/fe/package.json`
- Create: `Web/fe/tests/history-card.test.mjs`

- [ ] **Step 1: Write a failing frontend regression test**

Extract the message-rendering data adapter into a testable function or load the page module in JSDOM. Assert that a history assistant message passes its persisted card to `appendMessage`.

```javascript
assert.deepEqual(messageRenderOptions({ role: "assistant", assistant_card: card }), {
  assistantCard: card
});
```

- [ ] **Step 2: Run the test and verify RED**

Run: `npm test -- history-card.test.mjs`

Expected: FAIL because history rendering currently calls `appendMessage(message)` without the card.

- [ ] **Step 3: Implement the minimal history-card adapter**

Change history rendering to pass `{ assistantCard: message.assistant_card || null }`. Keep live-send rendering unchanged.

- [ ] **Step 4: Run frontend test and build**

Run:

```bash
npm test -- history-card.test.mjs
npm run build
```

Expected: PASS and Vite exits 0.

- [ ] **Step 5: Commit**

```bash
git add Web/fe/index.html Web/fe/package.json Web/fe/tests/history-card.test.mjs
git commit -m "fix: restore prediction cards from chat history"
```

### Task 8: Full verification and database inspection

**Files:**
- Modify: `/home/nhatbang/EXE101/PRJ/Task.md` outside the repository
- Verify: `Web/be`, `Web/fe`, PostgreSQL `stroke_ai`

- [ ] **Step 1: Run the full backend suite**

Run: `PYTHONPATH=. .venv/bin/pytest -q`

Expected: zero failures and no production Redis/PostgreSQL cleanup.

- [ ] **Step 2: Run frontend tests and production build**

Run:

```bash
npm test
npm run build
```

Expected: zero test failures and build exit 0.

- [ ] **Step 3: Inspect database objects and constraints**

Query `pg_tables`, `pg_indexes`, `pg_constraint`, and `pg_extension` for `stroke_ai`. Confirm the application tables exist and no rows were inserted by isolated automated tests.

- [ ] **Step 4: Run a local API smoke test without reloading production**

Start a separate local process on a non-production port with the configured database and a disposable anonymous owner. Exercise missing-field collection, completed prediction, reload/history card, and delete only the explicitly validated disposable session.

- [ ] **Step 5: Review the diff and check for secrets**

Run:

```bash
git diff --check
git status --short
rg -n "DEEPSEEK_API_KEY=.+|DATABASE_URL=postgres.+@" Web/be --glob '!*.example' --glob '!.env'
```

Expected: no whitespace errors, no staged unrelated files, and no credentials in tracked files.

- [ ] **Step 6: Update `Task.md` with exact verification evidence**

Mark the StrokeGuard database task completed only when all acceptance criteria pass. Otherwise record the precise unfinished step and failure.

- [ ] **Step 7: Stop before production reload**

Report local results and request explicit authorization before changing `.env`, restarting the port-9001 backend, or verifying the public tunnel.
