# StrokeGuard PostgreSQL and LangGraph Extraction Design

Date: 2026-08-20
Status: Approved direction; awaiting written-spec review before implementation

## 1. Objective

Replace volatile, globally shared chat storage and regex-based screening extraction with:

- durable, owner-scoped PostgreSQL storage in the existing `stroke_ai` database;
- a DeepSeek function-calling tool named `extract_screening_profile`;
- an explicit LangGraph workflow that validates and accumulates screening data across turns;
- a strict gate that runs the stroke model only after all six screening fields are present and valid;
- persisted assistant prediction cards that survive reload and history navigation.

The six required screening fields are `age`, `gender`, `hypertension`, `heart_disease`, `avg_glucose_level`, and `bmi`.

## 2. Confirmed Current-State Problems

1. Regex extraction only recognizes rigid forms such as `glucose 70` and `BMI 25`. Natural Vietnamese such as `glucose tầm 70` and `BMI là 25` is missed.
2. Prediction currently runs with missing required fields and silently fills them from preprocessing metadata.
3. Redis messages contain only `role`, `content`, and `time`; `assistant_card` is not persisted.
4. Redis is the current source of truth, uses DB 0 with a seven-day TTL, and does not scope sessions to an authenticated or anonymous owner.
5. Backend tests can load the production `.env` and clear the configured Redis session store.
6. The fallback symptom handler is template-based, does not use conversational context, and can treat negated phrases such as `không yếu tay` as positive emergency evidence.
7. The `stroke_ai` PostgreSQL database exists and is empty. PostgreSQL 15.15 is running, and `pgcrypto`, `citext`, and `vector` 0.8.1 are available.

## 3. Chosen Approach

Use an explicit `StateGraph`, not a free-running agent loop. DeepSeek is forced to call one extraction function for screening messages. Application code validates the tool arguments, merges accepted fields into the durable profile, computes missing fields, and selects the next graph node.

DeepSeek may interpret language, but it may not:

- invent a value;
- classify a single incomplete blood-pressure number as hypertension;
- decide whether the profile is complete;
- call the prediction model directly;
- bypass numeric and categorical validation;
- overwrite an existing value without evidence from the newest user turn.

The initial implementation uses standard DeepSeek function calling with a forced `tool_choice` and Pydantic validation. DeepSeek strict-mode beta is feature-flagged for later evaluation rather than required for the first production path.

## 4. PostgreSQL Ownership and Schema

Use the existing database `stroke_ai`, owned by PostgreSQL role `cloud`. Create a dedicated application schema named `strokeguard`; do not place application tables from this project into another database such as `verbaa`.

The migration is idempotent and runs in a transaction under a PostgreSQL advisory lock. It records success in `strokeguard.schema_migrations`. It never drops an existing table or schema.

### 4.1 Extensions

Enable:

- `pgcrypto` for UUID generation;
- `citext` for case-insensitive email uniqueness;
- `vector` for future document embeddings.

### 4.2 Application Tables

#### `strokeguard.schema_migrations`

- `version TEXT PRIMARY KEY`
- `description TEXT NOT NULL`
- `applied_at TIMESTAMPTZ NOT NULL DEFAULT now()`

#### `strokeguard.users`

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `email CITEXT UNIQUE NOT NULL`
- `full_name TEXT NOT NULL`
- `password_salt TEXT`
- `password_hash TEXT`
- `provider TEXT`
- `provider_id TEXT`
- `role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin'))`
- `created_at`, `updated_at TIMESTAMPTZ`

Passwords retain the existing PBKDF2 representation during migration. OAuth users may have null password fields.

#### `strokeguard.auth_sessions`

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE`
- `token_hash TEXT UNIQUE NOT NULL`
- `expires_at TIMESTAMPTZ NOT NULL`
- `created_at`, `last_seen_at TIMESTAMPTZ`

Only a cryptographic token hash is stored; raw bearer/cookie tokens are never persisted.

#### `strokeguard.chat_sessions`

- `id UUID PRIMARY KEY`
- `user_id UUID REFERENCES users(id) ON DELETE CASCADE`
- `anonymous_owner_hash TEXT`
- `title TEXT NOT NULL DEFAULT 'Cuộc hội thoại'`
- `status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted'))`
- `created_at`, `updated_at`, `deleted_at TIMESTAMPTZ`
- constraint requiring either `user_id` or `anonymous_owner_hash`

Authenticated queries filter by `user_id`. Guest queries filter by a hash derived from an HttpOnly anonymous-owner cookie. Logging in may atomically claim the current guest sessions.

#### `strokeguard.chat_messages`

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool'))`
- `content TEXT NOT NULL`
- `assistant_card JSONB`
- `metadata JSONB NOT NULL DEFAULT '{}'::jsonb`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`

`assistant_card` stores the complete response contract returned to the frontend. History APIs return it with the assistant message so prediction cards render identically after reload.

#### `strokeguard.patient_profiles`

One current merged profile per chat session:

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `session_id UUID UNIQUE NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `age DOUBLE PRECISION CHECK (age BETWEEN 0 AND 100)`
- `gender TEXT CHECK (gender IN ('male', 'female'))`
- `systolic_bp DOUBLE PRECISION CHECK (systolic_bp BETWEEN 50 AND 300)`
- `diastolic_bp DOUBLE PRECISION CHECK (diastolic_bp BETWEEN 30 AND 200)`
- `blood_pressure_category TEXT CHECK (blood_pressure_category IN ('normal', 'elevated', 'stage_1', 'stage_2', 'severe', 'unknown'))`
- `hypertension BOOLEAN`
- `heart_disease BOOLEAN`
- `avg_glucose_level DOUBLE PRECISION CHECK (avg_glucose_level BETWEEN 40 AND 400)`
- `bmi DOUBLE PRECISION CHECK (bmi BETWEEN 10 AND 80)`
- optional clinical-model columns: `avg_resting_bp`, `avg_max_hr`, `avg_oldpeak`, `heart_disease_rate`, `cardio_cholesterol`, `cardio_gluc`, `cardio_rate`, `smoke_flag`, `alco_flag`, `ever_married`, `work_type`, `residence_type`, `smoking_status`
- `extra_features JSONB NOT NULL DEFAULT '{}'::jsonb`
- `source_evidence JSONB NOT NULL DEFAULT '{}'::jsonb`
- `created_at`, `updated_at TIMESTAMPTZ`

`source_evidence` records the newest accepted quote and source message ID for each field.

#### `strokeguard.stroke_predictions`

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `profile_id UUID REFERENCES patient_profiles(id) ON DELETE SET NULL`
- `source_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL`
- `model_name`, `model_version TEXT NOT NULL`
- `risk_probability DOUBLE PRECISION CHECK (risk_probability BETWEEN 0 AND 1)`
- `risk_label TEXT CHECK (risk_label IN ('low_risk', 'medium_risk', 'high_risk'))`
- `threshold DOUBLE PRECISION NOT NULL`
- `input_features`, `filled_features`, `result_payload JSONB NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`

The progress API derives its timeline from ordered prediction rows instead of keyword-weighting raw chat text.

#### `strokeguard.tool_audit_logs`

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL`
- `tool_name TEXT NOT NULL`
- `status TEXT NOT NULL CHECK (status IN ('success', 'validation_error', 'provider_error'))`
- `input_metadata`, `output_metadata`, `error_detail JSONB NOT NULL DEFAULT '{}'::jsonb`
- `latency_ms INTEGER CHECK (latency_ms >= 0)`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`

Audit metadata omits API keys and avoids duplicating unrestricted medical free text.

#### Durable memory and RAG tables

- `strokeguard.memory_summaries`
- `strokeguard.documents`
- `strokeguard.ingestion_jobs`
- `strokeguard.document_chunks`, with `embedding vector(1024)` and full-text `TSVECTOR`

These tables establish the approved long-memory/document contract. Document ingestion and RAG behavior remain outside this implementation slice unless required for database repository tests.

### 4.3 Indexes

Create indexes for:

- chat sessions by owner and `updated_at DESC`;
- messages by `(session_id, created_at, id)`;
- predictions by `(session_id, created_at DESC)`;
- auth-session expiry;
- tool audit by `(session_id, created_at DESC)`;
- document status and active chunks;
- GIN full-text document search;
- HNSW vector cosine search for active chunks.

## 5. DeepSeek Extraction Tool

Tool name: `extract_screening_profile`

The API call includes recent conversation context, the current durable profile, and the newest user message. `tool_choice` forces DeepSeek to call this exact function for a screening turn.

Tool output fields:

- `age`
- `gender`
- `systolic_bp`
- `diastolic_bp`
- `hypertension_history`
- `heart_disease`
- `avg_glucose_level`
- `bmi`
- `explicit_unknown_fields`
- `corrections`
- `evidence`, containing `{field, quote}` items from the newest user turn

All values may be absent when not stated. `heart_disease=false` is valid only when the user explicitly denies it. `explicit_unknown_fields` represents statements such as “tôi không biết bệnh tim”, which must not be converted to `false`.

Pydantic performs the authoritative validation. If DeepSeek returns malformed arguments, unsupported categories, out-of-range values, or evidence not found in the newest message, the extraction is rejected and audited. There is no regex extraction fallback and no prediction fallback.

When DeepSeek is unavailable, the response explains that structured extraction is temporarily unavailable and asks the user to retry. The system does not silently fill the profile or run the model.

## 6. Blood-Pressure Rules

The extractor records raw systolic and diastolic numbers; application code classifies them.

For non-pregnant adults aged 18 or older, use the 2025 AHA/ACC categories:

- normal: systolic below 120 and diastolic below 80;
- elevated: systolic 120-129 and diastolic below 80;
- stage 1: systolic 130-139 or diastolic 80-89;
- stage 2: systolic at least 140 or diastolic at least 90;
- severe: systolic above 180 and/or diastolic above 120.

For adults, these classification thresholds do not vary by sex. A single reading is presented as a screening classification, not a confirmed diagnosis.

For users younger than 18, pregnant users, or users supplying only one pressure number, the application does not infer `hypertension` from the measurement. It asks for the missing context or for a clinician-confirmed hypertension history.

Severe readings and acute neurological symptoms route to safety guidance before prediction.

Reference: https://professional.heart.org/en/science-news/2025-high-blood-pressure-guideline/top-things-to-know

## 7. LangGraph State and Flow

### 7.1 State

The graph state contains:

- `session_id`, `owner_id`, `user_message_id`;
- recent `messages`;
- `current_profile`;
- `extracted_update`;
- `validation_issues`;
- `missing_required_fields`;
- `intent` and `safety_level`;
- `prediction_result` and `assistant_card`;
- `reply`.

### 7.2 Nodes

```text
START
  -> persist_user_message
  -> classify_intent_and_triage
       -> emergency_response -> persist_assistant_message -> END
       -> general_health_chat -> persist_assistant_message -> END
       -> extract_screening_profile
            -> validate_and_merge_profile
                 -> ask_for_missing_fields -> persist_assistant_message -> END
                 -> predict_stroke
                      -> build_prediction_card
                      -> persist_prediction_and_message
                      -> END
```

The graph uses explicit conditional edges. The extraction tool cannot route directly to `predict_stroke`.

LangGraph is compiled with PostgreSQL `PostgresSaver`, using `session_id` as `thread_id`. Its `setup()` method owns the internal checkpoint tables. Application migrations do not handcraft or alter those internal tables.

Redis remains optional for rate limiting and short-lived cache only. PostgreSQL is the source of truth for users, ownership, sessions, messages, cards, profiles, and predictions.

## 8. Missing-Field Conversation Contract

The model runs only when all six required fields are valid after merging prior turns.

If data is missing, the assistant lists only missing or invalid fields and provides a concrete response format. Examples:

- blood pressure: `Huyết áp 120/80 mmHg`;
- heart disease: `Đã từng được bác sĩ chẩn đoán bệnh tim: Có / Không / Không rõ`;
- glucose: `Glucose trung bình 70 mg/dL`;
- BMI: `BMI 25`.

`Không rõ` remains missing and prevents prediction. The assistant explains why the field matters without claiming the user has the condition.

The newest explicit correction wins. Every overwritten field retains its prior audit evidence through tool logs and prediction snapshots.

## 9. API Contracts

### `POST /api/chat/message`

Returns the existing chat response plus persisted message IDs:

```json
{
  "session_id": "uuid",
  "reply": "string",
  "messages": [],
  "assistant_card": null,
  "profile_status": {
    "detected_fields": ["age", "gender"],
    "missing_fields": ["hypertension", "heart_disease", "avg_glucose_level", "bmi"]
  }
}
```

### `GET /api/sessions`

Returns only sessions owned by the authenticated user or current anonymous-owner cookie.

### `GET /api/sessions/{session_id}/messages`

Returns each persisted message with `id`, `role`, `content`, `time`, and `assistant_card`. It returns 404 for a nonexistent session and 403 for a session owned by another identity.

### Rename and delete

Rename remains non-destructive. `Phiên mới` creates a fresh session and never deletes history. Explicit `DELETE` soft-deletes the session and removes its Redis cache; a later retention job may perform physical deletion.

## 10. Error Handling and Safety

- Database writes for a user turn, profile update, prediction, and assistant response use transactions appropriate to their node boundary.
- A database error returns a controlled response and never claims data was saved.
- DeepSeek timeouts/provider errors are audited without credentials.
- Tool output is untrusted until Pydantic validation passes.
- Negated symptoms are not treated as positive evidence solely by substring matching.
- Emergency triage remains conservative but context-aware. It distinguishes positive, negative, uncertain, and historical symptom mentions.
- Prediction never runs on default-filled required screening fields.
- Existing model metadata remains authoritative for the clinical decision threshold.

## 11. Test Strategy

Implementation follows red-green-refactor.

### Migration and repository tests

- migration creates the schema, extensions, tables, constraints, and indexes in an isolated test database;
- running the migration twice is safe;
- ownership filters prevent cross-user session access;
- assistant cards round-trip through PostgreSQL;
- deleting one session cannot delete another owner’s history.

### Extraction and graph tests

- DeepSeek tool arguments parse `glucose tầm 70` and `BMI là 25`;
- multi-turn fields merge without losing earlier valid values;
- explicit corrections replace previous values;
- `không biết bệnh tim` remains unknown;
- malformed or hallucinated evidence is rejected;
- prediction is not called when any required field is missing;
- prediction is called exactly once when the sixth field becomes valid;
- `không yếu tay` does not independently trigger an emergency;
- provider failure does not fall back to regex or partial prediction.

### API and frontend regression tests

- reload preserves the prediction card;
- opening a history session restores messages and cards;
- session lists are owner-scoped;
- `Phiên mới` preserves old sessions;
- explicit delete only removes the selected session;
- test configuration cannot connect to the production Redis or PostgreSQL database;
- backend contract suite and frontend production build pass.

## 12. Delivery Boundaries

This approved slice includes:

1. PostgreSQL application migration and database repository layer.
2. Persistent auth/session/message/profile/prediction storage.
3. DeepSeek extraction tool and explicit LangGraph workflow.
4. Strict required-field prediction gate.
5. Persisted assistant cards and owner-scoped history APIs.
6. Relevant backend, API, and frontend regression tests.

It does not include full document ingestion, embedding generation, RAG answer generation, or admin document UI implementation. Their tables are created now to preserve the already approved long-memory schema.

Creating tables in `stroke_ai` is authorized by the user. Reloading the production backend or changing the public deployment requires a separate explicit confirmation after local verification.

## 13. Dependencies and Configuration

Add pinned compatible versions of:

- `psycopg[binary,pool]`;
- `langgraph`;
- `langgraph-checkpoint-postgres`.

Add `DATABASE_URL` for `stroke_ai`. Keep credentials only in the ignored backend `.env` and a redacted `.env.example`. Do not log connection strings or DeepSeek keys.

The current live backend reports `deepseek_configured=false`. Live tool extraction requires `DEEPSEEK_API_KEY` to be configured before deployment verification.

## 14. Acceptance Criteria

The work is accepted when:

1. `stroke_ai` contains the documented application tables, constraints, and indexes.
2. PostgreSQL, not Redis, is the durable source of truth for chat history.
3. Natural Vietnamese screening input is extracted through the DeepSeek tool without regex NLP fallback.
4. The stroke model cannot run until the six required fields are present and valid.
5. Blood-pressure numbers are stored and classified safely according to age/context rules.
6. Prediction cards survive reload and history navigation.
7. Sessions and messages are inaccessible to other users/anonymous owners.
8. Follow-up questions use conversation state and do not repeat the same static symptom template.
9. Production data cannot be cleared by the automated test suite.
10. Focused tests, full backend contract tests, and the frontend production build complete successfully before any deployment reload.
