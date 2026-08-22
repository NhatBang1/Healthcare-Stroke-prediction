BEGIN;

SELECT pg_advisory_xact_lock(hashtext('strokeguard:migration:001'));

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS strokeguard;

CREATE TABLE IF NOT EXISTS strokeguard.schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email CITEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    password_salt TEXT,
    password_hash TEXT,
    provider TEXT,
    provider_id TEXT,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT users_auth_method_required CHECK (
        (password_salt IS NOT NULL AND password_hash IS NOT NULL)
        OR (provider IS NOT NULL AND provider_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS strokeguard.auth_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES strokeguard.users(id) ON DELETE CASCADE,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.chat_sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES strokeguard.users(id) ON DELETE CASCADE,
    anonymous_owner_hash TEXT,
    title TEXT NOT NULL DEFAULT 'Cuộc hội thoại',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT chat_sessions_owner_required CHECK (
        user_id IS NOT NULL OR anonymous_owner_hash IS NOT NULL
    )
);

CREATE TABLE IF NOT EXISTS strokeguard.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES strokeguard.chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    assistant_card JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.patient_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID UNIQUE NOT NULL REFERENCES strokeguard.chat_sessions(id) ON DELETE CASCADE,
    age DOUBLE PRECISION CHECK (age BETWEEN 0 AND 100),
    gender TEXT CHECK (gender IN ('male', 'female')),
    systolic_bp DOUBLE PRECISION CHECK (systolic_bp BETWEEN 50 AND 300),
    diastolic_bp DOUBLE PRECISION CHECK (diastolic_bp BETWEEN 30 AND 200),
    blood_pressure_category TEXT CHECK (
        blood_pressure_category IN ('normal', 'elevated', 'stage_1', 'stage_2', 'severe', 'unknown')
    ),
    hypertension BOOLEAN,
    heart_disease BOOLEAN,
    avg_glucose_level DOUBLE PRECISION CHECK (avg_glucose_level BETWEEN 40 AND 400),
    bmi DOUBLE PRECISION CHECK (bmi BETWEEN 10 AND 80),
    avg_resting_bp DOUBLE PRECISION,
    avg_max_hr DOUBLE PRECISION,
    avg_oldpeak DOUBLE PRECISION,
    heart_disease_rate DOUBLE PRECISION,
    cardio_cholesterol DOUBLE PRECISION,
    cardio_gluc DOUBLE PRECISION,
    cardio_rate DOUBLE PRECISION,
    smoke_flag BOOLEAN,
    alco_flag BOOLEAN,
    ever_married TEXT,
    work_type TEXT,
    residence_type TEXT,
    smoking_status TEXT,
    extra_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.stroke_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES strokeguard.chat_sessions(id) ON DELETE CASCADE,
    profile_id UUID REFERENCES strokeguard.patient_profiles(id) ON DELETE SET NULL,
    source_message_id UUID REFERENCES strokeguard.chat_messages(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    risk_probability DOUBLE PRECISION NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_label TEXT NOT NULL CHECK (risk_label IN ('low_risk', 'medium_risk', 'high_risk')),
    threshold DOUBLE PRECISION NOT NULL,
    input_features JSONB NOT NULL,
    filled_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.tool_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES strokeguard.chat_sessions(id) ON DELETE CASCADE,
    message_id UUID REFERENCES strokeguard.chat_messages(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'validation_error', 'provider_error')),
    input_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms INTEGER CHECK (latency_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.memory_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES strokeguard.chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES strokeguard.users(id) ON DELETE SET NULL,
    summary_text TEXT NOT NULL,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'admin_upload' CHECK (
        source_type IN ('admin_upload', 'seed_document', 'guideline', 'faq', 'research')
    ),
    storage_path TEXT NOT NULL,
    language TEXT,
    version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK (
        status IN ('uploaded', 'indexing', 'indexed', 'failed', 'deleted')
    ),
    uploaded_by UUID REFERENCES strokeguard.users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES strokeguard.documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (
        status IN ('queued', 'parsing', 'chunking', 'embedding', 'indexed', 'failed')
    ),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strokeguard.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES strokeguard.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    chunk_text TEXT NOT NULL,
    page_number INTEGER CHECK (page_number IS NULL OR page_number >= 1),
    section_title TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(section_title, '') || ' ' || chunk_text)
    ) STORED,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires
    ON strokeguard.auth_sessions (expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
    ON strokeguard.chat_sessions (user_id, updated_at DESC)
    WHERE status <> 'deleted';
CREATE INDEX IF NOT EXISTS idx_chat_sessions_anonymous_updated
    ON strokeguard.chat_sessions (anonymous_owner_hash, updated_at DESC)
    WHERE status <> 'deleted';
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON strokeguard.chat_messages (session_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_stroke_predictions_session_created
    ON strokeguard.stroke_predictions (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_audit_session_created
    ON strokeguard.tool_audit_logs (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON strokeguard.documents (status);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_active
    ON strokeguard.document_chunks (document_id, is_active);
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector
    ON strokeguard.document_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON strokeguard.document_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE is_active = true AND embedding IS NOT NULL;

INSERT INTO strokeguard.schema_migrations (version, description)
VALUES ('001', 'Create durable StrokeGuard application schema')
ON CONFLICT (version) DO NOTHING;

COMMIT;
