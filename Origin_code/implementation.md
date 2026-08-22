# Stroke AI Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checklist syntax in the phase sections for tracking.

**Goal:** Build a production-ready FE/BE chatbot web app around the trained stroke prediction artifacts in this repository.

**Architecture:** The frontend is a one-page Messenger-style React chat UI. The backend is a FastAPI service that uses LangGraph to orchestrate DeepSeek, Redis short memory, PostgreSQL long memory, RAG over `pgvector`, and the trained clinical stroke prediction model.

**Tech Stack:** React, TypeScript, Vite, Tailwind CSS, FastAPI, LangGraph, DeepSeek API, Redis, PostgreSQL, `pgvector`, `sentence-transformers`, `joblib`, scikit-learn, XGBoost.

---

## 1. Current Repository Assessment

This repository is currently an ML/research project, not a web application.

Existing assets to reuse:

- `Code/Model.py`: full training pipeline, preprocessing, feature definitions, threshold optimization, artifact export.
- `Code/evaluate_model.py`: saved model evaluation script.
- `Code/test_model.cpp`: C++ edge inference smoke test.
- `artifacts/stroke_model_clinical.joblib`: clinical production model.
- `artifacts/stroke_model.joblib`: lightweight edge/fallback XGBoost model.
- `artifacts/calories_model.joblib`: model used to generate `Estimated_calories`.
- `artifacts/preprocess_metadata.json`: preprocessing medians, categorical modes, selected feature columns, and decision thresholds.
- `artifacts/stroke_predictor_model.h`: C++ header for edge inference.
- `artifacts/model_comparison.csv`: baseline model comparison table.
- `Documents/Final_paper_IEEE.pdf`: initial RAG corpus document.

Environment note:

- Runtime verification was attempted, but the current shell environment does not have `joblib` / `pandas` installed. The implementation should create a dedicated backend virtualenv or container and install backend dependencies before running `Code/evaluate_model.py`.

---

## 2. Model Decision

Use `artifacts/stroke_model_clinical.joblib` as the default backend prediction model.

Reasoning:

- `Model.py` explicitly trains this as the final clinical model: `CalibratedClassifierCV` over a `StackingClassifier`.
- It uses the full 25-feature clinical feature set from `preprocess_metadata.json`.
- It is calibrated with sigmoid, which is better for communicating risk probability than a raw classifier score.
- `preprocess_metadata.json` marks it as the model for clinical reporting.

Use `artifacts/stroke_model.joblib` only as an edge/fallback model.

Reasoning:

- It uses only 6 features: `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `Estimated_calories`.
- It has ONNX and C++ exports, so it is suitable for low-latency or embedded inference.
- It loses clinical context compared with the 25-feature model.

Thresholds from `artifacts/preprocess_metadata.json`:

- Clinical model threshold: `decision_threshold_clinical = 0.03`
- Edge model threshold: `decision_threshold_edge = 0.605`

Baseline comparison from `artifacts/model_comparison.csv`:

- Best F1 in baseline comparison: `XGBoost`, F1 `0.3006`, AUC-ROC `0.8586`, recall `0.6842`.
- Best AUC in baseline comparison: `GradientBoosting`, AUC-ROC `0.8596`.
- These baseline rows do not override the final clinical model choice because the final pipeline separately trains and saves `stroke_model_clinical.joblib` for clinical reporting.

---

## 3. Product Goal

Build a chatbot web app for stroke-related Q&A and personal risk screening.

The system must:

- Answer user questions about stroke symptoms, risk factors, prevention, and interpretation of health indicators.
- Extract health indicators from natural Vietnamese or English text.
- Ask focused follow-up questions when required indicators are missing.
- Call the trained stroke model when enough data exists.
- Return a risk probability, risk label, threshold, and explanation.
- Use RAG over uploaded/internal documents to reduce hallucination.
- Use DeepSeek for response synthesis.
- Use Redis for short-term session memory.
- Use PostgreSQL for long-term history, user/admin data, uploaded documents, vector embeddings, and prediction audit.
- Provide admin document upload that triggers chunking, embedding, and storage in PostgreSQL with `pgvector`.

Medical safety boundary:

- This app provides educational information and risk screening, not diagnosis.
- If the user mentions acute red flags such as face drooping, arm weakness, speech difficulty, sudden severe headache, vision loss, or one-sided numbness, the assistant must prioritize emergency guidance and tell the user to call local emergency services immediately.

---

## 4. Target Architecture

Recommended stack:

- Frontend: React + TypeScript + Vite.
- Styling: Tailwind CSS with custom design tokens.
- Realtime transport: WebSocket for chat streaming events.
- Backend API: FastAPI.
- Agent orchestration: LangGraph.
- LLM: DeepSeek API through OpenAI-compatible SDK.
- Short memory: Redis.
- Long memory: PostgreSQL.
- RAG vector store: PostgreSQL + `pgvector`.
- Embeddings: local multilingual embedding model, recommended `BAAI/bge-m3`.
- Reranker: local multilingual reranker, recommended `BAAI/bge-reranker-v2-m3`.
- ML inference: Python `joblib` model service for clinical model, optional C++ edge binary for fallback benchmark.
- Auth: simple JWT/session auth if multi-user/admin is required; otherwise local anonymous session for MVP plus admin password.

High-level flow:

```text
Browser Chat UI
  -> WebSocket /ws/chat/{session_id}
  -> FastAPI Chat Gateway
  -> LangGraph
       -> safety_triage_tool
       -> memory_recall_tool
       -> intent_router_tool
       -> patient_profile_extraction_tool
       -> feature_validation_tool
       -> rag_retrieval_tool
       -> stroke_prediction_tool
       -> response_synthesis_node using DeepSeek
       -> memory_write_tool
  -> Stream events back to Browser

Admin Upload UI
  -> POST /api/admin/documents
  -> Document Ingestion Worker
       -> parse file
       -> chunk
       -> embed
       -> store document_chunks.embedding in PostgreSQL vector column
```

---

## 5. Proposed Repository Structure

Add these directories:

```text
Healthcare-Stroke-prediction/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── security.py
│   │   ├── api/
│   │   │   ├── routes_chat.py
│   │   │   ├── routes_prediction.py
│   │   │   ├── routes_documents.py
│   │   │   └── routes_sessions.py
│   │   ├── db/
│   │   │   ├── session.py
│   │   │   ├── models.py
│   │   │   └── repositories.py
│   │   ├── graph/
│   │   │   ├── state.py
│   │   │   ├── workflow.py
│   │   │   ├── nodes.py
│   │   │   └── tools.py
│   │   ├── llm/
│   │   │   ├── deepseek_client.py
│   │   │   └── prompts.py
│   │   ├── memory/
│   │   │   ├── redis_memory.py
│   │   │   └── long_memory.py
│   │   ├── prediction/
│   │   │   ├── feature_schema.py
│   │   │   ├── preprocess.py
│   │   │   ├── model_loader.py
│   │   │   └── service.py
│   │   ├── rag/
│   │   │   ├── loaders.py
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   ├── retrieval.py
│   │   │   └── ingestion.py
│   │   └── schemas/
│   │       ├── chat.py
│   │       ├── prediction.py
│   │       └── documents.py
│   ├── scripts/
│   │   ├── create_database.sh
│   │   ├── init_db.sql
│   │   ├── ingest_existing_documents.py
│   │   └── compile_edge_model.sh
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx
│   │   │   └── routes.tsx
│   │   ├── features/chat/
│   │   │   ├── ChatPage.tsx
│   │   │   ├── components/
│   │   │   │   ├── ChatHeader.tsx
│   │   │   │   ├── MessageList.tsx
│   │   │   │   ├── MessageBubble.tsx
│   │   │   │   ├── ChatComposer.tsx
│   │   │   │   ├── TypingIndicator.tsx
│   │   │   │   ├── RiskSummaryCard.tsx
│   │   │   │   ├── QuickReplyChips.tsx
│   │   │   │   └── DocumentSourceBadge.tsx
│   │   │   ├── chatStore.ts
│   │   │   ├── chatSocket.ts
│   │   │   └── types.ts
│   │   ├── features/admin/
│   │   │   ├── AdminDocumentsPage.tsx
│   │   │   └── documentApi.ts
│   │   └── styles/
│   │       └── globals.css
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
└── implementation.md
```

---

## 6. Frontend Specification

### 6.1 UI Direction

The frontend should feel like a polished healthcare Messenger, not a dashboard.

Required visual behavior:

- Single full-page chat experience.
- Main chat panel centered on desktop.
- No heavy dashboard sidebar for the user view.
- Message bubbles clearly separated between user and AI.
- AI avatar and name visible in the header.
- Input composer fixed at the bottom of the chat panel.
- Typing indicator with three animated dots.
- Streaming assistant responses rendered incrementally.
- Prediction results rendered as a dedicated card, not buried as plain text.
- RAG citations rendered as compact source badges below relevant assistant messages.
- Mobile layout must be one column and keep the input reachable.

Suggested design tokens:

```css
:root {
  --bg: #edf7f4;
  --bg-panel: rgba(255, 255, 255, 0.86);
  --ink: #10201d;
  --muted: #64756f;
  --brand: #0f766e;
  --brand-strong: #115e59;
  --user-bubble: #0f766e;
  --assistant-bubble: #ffffff;
  --warning: #b45309;
  --danger: #b91c1c;
  --border: rgba(15, 118, 110, 0.16);
  --shadow: 0 24px 80px rgba(17, 94, 89, 0.16);
}
```

Suggested typography:

- Primary UI font: `Manrope` or `Plus Jakarta Sans`.
- Content font: `Source Serif 4` only for long educational answers if desired.
- Avoid default Arial/system-only styling unless the project later adopts a design system.

### 6.2 User Chat Components

Required components:

- `ChatPage`: owns layout, session creation, socket lifecycle.
- `ChatHeader`: app name, AI online state, reset session action.
- `MessageList`: virtualized or scroll-managed message list.
- `MessageBubble`: supports user, assistant, system, error, and safety messages.
- `TypingIndicator`: animated dots while the graph is working.
- `ChatComposer`: textarea, send button, Enter/Shift+Enter behavior.
- `QuickReplyChips`: suggested follow-up answers such as `Có`, `Không`, `Tôi không biết`.
- `RiskSummaryCard`: probability, label, threshold, model name, key detected indicators.
- `DocumentSourceBadge`: document title, page/section, confidence score.
- `ConnectionStatus`: connected, reconnecting, offline, failed.

### 6.3 Admin Components

Admin can be minimal for MVP:

- `AdminDocumentsPage`: upload documents, view ingestion status, list indexed documents.
- `DocumentUploadDropzone`: PDF/DOCX/TXT/MD upload.
- `IngestionJobStatus`: queued, parsing, chunking, embedding, indexed, failed.
- `DocumentList`: title, source type, version, chunk count, embedding model.

### 6.4 Frontend State

Use Zustand for chat state because it is small and event-driven.

State shape:

```ts
type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  status: "sending" | "streaming" | "done" | "error";
  sources?: RagSource[];
  riskResult?: StrokePredictionResult;
};

type ChatState = {
  sessionId: string | null;
  messages: ChatMessage[];
  isTyping: boolean;
  connectionStatus: "connecting" | "connected" | "reconnecting" | "offline" | "failed";
  detectedProfile: Partial<PatientProfile>;
  activeRiskResult: StrokePredictionResult | null;
};
```

### 6.5 WebSocket Event Contract

Client to server:

```json
{
  "type": "user_message",
  "message_id": "uuid",
  "session_id": "uuid",
  "content": "Tôi 65 tuổi, bị tăng huyết áp, glucose 170, BMI 29."
}
```

Server to client:

```json
{ "type": "assistant_typing", "value": true }
```

```json
{
  "type": "assistant_delta",
  "message_id": "uuid",
  "content": "Tôi đã ghi nhận các chỉ số..."
}
```

```json
{
  "type": "profile_detected",
  "profile": {
    "age": 65,
    "hypertension": 1,
    "avg_glucose_level": 170,
    "bmi": 29
  },
  "missing_fields": ["gender", "heart_disease"]
}
```

```json
{
  "type": "prediction_result",
  "message_id": "uuid",
  "result": {
    "model_used": "stroke_model_clinical.joblib",
    "risk_probability": 0.084,
    "risk_label": "high_risk",
    "threshold": 0.03,
    "disclaimer": "Kết quả này là sàng lọc nguy cơ, không phải chẩn đoán."
  }
}
```

```json
{
  "type": "rag_sources",
  "message_id": "uuid",
  "sources": [
    {
      "document_id": "uuid",
      "title": "Final_paper_IEEE.pdf",
      "page": 3,
      "chunk_id": "uuid",
      "score": 0.82
    }
  ]
}
```

```json
{
  "type": "assistant_done",
  "message_id": "uuid"
}
```

```json
{
  "type": "error",
  "code": "MODEL_UNAVAILABLE",
  "message": "Model prediction service is temporarily unavailable."
}
```

### 6.6 FE Prompt For UI Generation

Use this prompt if generating the frontend UI with an AI UI tool:

```text
Design a one-page healthcare chatbot web app for stroke risk Q&A and screening. The app should feel like a premium Messenger-style chat experience, not a dashboard.

Requirements:
- Only one main user flow: chat between user and AI.
- Centered chat frame with a polished header, scrollable message thread, and fixed bottom composer.
- User and AI message bubbles must be visually distinct.
- AI typing indicator uses three animated dots.
- Support streaming assistant messages.
- When prediction is returned, show a separate risk card with probability, risk label, decision threshold, model used, and medical disclaimer.
- When RAG is used, show compact citation/source badges below the assistant message.
- Add quick reply chips for missing medical fields.
- Visual style: clean, trustworthy healthcare, modern teal/green palette, soft depth, subtle background gradients, no generic purple SaaS style.
- Desktop layout: centered chat panel with optional slim contextual side rail for detected indicators.
- Mobile layout: single column, input always reachable, no sidebar.
- Include loading, empty, error, reconnecting, and disabled-send states.
- Copy should be Vietnamese-first and concise.
```

---

## 7. Backend Specification

### 7.1 Backend Responsibilities

FastAPI backend must provide:

- Chat WebSocket gateway.
- REST fallback chat endpoint.
- Prediction endpoint.
- LangGraph workflow execution.
- DeepSeek client wrapper.
- Redis short memory.
- PostgreSQL persistence.
- RAG ingestion and retrieval.
- Admin document upload.
- Health checks and model readiness checks.

### 7.2 Backend Dependencies

Create `backend/requirements.txt`:

```text
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-multipart
sqlalchemy
asyncpg
psycopg[binary,pool]
redis
langgraph
langchain-core
openai
joblib
pandas
numpy
scikit-learn
xgboost
pgvector
sentence-transformers
rank-bm25
pypdf
python-docx
unstructured
pytest
pytest-asyncio
httpx
```

### 7.3 Environment Variables

Use `.env`:

```text
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

DATABASE_URL=postgresql+asyncpg://stroke:stroke_password@localhost:5432/stroke_ai
DATABASE_SYNC_URL=postgresql://stroke:stroke_password@localhost:5432/stroke_ai
REDIS_URL=redis://localhost:6379/0

DEEPSEEK_API_KEY=replace_me
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_FAST_MODEL=deepseek-v4-flash

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

ARTIFACT_DIR=../artifacts
CLINICAL_MODEL_PATH=../artifacts/stroke_model_clinical.joblib
EDGE_MODEL_PATH=../artifacts/stroke_model.joblib
CALORIES_MODEL_PATH=../artifacts/calories_model.joblib
PREPROCESS_METADATA_PATH=../artifacts/preprocess_metadata.json

UPLOAD_DIR=./uploads
JWT_SECRET=replace_me
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change_me
```

DeepSeek configuration note:

- As of 2026-07-19, official DeepSeek docs list `deepseek-v4-pro` and `deepseek-v4-flash` for chat completions. The legacy `deepseek-chat` and `deepseek-reasoner` names are documented as deprecated on 2026-07-24 15:59 UTC.
- Use `deepseek-v4-pro` for medical response synthesis and extraction when accuracy matters.
- Use `deepseek-v4-flash` for low-cost routing, summarization, and short follow-up generation.
- Keep model names in env vars, not hardcoded in LangGraph nodes.

### 7.4 API Endpoints

Health:

- `GET /health`: app alive.
- `GET /ready`: database, Redis, model artifacts, and vector extension ready.

Chat:

- `POST /api/chat/message`: non-stream fallback.
- `WS /ws/chat/{session_id}`: primary streaming chat.
- `GET /api/sessions`: list chat sessions.
- `GET /api/sessions/{session_id}/messages`: load history.
- `DELETE /api/sessions/{session_id}`: delete conversation and short memory.

Prediction:

- `POST /api/predict/stroke`: direct structured prediction.
- `GET /api/predict/schema`: required and optional fields.

Documents:

- `POST /api/admin/documents`: upload document and enqueue ingestion.
- `POST /api/admin/documents/{document_id}/ingest`: force re-ingestion.
- `GET /api/admin/documents`: list uploaded documents.
- `GET /api/admin/ingestion-jobs`: list ingestion status.
- `DELETE /api/admin/documents/{document_id}`: soft-delete document and chunks.

RAG:

- `POST /api/rag/search`: debug retrieval endpoint for admin/development.

---

## 8. Prediction Pipeline

### 8.1 Input Fields

The clinical model expects these source fields before preprocessing:

Numeric:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `avg_glucose_level`
- `bmi`
- `avg_RestingBP`
- `avg_MaxHR`
- `avg_Oldpeak`
- `heart_disease_rate`
- `cardio_cholesterol`
- `cardio_gluc`
- `cardio_rate`
- `smoke_flag`
- `alco_flag`

Categorical:

- `ever_married`
- `work_type`
- `Residence_type`
- `smoking_status`

Generated:

- `Estimated_calories`, predicted from `calories_model.joblib` using `Gender`, `Age`, `BMI`.

Final 25 model columns, exactly in this order:

```text
age
hypertension
heart_disease
avg_glucose_level
bmi
avg_RestingBP
avg_MaxHR
avg_Oldpeak
heart_disease_rate
cardio_cholesterol
cardio_gluc
cardio_rate
gender
smoke_flag
alco_flag
Estimated_calories
ever_married_Yes
work_type_Never_worked
work_type_Private
work_type_Self-employed
work_type_children
Residence_type_Urban
smoking_status_never smoked
smoking_status_smokes
smoking_status_unknown
```

### 8.2 Normalization Rules

Use the same rules as `Code/Model.py`:

- `gender`: `male -> 0`, `female -> 1`.
- `bmi`: values below `10` or above `80` are treated as missing.
- Missing numeric values use `numeric_fill` from `preprocess_metadata.json`.
- Missing categorical values use `categorical_fill` from `preprocess_metadata.json`.
- Drop leaky/noisy columns: `avg_weight`, `avg_height`, `avg_ap_hi`, `avg_ap_lo`, `heart_cholesterol`.
- One-hot encode categorical columns with `drop_first=True`.
- Reindex to the selected feature columns from metadata.

### 8.3 User-Friendly Field Mapping

The NLP extraction layer should map natural text to model fields:

```text
"tuổi", "age" -> age
"nam", "male" -> gender = 0
"nữ", "female" -> gender = 1
"cao huyết áp", "tăng huyết áp", "hypertension" -> hypertension
"bệnh tim", "heart disease" -> heart_disease
"đường huyết", "glucose", "blood sugar" -> avg_glucose_level
"BMI", "chỉ số khối cơ thể" -> bmi
"huyết áp nghỉ", "resting BP" -> avg_RestingBP
"nhịp tim tối đa", "max HR" -> avg_MaxHR
"oldpeak", "ST depression" -> avg_Oldpeak
"hút thuốc" -> smoke_flag or smoking_status
"uống rượu" -> alco_flag
"đã kết hôn" -> ever_married
"công việc" -> work_type
"thành thị/nông thôn" -> Residence_type
```

### 8.4 Risk Labels

Use probability relative to the selected threshold:

```text
high_risk:
  probability >= threshold

medium_risk:
  threshold * 0.5 <= probability < threshold

low_risk:
  probability < threshold * 0.5
```

For the clinical model, threshold `0.03` is intentionally low. Explain this to users as a screening threshold optimized to avoid missing possible risk, not as a diagnosis boundary.

### 8.5 Direct Prediction Response

`POST /api/predict/stroke` response:

```json
{
  "model_used": "stroke_model_clinical.joblib",
  "model_version": "local-artifacts-2026-07-19",
  "risk_probability": 0.084,
  "risk_label": "high_risk",
  "threshold": 0.03,
  "detected_features": {
    "age": 65,
    "gender": 0,
    "hypertension": 1,
    "avg_glucose_level": 170,
    "bmi": 29
  },
  "filled_features": {
    "avg_MaxHR": 141.0967742,
    "cardio_gluc": 1.0
  },
  "missing_user_fields": ["heart_disease"],
  "disclaimer": "Kết quả này chỉ dùng để sàng lọc nguy cơ, không phải chẩn đoán y khoa."
}
```

---

## 9. LangGraph Design

LangGraph should be the core orchestration layer. Official LangGraph docs distinguish short-term thread-scoped state through checkpointers and long-term cross-thread application data through stores. In this project, Redis handles fast short memory while PostgreSQL stores durable memory and audit records.

### 9.1 Graph State

```python
from typing import Any, Literal, TypedDict

class PatientProfile(TypedDict, total=False):
    age: float
    gender: int
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    avg_RestingBP: float
    avg_MaxHR: float
    avg_Oldpeak: float
    heart_disease_rate: float
    cardio_cholesterol: float
    cardio_gluc: float
    cardio_rate: float
    smoke_flag: int
    alco_flag: int
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str

class GraphState(TypedDict):
    session_id: str
    user_id: str | None
    user_message: str
    messages: list[dict[str, Any]]
    intent: Literal["emergency", "prediction", "medical_qa", "smalltalk", "admin", "unknown"]
    patient_profile: PatientProfile
    missing_fields: list[str]
    rag_query: str | None
    rag_sources: list[dict[str, Any]]
    prediction_result: dict[str, Any] | None
    memory_summary: str | None
    response_text: str
    safety_flags: list[str]
```

### 9.2 Graph Nodes

Recommended nodes:

```text
start
  -> safety_triage_node
  -> memory_recall_node
  -> intent_router_node

intent_router_node:
  emergency -> emergency_response_node -> memory_write_node -> end
  prediction -> entity_extraction_node -> feature_validation_node
  medical_qa -> rag_retrieval_node -> response_synthesis_node -> memory_write_node -> end
  smalltalk -> bounded_smalltalk_node -> memory_write_node -> end

feature_validation_node:
  enough_data -> prediction_node -> rag_retrieval_node -> response_synthesis_node -> memory_write_node -> end
  missing_data -> followup_question_node -> memory_write_node -> end
```

### 9.3 Required Tools

Tools already requested by user:

- `stroke_prediction_tool`: call trained model.
- `rag_retrieval_tool`: retrieve internal documents.

Additional tools recommended:

- `safety_triage_tool`: detect FAST symptoms and urgent red flags before normal chat.
- `intent_router_tool`: classify emergency, prediction, medical Q&A, small talk, admin.
- `extract_patient_profile_tool`: convert natural language into structured medical fields.
- `unit_normalization_tool`: normalize units such as glucose `mg/dL` vs `mmol/L`, BMI, blood pressure, heart rate.
- `required_field_validator_tool`: determine whether prediction is possible or follow-up is needed.
- `followup_question_tool`: generate 1-3 short questions for missing fields.
- `risk_explanation_tool`: explain model result without overclaiming diagnosis.
- `memory_dedup_tool`: compare current answer with recent answers to avoid repeating.
- `conversation_summary_tool`: compress old chat turns for long sessions.
- `source_citation_tool`: attach retrieved document title, page, and chunk id.
- `medical_scope_guard_tool`: refuse or redirect non-stroke medical requests when outside app scope.
- `bmi_calculator_tool`: compute BMI if user provides height and weight instead of BMI.
- `audit_log_tool`: persist tool calls, prediction inputs, and model output for traceability.

### 9.4 DeepSeek System Prompt

Use this as the main system prompt:

```text
Bạn là trợ lý AI y tế chuyên về đột quỵ, dùng cho giáo dục sức khỏe và sàng lọc nguy cơ. Bạn không phải bác sĩ và không được chẩn đoán chắc chắn.

Mục tiêu:
- Trả lời đúng trọng tâm câu hỏi của người dùng.
- Ưu tiên an toàn y khoa.
- Dùng tiếng Việt tự nhiên, ngắn gọn, rõ ràng.
- Khi người dùng cung cấp chỉ số sức khỏe, trích xuất dữ liệu có cấu trúc trước khi kết luận.
- Khi đủ dữ liệu, gọi tool dự đoán nguy cơ đột quỵ trước khi giải thích.
- Khi thiếu dữ liệu, hỏi lại tối đa 1-3 câu ngắn và chỉ hỏi các trường quan trọng nhất.
- Khi trả lời kiến thức, ưu tiên tài liệu nội bộ đã truy xuất bằng RAG và trích nguồn.

Guardrails:
- Không tự bịa chỉ số, xác suất, nguồn tài liệu, guideline hoặc khuyến nghị.
- Không trả lời lan man ngoài chủ đề đột quỵ, yếu tố nguy cơ, triệu chứng, phòng ngừa, xét nghiệm/chỉ số liên quan, hoặc cách hiểu kết quả mô hình.
- Không khẳng định người dùng chắc chắn bị hoặc không bị đột quỵ.
- Không hướng dẫn tự điều trị thay cho bác sĩ.
- Nếu có dấu hiệu cấp cứu như méo miệng, yếu/tê một bên cơ thể, nói khó, nhìn mờ đột ngột, đau đầu dữ dội đột ngột, mất thăng bằng đột ngột, hãy ưu tiên khuyến nghị gọi cấp cứu ngay.
- Nếu tài liệu RAG không đủ liên quan, nói rõ "Tôi chưa có đủ dữ liệu từ tài liệu nội bộ để trả lời chắc chắn".
- Nếu nội dung đã được trả lời gần đây, tóm tắt ngắn thay vì lặp lại nguyên văn.

Format ưu tiên:
- Tóm tắt ngắn.
- Chỉ số đã hiểu.
- Chỉ số còn thiếu nếu có.
- Kết quả nguy cơ nếu đã gọi model.
- Khuyến nghị an toàn.
- Nguồn tham khảo nếu dùng RAG.
```

### 9.5 Prediction Branch Prompt

```text
Bạn đang xử lý yêu cầu đánh giá nguy cơ đột quỵ cá nhân.

Hãy thực hiện theo thứ tự:
1. Dựa trên structured profile đã được tool trích xuất, không tự thêm dữ liệu.
2. Nếu thiếu chỉ số quan trọng, hỏi lại ngắn gọn.
3. Nếu có kết quả model, giải thích xác suất, risk label, threshold và ý nghĩa sàng lọc.
4. Nếu threshold clinical thấp, nói rõ đây là ngưỡng nhạy để giảm bỏ sót nguy cơ.
5. Không chẩn đoán chắc chắn.
6. Nếu kết quả nguy cơ cao hoặc có triệu chứng cấp cứu, khuyến nghị đi khám/cấp cứu phù hợp.
```

### 9.6 RAG Branch Prompt

```text
Bạn đang trả lời câu hỏi kiến thức về đột quỵ dựa trên tài liệu nội bộ.

Quy tắc:
- Chỉ dùng các đoạn tài liệu được retrieval cung cấp.
- Nếu retrieval yếu hoặc không có tài liệu liên quan, nói rõ chưa đủ dữ liệu.
- Không bịa nguồn.
- Trả lời ngắn gọn, dễ hiểu.
- Luôn kèm mục "Nguồn tham khảo" với tên tài liệu và trang/section nếu có.
```

---

## 10. Memory Design

### 10.1 Redis Short Memory

Purpose:

- Maintain current conversation context.
- Store recent extracted profile fields.
- Avoid repeating answers from the last few turns.

Keys:

```text
chat:{session_id}:turns
chat:{session_id}:profile
chat:{session_id}:last_intent
chat:{session_id}:answer_hashes
chat:{session_id}:working_summary
```

TTL:

- Anonymous sessions: 30 minutes.
- Logged-in sessions: 24 hours.
- Admin/debug sessions: 2 hours.

Data strategy:

- Keep last 10-20 turns.
- Keep latest merged patient profile.
- Store normalized answer hashes for repetition detection.
- Summarize older messages after 20 turns.

### 10.2 PostgreSQL Long Memory

Purpose:

- Persist conversation history.
- Store durable user profile facts.
- Store prediction audit records.
- Store memory summaries for future sessions.

Tables:

- `chat_sessions`
- `chat_messages`
- `memory_summaries`
- `patient_profiles`
- `stroke_predictions`

Policy:

- Do not store raw secrets.
- Avoid excessive logging of sensitive free-text medical data.
- Store prediction inputs as structured JSON for audit.
- Add user-facing delete/export later if this becomes a real deployed healthcare app.

---

## 11. RAG Design

### 11.1 Recommended RAG Pattern

Use hybrid RAG:

- Dense vector search with `pgvector`.
- Keyword search with PostgreSQL full-text search.
- Metadata filters by document type, language, version, and admin visibility.
- Reranking with a multilingual reranker.
- Strict answer grounding with citations.
- Confidence fallback if retrieval quality is low.

Why this is the best fit:

- Stroke questions often contain exact medical keywords, abbreviations, and symptoms. Keyword search protects those terms.
- Users may phrase questions naturally in Vietnamese. Dense multilingual embeddings capture semantic similarity.
- Medical answers must be grounded and cite source chunks.
- PostgreSQL keeps relational data, metadata, and vectors in one operational database.

### 11.2 Embedding Choice

Recommended default:

- `BAAI/bge-m3`
- Dimension: `1024`
- Strength: multilingual retrieval, good for Vietnamese and English, can run locally.

Optional managed alternative:

- Use a hosted embedding API if local CPU/GPU latency is unacceptable.
- If changing embedding model, create a new document version and re-embed; do not mix dimensions in the same vector column.

### 11.3 Chunking Strategy

Supported file types:

- PDF
- DOCX
- TXT
- Markdown
- CSV for FAQ/structured rows

Chunking rules:

- Preserve headings and page numbers.
- Chunk size: 450-800 tokens.
- Overlap: 80-150 tokens.
- Store metadata: page, section heading, file name, upload version, language.
- Do not split tables blindly; convert tables to compact Markdown before chunking.

### 11.4 Retrieval Strategy

Retrieval steps:

```text
1. Normalize query.
2. Detect language.
3. Generate query embedding.
4. Run vector search top 20.
5. Run keyword search top 20.
6. Merge by reciprocal rank fusion.
7. Rerank merged candidates top 8.
8. Return top 3-5 chunks to DeepSeek.
9. If top score is below threshold, return "not enough evidence".
```

Initial thresholds:

- Vector similarity minimum: tune empirically, start with cosine distance `< 0.45`.
- Reranker minimum: start with `0.35`.
- Final context chunks: `3-5`.

### 11.5 Ingestion Pipeline

Admin upload flow:

```text
POST /api/admin/documents
  -> save file metadata to documents
  -> save raw file under UPLOAD_DIR
  -> create ingestion_jobs row
  -> background worker:
       parsing
       cleaning
       chunking
       embedding
       insert document_chunks rows with embedding
       update document status to indexed
```

Failure handling:

- If parsing fails, mark job `failed` with error message.
- If embedding fails, keep document but mark `needs_reingest`.
- Re-ingestion creates new chunks and marks previous chunks inactive.

---

## 12. PostgreSQL Schema And Init Script

Create `backend/scripts/create_database.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

POSTGRES_SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
APP_DB="${APP_DB:-stroke_ai}"
APP_USER="${APP_USER:-stroke}"
APP_PASSWORD="${APP_PASSWORD:-stroke_password}"

psql "postgresql://${POSTGRES_SUPERUSER}@${POSTGRES_HOST}:${POSTGRES_PORT}/postgres" <<SQL
DO
\$do\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${APP_USER}') THEN
    CREATE ROLE ${APP_USER} LOGIN PASSWORD '${APP_PASSWORD}';
  END IF;
END
\$do\$;

SELECT 'CREATE DATABASE ${APP_DB} OWNER ${APP_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${APP_DB}')\gexec
SQL

psql "postgresql://${POSTGRES_SUPERUSER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${APP_DB}" -f backend/scripts/init_db.sql
```

Create `backend/scripts/init_db.sql`.

This SQL creates extensions, tables, and indexes after `stroke_ai` exists. Docker Compose can also run it automatically through `/docker-entrypoint-initdb.d/`.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    name TEXT,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    title TEXT NOT NULL DEFAULT 'Stroke AI Chat',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    sender_type TEXT NOT NULL CHECK (sender_type IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    age DOUBLE PRECISION,
    gender INTEGER CHECK (gender IN (0, 1)),
    hypertension INTEGER CHECK (hypertension IN (0, 1)),
    heart_disease INTEGER CHECK (heart_disease IN (0, 1)),
    avg_glucose_level DOUBLE PRECISION,
    bmi DOUBLE PRECISION,
    avg_restingbp DOUBLE PRECISION,
    avg_maxhr DOUBLE PRECISION,
    avg_oldpeak DOUBLE PRECISION,
    heart_disease_rate DOUBLE PRECISION,
    cardio_cholesterol DOUBLE PRECISION,
    cardio_gluc DOUBLE PRECISION,
    cardio_rate DOUBLE PRECISION,
    smoke_flag INTEGER CHECK (smoke_flag IN (0, 1)),
    alco_flag INTEGER CHECK (alco_flag IN (0, 1)),
    ever_married TEXT,
    work_type TEXT,
    residence_type TEXT,
    smoking_status TEXT,
    estimated_calories DOUBLE PRECISION,
    source_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stroke_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    patient_profile_id UUID REFERENCES patient_profiles(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    risk_probability DOUBLE PRECISION NOT NULL CHECK (risk_probability >= 0 AND risk_probability <= 1),
    risk_label TEXT NOT NULL CHECK (risk_label IN ('low_risk', 'medium_risk', 'high_risk')),
    threshold DOUBLE PRECISION NOT NULL,
    input_json JSONB NOT NULL,
    filled_features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    explanation TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'admin_upload' CHECK (source_type IN ('admin_upload', 'seed_document', 'guideline', 'faq', 'research')),
    storage_path TEXT NOT NULL,
    language TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'indexing', 'indexed', 'failed', 'deleted')),
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'parsing', 'chunking', 'embedding', 'indexed', 'failed')),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(section_title, '') || ' ' || chunk_text)
    ) STORED,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS memory_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    summary_text TEXT NOT NULL,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    input_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_patient_profiles_session_id ON patient_profiles(session_id);
CREATE INDEX IF NOT EXISTS idx_stroke_predictions_session_created ON stroke_predictions(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_active ON document_chunks(document_id, is_active);
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector ON document_chunks USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_tool_audit_logs_session_created ON tool_audit_logs(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WHERE is_active = true;
```

Docker command example:

```bash
docker exec -i stroke-postgres psql -U stroke -d stroke_ai < backend/scripts/init_db.sql
```

---

## 13. Backend Implementation Phases

### Phase 1 - Backend Skeleton

Deliverables:

- `backend/app/main.py`
- config loader
- health endpoints
- PostgreSQL connection
- Redis connection
- model artifact readiness checks

Acceptance:

- `GET /health` returns `ok`.
- `GET /ready` verifies DB, Redis, artifact files, and `pgvector`.

### Phase 2 - Prediction Service

Deliverables:

- `prediction/model_loader.py`: load clinical, edge, calories models once at startup.
- `prediction/preprocess.py`: reproduce `Code/Model.py` preprocessing.
- `prediction/service.py`: expose `predict(profile, mode="clinical")`.
- `routes_prediction.py`: direct prediction endpoint.

Tests:

- Gender normalization.
- BMI outlier handling.
- Missing numeric/categorical fill from metadata.
- Final feature order matches metadata.
- Clinical prediction returns probability in `[0, 1]`.
- Edge prediction uses only 6 columns.

### Phase 3 - LangGraph Chat

Deliverables:

- graph state schema
- tools
- routing logic
- DeepSeek client
- WebSocket event streaming

Tests:

- Emergency phrase routes to emergency response without normal prediction.
- Missing fields cause follow-up question.
- Sufficient fields call prediction tool.
- Medical Q&A calls RAG before response synthesis.
- Duplicate answer is shortened by memory dedup.

### Phase 4 - RAG

Deliverables:

- document parser
- chunker
- embedding client
- ingestion worker
- hybrid retrieval
- reranker
- source citation output

Tests:

- Existing `Documents/Final_paper_IEEE.pdf` can be ingested.
- Chunks include page metadata.
- Embeddings are inserted into PostgreSQL.
- Vector search and keyword search both return results.
- Low-confidence retrieval returns fallback instead of hallucinated answer.

### Phase 5 - Frontend Chat

Deliverables:

- Vite React app
- Messenger-style chat page
- WebSocket client
- streaming rendering
- prediction card
- RAG citation badges
- responsive mobile layout

Tests:

- User can send message.
- Assistant deltas append to same bubble.
- Typing indicator appears/disappears correctly.
- Prediction event renders `RiskSummaryCard`.
- Source event renders badges.
- Socket reconnect state is visible.

### Phase 6 - Admin Upload

Deliverables:

- admin documents page
- upload endpoint integration
- ingestion job status polling
- document list

Tests:

- Admin uploads PDF.
- UI shows queued/indexing/indexed.
- Uploaded document becomes searchable through `/api/rag/search`.

### Phase 7 - Hardening

Deliverables:

- Docker Compose for frontend/backend/postgres/redis.
- structured logging
- request id
- CORS config
- basic auth/admin guard
- rate limiting for chat endpoints
- model monitoring fields in prediction logs

Acceptance:

- Full stack starts with one command.
- Seed document is indexed.
- Chat can answer a RAG question.
- Chat can ask follow-up for missing fields.
- Chat can produce prediction when enough fields are available.

---

## 14. Frontend Implementation Phases

### Phase 1 - Static Chat UI

Build the chat shell first without backend dependency.

Acceptance:

- Empty state suggests example questions.
- Message bubbles render correctly.
- Composer supports Enter to send and Shift+Enter newline.
- Mobile and desktop both usable.

### Phase 2 - WebSocket Integration

Wire `chatSocket.ts` to backend.

Acceptance:

- User message renders optimistically.
- Server `assistant_delta` streams into one assistant message.
- Server `error` event renders recoverable error bubble.
- Reconnect state is visible.

### Phase 3 - Prediction UI

Render structured events.

Acceptance:

- `profile_detected` updates a detected indicators panel.
- `prediction_result` renders `RiskSummaryCard`.
- Risk card includes disclaimer.

### Phase 4 - RAG Sources UI

Render citations.

Acceptance:

- `rag_sources` event attaches source badges to message.
- Source badge displays title and page/section.

### Phase 5 - Admin Documents

Add admin-only route.

Acceptance:

- Upload documents.
- Track ingestion status.
- List indexed documents and chunk counts.

---

## 15. C++ Edge Model Integration

Do not use the C++ model as the primary web backend model.

Recommended use:

- Keep `stroke_predictor_model.h` and `Code/test_model.cpp` for benchmark and fallback.
- Add `backend/scripts/compile_edge_model.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
g++ -O3 Code/test_model.cpp -o Code/test_model
./Code/test_model
```

Optional future integration:

- Build a small C++ CLI that accepts JSON input and returns JSON output.
- Backend calls it only when Python clinical model is unavailable or when benchmarking edge latency.
- Keep the output contract identical to Python prediction service.

Reason:

- The C++ artifact is based on the edge feature subset and is not the best clinical model.
- Calling Python `joblib` model in-process is simpler and more faithful for the web product.

---

## 16. Docker Compose

Add `docker-compose.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: stroke-postgres
    environment:
      POSTGRES_DB: stroke_ai
      POSTGRES_USER: stroke
      POSTGRES_PASSWORD: stroke_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/scripts/init_db.sql:/docker-entrypoint-initdb.d/001_init_db.sql:ro

  redis:
    image: redis:7-alpine
    container_name: stroke-redis
    ports:
      - "6379:6379"

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: stroke-backend
    env_file:
      - backend/.env
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./Documents:/app/Documents:ro
      - ./backend/uploads:/app/backend/uploads

  frontend:
    build:
      context: frontend
    container_name: stroke-frontend
    depends_on:
      - backend
    ports:
      - "5173:5173"

volumes:
  postgres_data:
```

---

## 17. Safety And Compliance Requirements

Minimum safety requirements:

- Always show disclaimer for prediction results.
- Emergency red flags bypass normal RAG/prediction flow.
- Do not claim diagnosis.
- Do not claim treatment instructions.
- Log structured prediction inputs but avoid unnecessary raw free-text logging.
- Admin uploads should be authenticated.
- File uploads must validate MIME type and size.
- RAG answers must cite sources or say there is not enough evidence.

Emergency response template:

```text
Các dấu hiệu bạn mô tả có thể là tình huống cấp cứu liên quan đến đột quỵ. Hãy gọi cấp cứu ngay hoặc đến cơ sở y tế gần nhất. Không tự lái xe nếu có thể nhờ người khác hỗ trợ.

Tôi có thể giúp bạn ghi lại các triệu chứng và thời điểm bắt đầu để cung cấp cho nhân viên y tế.
```

---

## 18. Testing Strategy

Backend unit tests:

- `tests/prediction/test_preprocess.py`
- `tests/prediction/test_service.py`
- `tests/rag/test_chunking.py`
- `tests/rag/test_retrieval.py`
- `tests/graph/test_routing.py`
- `tests/api/test_prediction_routes.py`
- `tests/api/test_chat_ws.py`

Frontend tests:

- `ChatComposer.test.tsx`
- `MessageList.test.tsx`
- `RiskSummaryCard.test.tsx`
- `chatSocket.test.ts`

End-to-end smoke tests:

- Start full stack.
- Ingest `Documents/Final_paper_IEEE.pdf`.
- Ask: "Dấu hiệu đột quỵ thường gặp là gì?"
- Ask: "Tôi 67 tuổi, nam, cao huyết áp, bệnh tim, glucose 228, BMI 36.6 thì nguy cơ sao?"
- Verify response includes risk card and disclaimer.

Verification commands:

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q

cd ../frontend
npm install
npm run test
npm run build
```

Model verification command after backend deps are installed:

```bash
cd /home/nhatbang/Samsung_NIC/Healthcare-Stroke-prediction
backend/.venv/bin/python Code/evaluate_model.py
```

---

## 19. Acceptance Criteria

Frontend:

- One-page Messenger-style chatbot.
- Responsive desktop/mobile.
- Streaming assistant responses.
- Three-dot typing indicator.
- Prediction card for model output.
- RAG citation badges for grounded answers.
- Clear error and reconnect states.
- Admin document upload page exists.

Backend:

- FastAPI starts successfully.
- `/ready` checks PostgreSQL, Redis, models, and `pgvector`.
- Direct prediction endpoint calls `stroke_model_clinical.joblib`.
- Prediction preprocessing matches `Code/Model.py`.
- LangGraph routes emergency, prediction, and medical Q&A separately.
- DeepSeek client uses env-configured model names.
- Redis stores short memory.
- PostgreSQL stores sessions, messages, predictions, documents, chunks, embeddings, and memory summaries.
- Admin upload triggers RAG ingestion.

RAG:

- `CREATE EXTENSION IF NOT EXISTS vector` runs successfully.
- Documents are chunked with metadata.
- Embeddings are stored in `document_chunks.embedding`.
- HNSW cosine index exists.
- Retrieval uses hybrid vector + keyword search.
- Answers cite sources or state insufficient evidence.

Medical safety:

- Acute stroke symptoms produce emergency guidance.
- Prediction output never claims diagnosis.
- Missing user indicators trigger focused follow-up questions.

---

## 20. Recommended Build Order

1. Create `backend/` skeleton, config, Dockerfile, health checks.
2. Create PostgreSQL/Redis Docker Compose and `backend/scripts/init_db.sql`.
3. Implement prediction service with clinical model.
4. Add direct prediction endpoint and tests.
5. Implement RAG ingestion for `Documents/Final_paper_IEEE.pdf`.
6. Implement RAG retrieval endpoint and tests.
7. Implement DeepSeek client and prompt templates.
8. Implement LangGraph nodes/tools and REST chat fallback.
9. Add WebSocket streaming chat endpoint.
10. Create `frontend/` Vite React chat UI.
11. Wire frontend WebSocket event handling.
12. Add prediction card and RAG source badges.
13. Add admin document upload UI.
14. Add full-stack smoke tests.
15. Harden auth, logging, upload validation, and rate limiting.

---

## 21. Implementation Notes For The Next Coding Step

Do not start with the frontend. Start with the backend prediction service because all chatbot behavior depends on a reliable model contract.

First concrete implementation task:

```text
Create backend skeleton + prediction service:
- backend/requirements.txt
- backend/app/main.py
- backend/app/core/config.py
- backend/app/prediction/feature_schema.py
- backend/app/prediction/model_loader.py
- backend/app/prediction/preprocess.py
- backend/app/prediction/service.py
- backend/app/api/routes_prediction.py
- backend/tests/prediction/test_preprocess.py
- backend/tests/api/test_prediction_routes.py
```

The first test should verify that a high-risk sample based on `Code/test_model.cpp` returns a valid probability and uses the clinical model by default.

---

## 22. External References Checked

- LangGraph memory docs: https://docs.langchain.com/oss/python/langgraph/add-memory
- LangGraph persistence docs: https://docs.langchain.com/oss/python/langgraph/persistence
- pgvector HNSW index docs: https://github.com/pgvector/pgvector
- DeepSeek quickstart and current model names: https://api-docs.deepseek.com/
- DeepSeek chat completion API: https://api-docs.deepseek.com/api/create-chat-completion/
- DeepSeek list models API: https://api-docs.deepseek.com/api/list-models/
