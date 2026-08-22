from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.config import settings
from app.auth import admin_router, get_optional_auth_user, require_admin_access, router as auth_router
from app.history import router as history_router
from app.llm import deepseek_client
from app.chat.graph import ScreeningWorkflow
from app.chat.web_search import search_images_for_message, search_risk_factor_sources, search_sources_for_message
from app.extraction.deepseek_tool import DeepSeekProfileExtractor
from app.memory import session_store
from app.persistence.conversation_store import PostgresConversationStore
from app.persistence.database import database_available, get_checkpoint_pool
from app.conversation_runtime import conversation_store
from app.progress import build_progress_summary, router as progress_router
from app.prediction.preprocess import REQUIRED_SCREENING_FIELDS
from app.prediction.service import prediction_service
from app.schemas import AssistantCard, ChatRequest, PredictionRequest, RagSearchRequest, RenameSessionRequest, SessionListResponse
from app.security import RateLimitRule, client_ip, rate_limiter


app = FastAPI(title=settings.app_name)

SCREENING_GUARD_MESSAGE = (
    "Giới hạn hợp lệ: tuổi 0-100, giới tính nam/nữ, huyết áp dạng tâm thu/tâm trương, "
    "glucose 40-400, BMI 10-80. "
    "Bạn hãy gửi lại các chỉ số theo đúng khoảng này."
)

_BASE_CONFIDENCE_PERCENT = 92.0
_MIN_CONFIDENCE_PERCENT = 48.0
_memory_profiles: dict[str, dict[str, Any]] = defaultdict(dict)
profile_extractor = DeepSeekProfileExtractor(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    model=settings.deepseek_model,
    timeout=settings.deepseek_timeout_seconds,
)


def _general_response(message: str, history: list[dict[str, Any]]) -> str:
    if deepseek_client.configured():
        try:
            messages = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in history[-12:]
                if item.get("role") in {"user", "assistant"}
            ]
            messages.append({"role": "user", "content": message})
            return deepseek_client.complete(messages)
        except Exception:
            pass
    lowered = message.casefold()
    return (
        _build_stroke_education_reply(lowered)
        or _build_symptom_reply(lowered)
        or (
            "Tôi là trợ lý AI về tầm soát nguy cơ đột quỵ. Bạn có thể hỏi sâu về triệu chứng, "
            "yếu tố nguy cơ, hoặc gửi các chỉ số sức khỏe để hệ thống hỗ trợ sàng lọc."
        )
    )


def _prediction_synthesis(card: dict[str, Any], profile: dict[str, Any]) -> str:
    """Explain a deterministic model result with optional user context.

    The LLM may write the interpretation, but it is never allowed to change
    the probability, label, or feature values produced by the predictor.
    """
    evidence_sources = search_risk_factor_sources(profile)
    context = {
        "model_result": {
            "risk_label": card.get("risk_label"),
            "risk_score_10": card.get("risk_score_10"),
            "estimated_calories": next(
                (
                    item.get("value")
                    for item in card.get("feature_cards", [])
                    if item.get("label") == "Lượng calo ước tính"
                ),
                None,
            ),
        },
        "accepted_profile": profile,
        "evidence_sources": [
            {"title": item.get("title"), "url": item.get("url"), "summary": item.get("content")}
            for item in evidence_sources
        ],
        "instruction": (
            "Viết Nhận định tổng hợp bằng tiếng Việt, tối đa 3 đoạn ngắn hoặc 4 gạch đầu dòng. "
            "Không viết tiêu đề 'Nhận định tổng hợp' vì giao diện đã hiển thị tiêu đề. "
            "Chỉ sử dụng điểm trên thang 10, không hiển thị hoặc quy đổi sang phần trăm. "
            "Giữ nguyên điểm số/nhãn model và không chẩn đoán. "
            "Với bệnh tim, hút thuốc hoặc rượu được người dùng cung cấp, hãy dùng các nguồn "
            "evidence_sources để tóm tắt mức liên quan (mạnh/yếu hoặc chưa đủ chắc chắn), "
            "không tự bịa loại bệnh tim hay liều lượng. Nếu có nguồn, gắn 1-3 liên kết Markdown "
            "ngay sau ý tương ứng. Không dùng câu mẫu cứng nhắc kiểu 'không được đưa vào model'; "
            "hãy diễn giải tự nhiên rằng đây là bối cảnh nguy cơ bổ sung. "
            "Nêu Estimated_calories là giá trị ước tính từ tuổi, giới tính và BMI."
        ),
    }
    if deepseek_client.configured():
        try:
            interpretation = deepseek_client.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Bạn là agent giải thích kết quả sàng lọc đột quỵ. "
                            "Chỉ diễn giải dữ liệu JSON được cung cấp; không được thay đổi "
                            "điểm model, không khẳng định chẩn đoán, và phải khuyến nghị cấp cứu "
                            "nếu người dùng có dấu hiệu nguy hiểm."
                        ),
                    },
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ]
            ).strip()
            # Avoid a duplicated card heading if the provider ignores the
            # instruction and emits it anyway.
            return interpretation.removeprefix("Nhận định tổng hợp:").removeprefix("Nhận định tổng hợp").strip()
        except Exception:
            pass
    extras = []
    if profile.get("smoke_flag"):
        extras.append("hút thuốc là yếu tố nguy cơ bổ sung cần lưu ý")
    if profile.get("alco_flag"):
        extras.append("uống rượu/bia có thể làm tăng nguy cơ khi sử dụng nhiều")
    if profile.get("heart_disease"):
        extras.append("tiền sử bệnh tim cần được bác sĩ đánh giá cụ thể")
    extra_text = f" Ngoài điểm model, {', '.join(extras)}." if extras else ""
    source_text = ""
    if evidence_sources:
        source_text = "\n\nNguồn tham khảo: " + "; ".join(
            f"[{item['title']}]({item['url']})" for item in evidence_sources
        )
    return (
        f"Kết quả model hiện tại là {card.get('risk_label')} "
        f"({float(card.get('risk_score_10', 0.0)):.2f}/10).{extra_text} "
        "Estimated_calories là lượng calo ước tính từ tuổi, giới tính và BMI; "
        "kết quả chỉ dùng để sàng lọc, không thay thế bác sĩ." + source_text
    )


def _validate_reply_context(
    reply_context: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if reply_context is None:
        return None
    message_id = str(reply_context["message_id"])
    quote = " ".join(str(reply_context["quote"]).split())
    source = next(
        (
            item
            for item in history
            if str(item.get("id") or "") == message_id and item.get("role") == "assistant"
        ),
        None,
    )
    candidates = [source] if source else []
    # Cards rendered after a reload can have a client-side wrapper id that is
    # different from the persisted message id. The quote is still safe to
    # accept when its exact text is found in another assistant message in the
    # same owned session.
    candidates.extend(
        item for item in history
        if item.get("role") == "assistant" and item is not source
    )
    def canonical(value: str) -> str:
        return re.sub(r"[^\w\dÀ-ỹ]+", " ", value.casefold()).strip()

    canonical_quote = canonical(quote)
    matched = False
    for candidate in candidates:
        source_texts = [" ".join(str(candidate.get("content") or "").split())]
        assistant_card = candidate.get("assistant_card")
        if isinstance(assistant_card, dict):
            source_texts.extend(
                " ".join(str(assistant_card.get(field) or "").split())
                for field in ("summary", "interpretation")
            )
        if quote and any(
            quote in text or (canonical_quote and canonical_quote in canonical(text))
            for text in source_texts
        ):
            matched = True
            break
    if not matched:
        raise HTTPException(
            status_code=422,
            detail="Đoạn trích không thuộc câu trả lời StrokeGuard AI trong phiên hiện tại.",
        )
    return reply_context


def _build_screening_workflow() -> ScreeningWorkflow:
    checkpointer = None
    if database_available() and settings.langgraph_checkpoint_enabled:
        checkpointer = PostgresSaver(get_checkpoint_pool())
        checkpointer.setup()
    return ScreeningWorkflow(
        extractor=profile_extractor,
        prediction_service=prediction_service,
        general_responder=_general_response,
        media_searcher=search_images_for_message,
        web_searcher=search_sources_for_message,
        synthesis_responder=_prediction_synthesis,
        checkpointer=checkpointer,
    )


screening_workflow = _build_screening_workflow()


def _http_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=422, detail="session_id must be a UUID")
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=404, detail="Session not found")
    return HTTPException(status_code=503, detail="Conversation database is unavailable")


def _confidence_percent_from_missing_fields(missing_fields: list[str]) -> float:
    penalty = 8.0 * len(missing_fields)
    return max(_MIN_CONFIDENCE_PERCENT, _BASE_CONFIDENCE_PERCENT - penalty)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(history_router)
app.include_router(progress_router)


@app.middleware("http")
async def security_guard(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_request_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})

        body = await request.body()
        if len(body) > settings.max_request_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )

    path = request.url.path
    if path.startswith("/api") or path in {"/health", "/ready"}:
        ip = client_ip(request)
        rule = (
            RateLimitRule(settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds)
            if path.startswith("/api/auth")
            else RateLimitRule(settings.general_rate_limit_requests, settings.general_rate_limit_window_seconds)
        )
        allowed, remaining, retry_after = rate_limiter.allow(f"{ip}:{'auth' if path.startswith('/api/auth') else 'general'}", rule)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    status = prediction_service.ready_status()
    return {
        "status": "ready" if status["model_status"] == "loaded" else "degraded",
        **status,
        "deepseek_configured": deepseek_client.configured(),
        "deepseek_model": deepseek_client.model,
        "static_dir": str(settings.static_dir),
    }


@app.get("/api/session/me")
def session_me(request: Request) -> dict[str, Any]:
    user = get_optional_auth_user(request)
    return {
        "authenticated": bool(user),
        "user": user.model_dump() if user else None,
    }


@app.post("/api/chat/message")
def chat_message(payload: ChatRequest, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    session_id = payload.session_id or store.new_session_id()
    try:
        if isinstance(store, PostgresConversationStore):
            if payload.reply_context is None:
                store.ensure_session(session_id, payload.message)
            current_profile = store.profile(session_id)
        else:
            current_profile = dict(_memory_profiles.get(session_id, {}))
        full_history = store.messages(session_id)
        reply_context = _validate_reply_context(
            payload.reply_context.model_dump() if payload.reply_context else None,
            full_history,
        )
        history = full_history[-20:]
        result = screening_workflow.invoke(
            message=payload.message,
            reply_context=reply_context,
            current_profile=current_profile,
            recent_messages=history,
            thread_id=session_id,
        )
        if isinstance(store, PostgresConversationStore):
            messages = store.save_turn(
                session_id,
                payload.message,
                result,
                reply_context=reply_context,
                media=result.get("media") or [],
            )
        else:
            _memory_profiles[session_id] = dict(result.get("current_profile", current_profile))
            messages = store.add_turn(
                session_id,
                payload.message,
                result["reply"],
                assistant_card=result.get("assistant_card"),
                reply_context=reply_context,
                media=result.get("media") or [],
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise _http_store_error(exc) from exc
    return {
        "session_id": session_id,
        "reply": result["reply"],
        "messages": messages,
        "assistant_card": result.get("assistant_card"),
        "media": result.get("media") or [],
        "sources": result.get("sources") or [],
        "missing_required_fields": result.get("missing_required_fields", []),
    }


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        return {"session_id": session_id, "messages": store.messages(session_id)}
    except Exception as exc:
        raise _http_store_error(exc) from exc


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        messages = store.messages(session_id)
    except Exception as exc:
        raise _http_store_error(exc) from exc
    recent_messages = messages[-20:]
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "recent_messages": recent_messages,
        "has_activity": bool(messages),
    }


@app.get("/api/sessions/{session_id}/overview")
def get_session_overview(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        messages = store.messages(session_id)
    except Exception as exc:
        raise _http_store_error(exc) from exc
    return {
        "session_id": session_id,
        "progress": build_progress_summary(
            session_id,
            store.progress_messages(session_id) if hasattr(store, "progress_messages") else messages,
        ).model_dump(),
        "history": {
            "session_id": session_id,
            "message_count": len(messages),
            "recent_messages": messages[-20:],
            "has_activity": bool(messages),
        },
    }


@app.get("/api/sessions", response_model=SessionListResponse)
def list_sessions(request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        return {"sessions": store.sessions()}
    except Exception as exc:
        raise _http_store_error(exc) from exc


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        store.reset(session_id)
        _memory_profiles.pop(session_id, None)
    except Exception as exc:
        raise _http_store_error(exc) from exc
    return {"session_id": session_id, "reset": True}


@app.patch("/api/sessions/{session_id}/title")
def rename_session(session_id: str, payload: RenameSessionRequest, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        store.rename(session_id, payload.title)
    except Exception as exc:
        raise _http_store_error(exc) from exc
    return {"session_id": session_id, "title": payload.title}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request, response: Response) -> dict[str, Any]:
    store = conversation_store(request, response)
    try:
        store.reset(session_id)
        _memory_profiles.pop(session_id, None)
    except Exception as exc:
        raise _http_store_error(exc) from exc
    return {"session_id": session_id, "deleted": True}


@app.post("/api/predict/stroke")
def predict_stroke(payload: PredictionRequest) -> dict[str, Any]:
    profile = payload.model_dump(exclude_none=True)
    missing = [field for field in REQUIRED_SCREENING_FIELDS if field not in profile]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Cần cung cấp đủ các chỉ số sàng lọc trước khi dự đoán.",
                "missing_required_fields": missing,
            },
        )
    return prediction_service.predict(profile)


@app.get("/api/predict/schema")
def prediction_schema() -> dict[str, Any]:
    return {
        "required_screening_fields": REQUIRED_SCREENING_FIELDS,
        "clinical_model": "stroke_model_clinical.joblib",
        "note": "API chỉ chạy model sau khi có đủ toàn bộ field sàng lọc bắt buộc.",
    }


@app.post("/api/rag/search")
def rag_search(payload: RagSearchRequest) -> dict[str, Any]:
    query = payload.query
    return {
        "query": query,
        "sources": [
            {
                "title": "Final_paper_IEEE.pdf",
                "page": 3,
                "score": 0.5,
                "note": "RAG vector ingestion sẽ được nối ở phase PostgreSQL + pgvector.",
            }
        ]
        if query
        else [],
    }


@app.get("/api/admin/documents")
def list_documents(_: Any = Depends(require_admin_access)) -> dict[str, Any]:
    return {
        "documents": [
            {
                "id": "seed-final-paper",
                "title": "Final_paper_IEEE.pdf",
                "source_type": "seed_document",
                "status": "available",
                "storage_path": str(settings.artifact_dir.parent / "Documents" / "Final_paper_IEEE.pdf"),
            }
        ]
    }


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(settings.static_dir / "admin.html")


@app.get("/settings", include_in_schema=False)
def settings_page() -> FileResponse:
    return FileResponse(settings.static_dir / "settings.html")


@app.get("/history", include_in_schema=False)
def history_page() -> FileResponse:
    return FileResponse(settings.static_dir / "history.html")


@app.get("/progress", include_in_schema=False)
def progress_page() -> FileResponse:
    return FileResponse(settings.static_dir / "progress.html")


def _build_chat_reply(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, AssistantCard | None]:
    lowered = message.lower()
    emergency_terms = ["méo miệng", "yếu tay", "nói khó", "tê một bên", "đau đầu dữ dội", "face droop", "speech"]
    if any(term in lowered for term in emergency_terms):
        reply = (
            "Các dấu hiệu bạn mô tả có thể là tình huống cấp cứu liên quan đến đột quỵ. "
            "Hãy gọi cấp cứu ngay hoặc đến cơ sở y tế gần nhất. Tôi có thể giúp bạn ghi lại "
            "thời điểm bắt đầu triệu chứng để cung cấp cho nhân viên y tế."
        )
        return reply, _build_assistant_card(
            {
                "model_used": "stroke_model_clinical.joblib",
                "model_status": "loaded",
                "model_version": "clinical-1",
                "risk_probability": 1.0,
                "risk_label": "high_risk",
                "missing_user_fields": [],
                "detected_features": {},
            },
            reply,
            alert_level="critical",
            alert_message="Dấu hiệu cảnh báo khẩn. Không trì hoãn xử trí.",
        )

    if deepseek_client.configured():
        try:
            messages = [
                {"role": item["role"], "content": item["content"]}
                for item in (history or [])
                if item["role"] in {"user", "assistant"}
            ]
            messages.append({"role": "user", "content": message})
            return deepseek_client.complete(messages), None
        except Exception:
            pass

    stroke_education_reply = _build_stroke_education_reply(lowered)
    if stroke_education_reply is not None:
        return stroke_education_reply, None

    symptom_reply = _build_symptom_reply(lowered)
    if symptom_reply is not None:
        return symptom_reply, None

    screening_profile, screening_issues = _extract_screening_profile(message)
    if screening_issues:
        return _format_screening_guard_reply(screening_issues), None
    if screening_profile:
        prediction = prediction_service.predict(screening_profile)
        reply = _format_prediction_reply(prediction)
        return reply, _build_assistant_card(prediction, reply)

    if any(term in lowered for term in ["tuổi", "bmi", "glucose", "huyết áp", "tăng huyết áp"]):
        return (
            "Tôi đã ghi nhận thông tin sàng lọc. Để dự đoán bằng model, hãy cung cấp tối thiểu: "
            "tuổi, giới tính, tăng huyết áp, bệnh tim, glucose trung bình và BMI. "
            f"{SCREENING_GUARD_MESSAGE}",
            None,
        )

    return (
        "Tôi là trợ lý AI về tầm soát nguy cơ đột quỵ. Bạn có thể hỏi về triệu chứng, yếu tố nguy cơ, "
        "hoặc gửi các chỉ số sức khỏe để hệ thống hỗ trợ sàng lọc.",
        None,
    )


def _build_stroke_education_reply(lowered: str) -> str | None:
    education_terms = [
        "đột quỵ",
        "stroke",
        "nguy cơ đột quỵ",
        "dấu hiệu đột quỵ",
        "triệu chứng đột quỵ",
        "yếu tố nguy cơ",
        "phòng ngừa",
        "nguyên nhân",
        "fast",
    ]
    if not any(term in lowered for term in education_terms):
        return None

    return (
        "Dấu hiệu thường gặp của đột quỵ theo kiểu FAST gồm méo miệng, yếu tay/chân một bên và nói khó. "
        "Ngoài ra còn có nhìn mờ, chóng mặt, đau đầu dữ dội hoặc tê yếu đột ngột. "
        "Nếu những dấu hiệu này xuất hiện đột ngột, hãy gọi cấp cứu ngay. "
        "Nếu bạn muốn, tôi có thể giúp bạn phân biệt dấu hiệu khẩn với triệu chứng nhẹ hơn."
    )


def _build_symptom_reply(lowered: str) -> str | None:
    symptom_terms = [
        "đau đầu",
        "đau nửa đầu",
        "nhức đầu",
        "headache",
        "chóng mặt",
        "buồn nôn",
        "mệt",
        "tê",
        "yếu",
        "mờ mắt",
        "nói khó",
        "méo miệng",
        "đột quỵ",
        "triệu chứng",
    ]
    if not any(term in lowered for term in symptom_terms):
        return None

    if len(lowered.split()) <= 4 and not any(term in lowered for term in ["đau đầu", "chóng mặt", "buồn nôn", "tê", "yếu", "méo miệng", "nói khó", "mờ mắt"]):
        return None

    return (
        "Triệu chứng này có thể liên quan đến nhiều nguyên nhân, không nhất thiết là đột quỵ. "
        "Nếu cơn đau đầu xuất hiện đột ngột, rất dữ dội, kèm tê/yếu một bên, nói khó, mờ mắt, "
        "buồn nôn nhiều hoặc lú lẫn, hãy đi cấp cứu ngay. Nếu không, bạn nên theo dõi thời điểm bắt đầu, "
        "mức độ đau, vị trí đau và các triệu chứng đi kèm để tôi hỗ trợ sàng lọc kỹ hơn."
    )


def _extract_screening_profile(message: str) -> tuple[dict[str, Any], list[str]]:
    raise RuntimeError(
        "Regex screening extraction was removed; invoke screening_workflow with DeepSeekProfileExtractor"
    )


def _format_screening_guard_reply(issues: list[str]) -> str:
    unique_issues: list[str] = []
    for issue in issues:
        if issue not in unique_issues:
            unique_issues.append(issue)
    issue_text = " ".join(unique_issues)
    return f"Thông tin không hợp lệ: {issue_text} {SCREENING_GUARD_MESSAGE}"


def _format_prediction_reply(prediction: dict[str, Any]) -> str:
    label_map = {
        "high_risk": "Cao",
        "medium_risk": "Trung bình",
        "low_risk": "Thấp",
    }
    label = label_map.get(str(prediction.get("risk_label")), str(prediction.get("risk_label", "Không rõ")))
    probability_score = min(10.0, max(0.0, float(prediction.get("risk_probability") or 0.0) * 100.0))
    missing_fields = prediction.get("missing_user_fields") or []
    confidence_percent = _confidence_percent_from_missing_fields(missing_fields)
    confidence_text = f" Độ tin cậy của dự đoán hiện tại: {confidence_percent:.1f}%."
    missing_text = ""
    if missing_fields:
        friendly = {
            "age": "tuổi",
            "gender": "giới tính",
            "hypertension": "tăng huyết áp",
            "heart_disease": "bệnh tim",
            "avg_glucose_level": "glucose trung bình",
            "bmi": "BMI",
        }
        missing_names = ", ".join(friendly.get(field, field) for field in missing_fields)
        missing_text = f" Hiện còn thiếu: {missing_names}, nên kết quả có thể sai lệch."

    model_name = str(prediction.get("model_used") or "stroke_model_clinical.joblib")
    return (
        f"Tôi đã chạy model sàng lọc bằng {model_name}. Kết quả tạm thời: nguy cơ {label} ({probability_score:.1f}/10)."
        f"{missing_text}{confidence_text} Kết quả này chỉ dùng để sàng lọc, không thay thế bác sĩ."
    )


def _build_assistant_card(
    prediction: dict[str, Any],
    reply: str,
    *,
    alert_level: str = "neutral",
    alert_message: str | None = None,
) -> AssistantCard:
    label_map = {
        "high_risk": "Cao",
        "medium_risk": "Trung bình",
        "low_risk": "Thấp",
    }
    label = label_map.get(str(prediction.get("risk_label")), str(prediction.get("risk_label", "Không rõ")))
    risk_probability_percent = float(prediction.get("risk_probability") or 0.0) * 100.0
    risk_score_10 = min(10.0, max(0.0, risk_probability_percent))
    missing_fields_raw = prediction.get("missing_user_fields") or []
    friendly = {
        "age": "tuổi",
        "gender": "giới tính",
        "hypertension": "tăng huyết áp",
        "heart_disease": "bệnh tim",
        "avg_glucose_level": "glucose trung bình",
        "bmi": "BMI",
    }
    missing_fields = [friendly.get(field, str(field)) for field in missing_fields_raw]
    confidence_percent = _confidence_percent_from_missing_fields(missing_fields)
    model_name = str(prediction.get("model_used") or "stroke_model_clinical.joblib")
    model_status = str(prediction.get("model_status") or "loaded")
    model_version = str(prediction.get("model_version") or "clinical-1")
    detected = prediction.get("detected_features") or {}
    highlights = ["Mô hình lâm sàng", f"Nguy cơ {label}", f"{risk_score_10:.1f}/10"]
    if confidence_percent is not None:
        highlights.append(f"Tin cậy {confidence_percent:.1f}%")
    if "age" in detected:
        highlights.append(f"Tuổi {detected['age']}")
    if "bmi" in detected:
        highlights.append(f"BMI {detected['bmi']}")
    if "avg_glucose_level" in detected:
        highlights.append(f"Glucose {detected['avg_glucose_level']}")

    return AssistantCard(
        model_used=model_name,
        model_status=model_status,
        model_version=model_version,
        verified=True,
        alert_level=alert_level,
        risk_probability_percent=risk_probability_percent,
        risk_score_10=risk_score_10,
        risk_label=label,
        confidence_percent=confidence_percent,
        missing_fields=missing_fields,
        highlights=highlights,
        feature_cards=_build_feature_cards(detected),
        alert_message=alert_message,
        disclaimer="Kết quả này chỉ dùng để sàng lọc, không thay thế bác sĩ.",
        summary=reply,
    )


def _build_feature_cards(detected: dict[str, Any]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []

    def add_card(label: str, value: str, detail: str) -> None:
        cards.append({"label": label, "value": value, "detail": detail})

    if "age" in detected:
        age_value = detected["age"]
        add_card("Tuổi", f"{int(age_value)}", "Đã nhận từ câu chat")
    if "gender" in detected:
        gender_value = "Nam" if float(detected["gender"]) <= 0 else "Nữ"
        add_card("Giới tính", gender_value, "Đã nhận từ câu chat")
    if "hypertension" in detected:
        hypertension_value = "Cao" if float(detected["hypertension"]) >= 1 else "Thấp"
        add_card("Huyết áp", hypertension_value, "Đã nhận từ câu chat")
    if "heart_disease" in detected:
        heart_value = "Có" if float(detected["heart_disease"]) >= 1 else "Không có"
        add_card("Bệnh tim", heart_value, "Đã nhận từ câu chat")
    if "avg_glucose_level" in detected:
        add_card("Glucose", f"{float(detected['avg_glucose_level']):.0f}", "mg/dL")
    if "bmi" in detected:
        add_card("BMI", f"{float(detected['bmi']):.1f}".rstrip("0").rstrip("."), "Chỉ số khối cơ thể")

    return cards


if settings.static_dir.exists():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
