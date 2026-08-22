from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from psycopg_pool import ConnectionPool


DATABASE_URL = os.environ.get("STROKE_TEST_DATABASE_URL")
SESSION_ID = UUID("10000000-0000-0000-0000-000000000001")
AUTH_EMAIL = "strokeguard-repository-test@example.invalid"


@pytest.fixture()
def pool():
    if not DATABASE_URL:
        pytest.skip("STROKE_TEST_DATABASE_URL is not configured")
    test_pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=2, open=True)
    try:
        yield test_pool
    finally:
        with test_pool.connection() as connection:
            connection.execute(
                "DELETE FROM strokeguard.chat_sessions WHERE id = %s",
                (SESSION_ID,),
            )
            connection.execute(
                "DELETE FROM strokeguard.users WHERE email = %s",
                (AUTH_EMAIL,),
            )
        test_pool.close()


def test_owner_scoped_messages_preserve_assistant_card(pool):
    from app.persistence.repositories import ChatRepository, OwnerIdentity

    repository = ChatRepository(pool)
    owner = OwnerIdentity(anonymous_owner_hash="owner-a")
    other = OwnerIdentity(anonymous_owner_hash="owner-b")

    repository.create_session(SESSION_ID, owner, title="Ca kiểm thử")
    repository.add_message(SESSION_ID, owner, "user", "Tôi 20 tuổi")
    repository.add_message(
        SESSION_ID,
        owner,
        "assistant",
        "Kết quả sàng lọc",
        assistant_card={"risk_label": "Thấp", "risk_score_10": 0.2},
    )

    messages = repository.messages(SESSION_ID, owner)

    assert len(messages) == 2
    assert messages[1]["assistant_card"] == {
        "risk_label": "Thấp",
        "risk_score_10": 0.2,
    }
    assert repository.messages(SESSION_ID, other) == []
    assert repository.sessions(other) == []
    assert repository.rename(SESSION_ID, other, "Không được phép") is False
    assert repository.soft_delete(SESSION_ID, other) is False


def test_profile_merge_keeps_prior_fields_and_tracks_evidence(pool):
    from app.persistence.repositories import ChatRepository, OwnerIdentity, ProfileRepository

    owner = OwnerIdentity(anonymous_owner_hash="owner-a")
    ChatRepository(pool).create_session(SESSION_ID, owner, title="Profile")
    repository = ProfileRepository(pool)

    first = repository.merge(
        SESSION_ID,
        owner,
        {"age": 20, "gender": "male"},
        {"age": {"quote": "20 tuổi"}},
    )
    second = repository.merge(
        SESSION_ID,
        owner,
        {"bmi": 25, "avg_glucose_level": 70},
        {"bmi": {"quote": "BMI là 25"}},
    )

    assert first["age"] == 20
    assert second["age"] == 20
    assert second["gender"] == "male"
    assert second["bmi"] == 25
    assert second["avg_glucose_level"] == 70
    assert second["source_evidence"]["bmi"]["quote"] == "BMI là 25"

    cleared = repository.merge(SESSION_ID, owner, {}, clear_fields={"avg_glucose_level"})
    assert cleared["avg_glucose_level"] is None
    assert cleared["age"] == 20


def test_auth_repository_persists_only_hashed_session_token(pool):
    from app.persistence.repositories import AuthRepository

    repository = AuthRepository(pool)
    user = repository.create_password_user(
        email=AUTH_EMAIL,
        full_name="Repository Test",
        password_salt="salt",
        password_hash="hash",
    )
    raw_token = "sg_disposable_auth_token"
    repository.create_auth_session(
        user["id"],
        raw_token,
        datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert repository.user_by_token(raw_token)["email"] == AUTH_EMAIL
    with pool.connection() as connection:
        stored = connection.execute(
            "SELECT token_hash FROM strokeguard.auth_sessions WHERE user_id = %s",
            (user["id"],),
        ).fetchone()[0]
    assert stored != raw_token

    repository.delete_auth_session(raw_token)
    assert repository.user_by_token(raw_token) is None


def test_conversation_turn_rolls_back_when_prediction_persistence_fails(pool, monkeypatch):
    from app.extraction.schemas import ScreeningProfileUpdate
    from app.persistence.conversation_store import PostgresConversationStore
    from app.persistence.repositories import PredictionRepository

    store = PostgresConversationStore(pool, "atomic-owner")
    store.ensure_session(str(SESSION_ID), "Atomic turn")
    profile = {
        "age": 20,
        "gender": "male",
        "systolic_bp": 100,
        "diastolic_bp": 70,
        "hypertension": False,
        "heart_disease": False,
        "avg_glucose_level": 70,
        "bmi": 25,
    }
    update = ScreeningProfileUpdate(
        age=20,
        gender="male",
        systolic_bp=100,
        diastolic_bp=70,
        heart_disease=False,
        avg_glucose_level=70,
        bmi=25,
    )
    prediction = {
        "model_used": "stroke_model_clinical.joblib",
        "model_version": "test",
        "risk_probability": 0.02,
        "risk_label": "medium_risk",
        "threshold": 0.03,
        "detected_features": profile,
        "filled_features": {},
    }
    monkeypatch.setattr(
        PredictionRepository,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    with pytest.raises(RuntimeError, match="forced failure"):
        store.save_turn(
            str(SESSION_ID),
            "Thông tin",
            {
                "reply": "Kết quả",
                "current_profile": profile,
                "extracted_update": update.model_dump(mode="json"),
                "prediction_result": prediction,
                "assistant_card": {"risk_label": "Trung bình"},
            },
        )

    assert store.messages(str(SESSION_ID)) == []


def test_conversation_turn_persists_reply_context_in_user_metadata(pool):
    from app.persistence.conversation_store import PostgresConversationStore

    store = PostgresConversationStore(pool, "reply-context-owner")
    store.ensure_session(str(SESSION_ID), "Câu hỏi đầu")
    reply_context = {
        "message_id": "assistant-message-1",
        "quote": "Đoạn được chọn",
    }

    messages = store.save_turn(
        str(SESSION_ID),
        "Giải thích thêm",
        {"reply": "Nội dung giải thích", "current_profile": {}},
        reply_context=reply_context,
    )

    assert messages[0]["reply_context"] == reply_context
