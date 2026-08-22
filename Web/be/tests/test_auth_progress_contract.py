from fastapi.testclient import TestClient

from app.core.config import settings
from app.memory import session_store
from app.main import app
from app.security import rate_limiter


client = TestClient(app)


def setup_function() -> None:
    rate_limiter.reset()
    for item in list(session_store.sessions()):
        session_store.reset(str(item["session_id"]))


def test_auth_register_login_me_and_google_redirect_contract(monkeypatch):
    monkeypatch.setattr(settings, "auth_rate_limit_requests", 10)

    register = client.post(
        "/api/auth/register",
        json={
            "email": "user@example.com",
            "password": "secret123",
            "full_name": "Nguyen Van User",
        },
    )

    assert register.status_code == 200
    created = register.json()
    assert created["email"] == "user@example.com"
    assert created["full_name"] == "Nguyen Van User"
    assert created["access_token"]
    assert created["token_type"] == "bearer"
    assert "strokeguard_token=" in register.headers.get("set-cookie", "")

    me = client.get(
        "/api/auth/me",
    )
    assert me.status_code == 200
    assert me.json()["email"] == "user@example.com"
    assert me.json()["is_admin"] is False

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert logout.json()["ok"] is True

    after_logout = client.get("/api/auth/me")
    assert after_logout.status_code == 401

    session = client.get("/api/session/me")
    assert session.status_code == 200
    assert session.json()["authenticated"] is False
    assert session.json()["user"] is None

    login = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "secret123"},
    )
    assert login.status_code == 200
    assert login.json()["access_token"]

    session_after_login = client.get("/api/session/me")
    assert session_after_login.status_code == 200
    assert session_after_login.json()["authenticated"] is True
    assert session_after_login.json()["user"]["email"] == "user@example.com"

    monkeypatch.setattr(settings, "google_client_id", "google-client")
    monkeypatch.setattr(settings, "google_client_secret", "google-secret")
    monkeypatch.setattr(settings, "frontend_public_url", "http://frontend.local")
    monkeypatch.setattr(settings, "api_public_url", "http://api.local/api")

    google = client.get("/api/auth/oauth/google?mode=login")
    assert google.status_code == 200
    url = google.json()["url"]
    assert "accounts.google.com" in url
    assert "state=login" in url
    assert "redirect_uri=" in url


def test_admin_documents_require_admin_auth(monkeypatch):
    monkeypatch.setattr(settings, "admin_emails_raw", "admin@example.com")

    anon = TestClient(app)
    forbidden = anon.get("/api/admin/documents")
    assert forbidden.status_code in {401, 403}

    admin_register = anon.post(
        "/api/auth/register",
        json={
            "email": "admin@example.com",
            "password": "secret12345",
            "full_name": "Admin User",
        },
    )
    assert admin_register.status_code == 200
    docs = anon.get("/api/admin/documents")
    assert docs.status_code == 200
    assert docs.json()["documents"]


def test_progress_summary_tracks_persisted_predictions(monkeypatch):
    from app import main
    from app.extraction.schemas import ScreeningProfileUpdate

    def extract(**kwargs):
        if "Không có bệnh tim" in kwargs["new_message"]:
            return ScreeningProfileUpdate(heart_disease=False)
        return ScreeningProfileUpdate(
            age=65,
            gender="male",
            systolic_bp=145,
            diastolic_bp=92,
            heart_disease=True,
            avg_glucose_level=170,
            bmi=29,
        )

    monkeypatch.setattr(main.profile_extractor, "extract", extract)
    first = client.post(
        "/api/chat/message",
        json={"session_id": "progress-test", "message": "Tôi 65 tuổi, tăng huyết áp, glucose 170, BMI 29."},
    )
    assert first.status_code == 200

    progress = client.get("/api/progress/progress-test")
    assert progress.status_code == 200
    body = progress.json()
    assert body["session_id"] == "progress-test"
    assert body["current_risk"]["label"] in {"Thấp", "Trung bình", "Cao"}
    assert len(body["chart_points"]) >= 1
    assert body["entries"][0]["source"] == "model"

    second = client.post(
        "/api/chat/message",
        json={"session_id": "progress-test", "message": "Không có bệnh tim."},
    )
    assert second.status_code == 200

    updated = client.get("/api/progress/progress-test")
    assert updated.status_code == 200
    updated_body = updated.json()
    assert len(updated_body["chart_points"]) >= 2
    assert updated_body["current_risk"]["trend"] in {"Tăng", "Giảm", "Ổn định", "Chờ dữ liệu"}


def test_chat_message_with_complete_screening_data_uses_model_reply(monkeypatch):
    from app import main
    from app.extraction.schemas import ScreeningProfileUpdate

    monkeypatch.setattr(
        main.profile_extractor,
        "extract",
        lambda **_: ScreeningProfileUpdate(
            age=21,
            gender="male",
            systolic_bp=135,
            diastolic_bp=85,
            heart_disease=False,
            avg_glucose_level=130,
            bmi=27,
        ),
    )
    response = client.post(
        "/api/chat/message",
        json={"session_id": "model-route-test", "message": "Tôi 21 tuổi, bị tăng huyết áp, glucose 130, BMI 27."},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Tôi đã chạy model sàng lọc" in body["reply"]
    assert "nguy cơ" in body["reply"]
    assert body["assistant_card"]["model_used"] == "stroke_model_clinical.joblib"
    assert body["assistant_card"]["risk_probability_percent"] <= 100
    assert body["assistant_card"]["confidence_percent"] == 92.0
    assert body["assistant_card"]["verified"] is True


def test_session_history_has_its_own_endpoint():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "history-route-test", "message": "Tôi 65 tuổi, tăng huyết áp, glucose 170, BMI 29."},
    )
    assert response.status_code == 200

    history = client.get("/api/sessions/history-route-test/history-summary")
    assert history.status_code == 200
    body = history.json()
    assert body["session_id"] == "history-route-test"
    assert body["message_count"] >= 2
    assert body["has_activity"] is True
    assert len(body["recent_messages"]) >= 2


def test_invalid_email_is_rejected_by_schema():
    response = client.post(
        "/api/auth/login",
        json={"email": "not-an-email", "password": "secret123"},
    )

    assert response.status_code == 422


def test_auth_rate_limit_blocks_after_five_attempts():
    for _ in range(5):
        response = client.post(
            "/api/auth/login",
            json={"email": "ratelimit@example.com", "password": "wrongpass123"},
        )
        assert response.status_code == 401

    limited = client.post(
        "/api/auth/login",
        json={"email": "ratelimit@example.com", "password": "wrongpass123"},
    )

    assert limited.status_code == 429
    assert limited.json()["detail"] == "Too many requests"


def test_large_payload_is_rejected_before_schema_validation():
    response = client.post(
        "/api/chat/message",
        json={
            "session_id": "payload-limit",
            "message": "Xin chào",
            "padding": "x" * 50000,
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"
