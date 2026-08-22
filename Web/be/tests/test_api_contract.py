from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_prediction_endpoint_returns_clinical_risk_contract():
    response = client.post(
        "/api/predict/stroke",
        json={
            "age": 67,
            "gender": "male",
            "hypertension": 1,
            "heart_disease": 1,
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "avg_RestingBP": 140,
            "avg_MaxHR": 130,
            "avg_Oldpeak": 1.5,
            "heart_disease_rate": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "stroke_model_clinical.joblib"
    assert body["threshold"] == 0.03
    assert body["risk_label"] in {"low_risk", "medium_risk", "high_risk"}
    assert 0 <= body["risk_probability"] <= 1
    assert body["model_status"] in {"loaded", "fallback"}
    assert "disclaimer" in body
    assert body["detected_features"]["age"] == 67
    assert body["detected_features"]["gender"] == 0


def test_prediction_endpoint_rejects_out_of_range_screening_values():
    response = client.post(
        "/api/predict/stroke",
        json={
            "age": 120,
            "gender": "male",
            "hypertension": 1,
            "heart_disease": 0,
            "avg_glucose_level": 170,
            "bmi": 20,
        },
    )

    assert response.status_code == 422


def test_prediction_endpoint_rejects_missing_required_screening_fields():
    response = client.post("/api/predict/stroke", json={"age": 65})

    assert response.status_code == 422
    assert set(response.json()["detail"]["missing_required_fields"]) == {
        "gender",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
    }


def test_prediction_endpoint_rejects_invalid_categorical_and_binary_values():
    response = client.post(
        "/api/predict/stroke",
        json={
            "age": 65,
            "gender": 999,
            "hypertension": -7,
            "heart_disease": 2,
            "avg_glucose_level": 120,
            "bmi": 25,
        },
    )

    assert response.status_code == 422


def test_chat_session_can_be_reset():
    first = client.post(
        "/api/chat/message",
        json={"session_id": "reset-test", "message": "Tôi 65 tuổi, glucose 170"},
    )
    assert first.status_code == 200
    assert first.json()["session_id"] == "reset-test"

    sessions = client.get("/api/sessions")
    assert sessions.status_code == 200
    session_items = sessions.json()["sessions"]
    assert any(item["session_id"] == "reset-test" and item["title"].startswith("Tôi 65 tuổi") for item in session_items)

    history = client.get("/api/sessions/reset-test/messages")
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 2

    reset = client.post("/api/sessions/reset-test/reset")
    assert reset.status_code == 200
    assert reset.json() == {"session_id": "reset-test", "reset": True}

    cleared = client.get("/api/sessions/reset-test/messages")
    assert cleared.status_code == 200
    assert cleared.json()["messages"] == []

    sessions_after_reset = client.get("/api/sessions")
    assert sessions_after_reset.status_code == 200
    assert all(item["session_id"] != "reset-test" for item in sessions_after_reset.json()["sessions"])


def test_session_can_be_renamed_and_deleted():
    first = client.post(
        "/api/chat/message",
        json={"session_id": "rename-delete-test", "message": "Tôi 21 tuổi, glucose 130"},
    )
    assert first.status_code == 200

    rename = client.patch(
        "/api/sessions/rename-delete-test/title",
        json={"title": "Ca screening đầu tiên"},
    )
    assert rename.status_code == 200
    assert rename.json()["title"] == "Ca screening đầu tiên"

    sessions = client.get("/api/sessions")
    assert sessions.status_code == 200
    session_items = sessions.json()["sessions"]
    assert any(item["session_id"] == "rename-delete-test" and item["title"] == "Ca screening đầu tiên" for item in session_items)

    second = client.post(
        "/api/chat/message",
        json={"session_id": "rename-delete-test", "message": "BMI 27"},
    )
    assert second.status_code == 200

    sessions_after_message = client.get("/api/sessions")
    assert sessions_after_message.status_code == 200
    assert any(item["session_id"] == "rename-delete-test" and item["title"] == "Ca screening đầu tiên" for item in sessions_after_message.json()["sessions"])

    deleted = client.delete("/api/sessions/rename-delete-test")
    assert deleted.status_code == 200
    assert deleted.json() == {"session_id": "rename-delete-test", "deleted": True}

    sessions_after_delete = client.get("/api/sessions")
    assert sessions_after_delete.status_code == 200
    assert all(item["session_id"] != "rename-delete-test" for item in sessions_after_delete.json()["sessions"])


def test_chat_uses_deepseek_client_when_available(monkeypatch):
    from app import main

    def fake_complete(messages):
        assert messages[-1]["role"] == "user"
        assert "đột quỵ" in messages[-1]["content"].lower()
        return "Đây là câu trả lời từ DeepSeek đã được kiểm soát an toàn."

    monkeypatch.setattr(main.deepseek_client, "configured", lambda: True)
    monkeypatch.setattr(main.deepseek_client, "complete", fake_complete)

    response = client.post(
        "/api/chat/message",
        json={"session_id": "deepseek-test", "message": "Dấu hiệu đột quỵ là gì?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Đây là câu trả lời từ DeepSeek đã được kiểm soát an toàn."
    assert body["assistant_card"] is None


def test_chat_keeps_reply_context_separate_and_persists_it(monkeypatch):
    from app import main

    seed = client.post(
        "/api/chat/message",
        json={"session_id": "reply-context-test", "message": "Dấu hiệu đột quỵ là gì?"},
    )
    assert seed.status_code == 200
    assistant_message = seed.json()["messages"][-1]
    captured: dict[str, object] = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return {
            "reply": "Tôi sẽ giải thích đoạn đã chọn.",
            "current_profile": kwargs["current_profile"],
            "assistant_card": None,
            "missing_required_fields": [],
        }

    monkeypatch.setattr(main.screening_workflow, "invoke", fake_invoke)
    reply_context = {
        "message_id": assistant_message["id"],
        "quote": assistant_message["content"].split(".")[0] + ".",
    }

    response = client.post(
        "/api/chat/message",
        json={
            "session_id": "reply-context-test",
            "message": "Bạn giải thích kỹ hơn được không?",
            "reply_context": reply_context,
        },
    )

    assert response.status_code == 200
    assert captured["message"] == "Bạn giải thích kỹ hơn được không?"
    assert captured["reply_context"] == reply_context
    assert response.json()["messages"][-2]["reply_context"] == reply_context

    history = client.get("/api/sessions/reply-context-test/messages")
    assert history.status_code == 200
    assert history.json()["messages"][-2]["reply_context"] == reply_context


def test_chat_rejects_reply_context_that_is_not_from_the_session():
    response = client.post(
        "/api/chat/message",
        json={
            "session_id": "invalid-reply-context-test",
            "message": "Giải thích thêm",
            "reply_context": {
                "message_id": "not-a-session-message",
                "quote": "Nội dung tự nhận là của StrokeGuard AI",
            },
        },
    )

    assert response.status_code == 422
    assert "đoạn trích" in str(response.json()["detail"]).lower()


def test_reply_context_accepts_quote_from_prediction_interpretation():
    from app.main import _validate_reply_context

    context = {
        "message_id": "assistant-card-1",
        "quote": "Hút thuốc làm tăng nguy cơ đột quỵ.",
    }
    assert _validate_reply_context(
        context,
        [
            {
                "id": "assistant-card-1",
                "role": "assistant",
                "content": "Tôi đã chạy model.",
                "assistant_card": {
                    "summary": "Kết quả sàng lọc",
                    "interpretation": "Hút thuốc làm tăng nguy cơ đột quỵ.",
                },
            }
        ],
    ) == context
    assert _validate_reply_context(
        {"message_id": "client-wrapper-id", "quote": context["quote"]},
        [
            {
                "id": "persisted-id",
                "role": "assistant",
                "content": "Tôi đã chạy model.",
                "assistant_card": {"interpretation": context["quote"]},
            }
        ],
    )["quote"] == context["quote"]
    assert _validate_reply_context(
        {"message_id": "client-wrapper-id", "quote": "Hút thuốc làm tăng nguy cơ đột quỵ"},
        [
            {
                "id": "persisted-id",
                "role": "assistant",
                "content": "Tôi đã chạy model.",
                "assistant_card": {"interpretation": "Hút thuốc làm tăng nguy cơ đột quỵ."},
            }
        ],
    )["quote"].startswith("Hút thuốc")


def test_chat_persists_general_response_media_without_an_assistant_card(monkeypatch):
    from app import main

    media = [{
        "type": "image",
        "url": "https://example.org/fast.png",
        "title": "Dấu hiệu FAST",
        "source_url": "https://example.org/fast",
    }]

    monkeypatch.setattr(
        main.screening_workflow,
        "invoke",
        lambda **kwargs: {
            "reply": "Tư vấn triệu chứng dạng text",
            "current_profile": kwargs["current_profile"],
            "assistant_card": None,
            "media": media,
            "missing_required_fields": [],
        },
    )

    response = client.post(
        "/api/chat/message",
        json={"session_id": "media-contract-test", "message": "Hay đau đầu"},
    )

    assert response.status_code == 200
    assert response.json()["assistant_card"] is None
    assert response.json()["media"] == media
    assert response.json()["messages"][-1]["media"] == media


def test_chat_can_reply_to_assistant_message_older_than_context_window(monkeypatch):
    from app import main
    from app.memory import MemorySessionStore

    assert isinstance(main.session_store, MemorySessionStore)
    session_id = "old-reply-context-test"
    for index in range(11):
        main.session_store.add_turn(
            session_id,
            f"Câu hỏi {index}",
            f"Câu trả lời cũ số {index}",
        )
    old_assistant = main.session_store.messages(session_id)[1]
    captured: dict[str, object] = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return {
            "reply": "Giải thích câu trả lời cũ",
            "current_profile": kwargs["current_profile"],
            "assistant_card": None,
            "missing_required_fields": [],
        }

    monkeypatch.setattr(main.screening_workflow, "invoke", fake_invoke)
    response = client.post(
        "/api/chat/message",
        json={
            "session_id": session_id,
            "message": "Giải thích lại đoạn này",
            "reply_context": {
                "message_id": old_assistant["id"],
                "quote": old_assistant["content"],
            },
        },
    )

    assert response.status_code == 200
    assert len(captured["recent_messages"]) == 20


def test_chat_answers_general_stroke_questions_without_fallback():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "stroke-question-test", "message": "Dấu hiệu đột quỵ là gì?"},
    )

    assert response.status_code == 200
    reply = response.json()["reply"].lower()
    assert "trợ lý ai" not in reply
    assert "đột quỵ" in reply
    assert "dấu hiệu thường gặp" in reply or "fast" in reply or "méo miệng" in reply
    assert response.json()["assistant_card"] is None


def test_chat_replies_helpfully_to_medical_symptom_questions():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "symptom-test", "message": "Tôi thức dậy bị đau đầu nhiều có sao k"},
    )

    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()
    assert "trợ lý ai" not in reply
    assert "đau đầu" in reply or "triệu chứng" in reply or "cấp cứu" in reply
    assert body["assistant_card"] is None


def test_chat_only_redirects_very_short_non_medical_messages():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "short-test", "message": "hi"},
    )

    assert response.status_code == 200
    reply = response.json()["reply"].lower()
    assert "trợ lý ai về tầm soát nguy cơ đột quỵ" in reply


def test_chat_marks_emergency_symptoms_as_critical():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "emergency-test", "message": "Tôi bị méo miệng và yếu tay phải"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "cấp cứu" in body["reply"].lower()
    assert body["assistant_card"]["alert_level"] == "critical"


def test_chat_triages_short_medical_symptoms_without_fallback():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "triage-test", "message": "đau đầu nhiều"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "đau đầu" in body["reply"].lower()
    assert "trợ lý ai" not in body["reply"].lower()
    assert body["assistant_card"] is None


def test_chat_requires_all_fields_then_persists_prediction_card(monkeypatch):
    from app import main
    from app.extraction.schemas import ScreeningProfileUpdate

    monkeypatch.setattr(
        main.profile_extractor,
        "extract",
        lambda **_: ScreeningProfileUpdate(
            age=20,
            gender="male",
            systolic_bp=100,
            diastolic_bp=70,
            heart_disease=False,
            avg_glucose_level=70,
            bmi=25,
        ),
    )
    response = client.post(
        "/api/chat/message",
        json={
            "session_id": "feature-card-test",
            "message": "Nam, huyết áp 100/70, glucose tầm 70, 20 tuổi, BMI là 25, không bệnh tim.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "/10" in body["reply"]
    card = body["assistant_card"]
    assert card["risk_score_10"] <= 10
    feature_labels = [item["label"] for item in card["feature_cards"]]
    assert feature_labels == ["Huyết áp", "Tuổi", "Giới tính", "Tăng huyết áp", "Bệnh tim", "Glucose", "BMI", "Lượng calo ước tính"]
    assert card["feature_cards"][0]["value"] == "100/70 - Bình thường"
    calories_card = card["feature_cards"][-1]
    assert calories_card["label"] == "Lượng calo ước tính"
    assert calories_card["value"].endswith(" kcal")

    history = client.get("/api/sessions/feature-card-test/messages").json()["messages"]
    assert history[-1]["assistant_card"] == card


def test_chat_does_not_predict_when_extractor_is_unavailable():
    response = client.post(
        "/api/chat/message",
        json={"session_id": "range-guard-test", "message": "Tôi 120 tuổi, nam, huyết áp cao, glucose 500, bmi 5."},
    )

    assert response.status_code == 200
    body = response.json()
    assert "tạm thời" in body["reply"].lower()
    assert body["assistant_card"] is None


def test_chat_does_not_predict_with_missing_screening_values(monkeypatch):
    from app import main
    from app.extraction.schemas import ScreeningProfileUpdate

    monkeypatch.setattr(
        main.profile_extractor,
        "extract",
        lambda **_: ScreeningProfileUpdate(age=20, systolic_bp=100, diastolic_bp=70, bmi=25),
    )
    response = client.post(
        "/api/chat/message",
        json={"session_id": "unknown-guard-test", "message": "20 tuổi, huyết áp bình thường, không biết glucose, BMI 25."},
    )

    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()
    assert "vui lòng bổ sung" in reply
    assert "glucose trung bình" in reply
    assert "bệnh tim" in reply
    assert body["assistant_card"] is None


def test_health_and_ready_endpoints():
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] in {"ready", "degraded"}
