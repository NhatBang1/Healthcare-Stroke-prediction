from __future__ import annotations

from app.extraction.deepseek_tool import ExtractionUnavailable
from app.extraction.schemas import EvidenceItem, ScreeningProfileUpdate


class FakePredictionService:
    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, profile: dict):
        self.calls.append(dict(profile))
        return {
            "model_used": "stroke_model_clinical.joblib",
            "model_status": "loaded",
            "model_version": "test",
            "risk_probability": 0.02,
            "risk_label": "medium_risk",
            "threshold": 0.03,
            "detected_features": dict(profile),
            "filled_features": {},
            "missing_user_fields": [],
            "disclaimer": "Sàng lọc",
        }


class FakeExtractor:
    def __init__(self, updates: dict[str, ScreeningProfileUpdate | Exception]):
        self.updates = updates

    def extract(self, *, new_message, current_profile, recent_messages):
        result = self.updates[new_message]
        if isinstance(result, Exception):
            raise result
        return result


def make_workflow(updates):
    from app.chat.graph import ScreeningWorkflow

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=FakeExtractor(updates),
        prediction_service=prediction,
        general_responder=lambda message, history: "general response",
    )
    return workflow, prediction


def test_graph_accumulates_turns_and_predicts_only_after_sixth_field():
    first_message = "Nam, huyết áp 100/70, glucose tầm 70, 20 tuổi, BMI là 25"
    second_message = "Tôi không có bệnh tim"
    workflow, prediction = make_workflow(
        {
            first_message: ScreeningProfileUpdate(
                age=20,
                gender="male",
                systolic_bp=100,
                diastolic_bp=70,
                avg_glucose_level=70,
                bmi=25,
            ),
            second_message: ScreeningProfileUpdate(
                heart_disease=False,
                evidence=[EvidenceItem(field="heart_disease", quote="không có bệnh tim")],
            ),
        }
    )

    first = workflow.invoke(message=first_message, current_profile={}, recent_messages=[])

    assert prediction.calls == []
    assert first["missing_required_fields"] == ["heart_disease"]
    assert first["current_profile"]["hypertension"] is False
    assert "100/70" in first["reply"]
    assert "bình thường" in first["reply"].lower()

    second = workflow.invoke(
        message=second_message,
        current_profile=first["current_profile"],
        recent_messages=[{"role": "user", "content": first_message}],
    )

    assert len(prediction.calls) == 1
    assert prediction.calls[0]["heart_disease"] is False
    assert second["prediction_result"]["risk_label"] == "medium_risk"
    assert second["assistant_card"]["risk_score_10"] == 2.0
    assert "=" not in second["reply"]


def test_profile_corrections_after_emergency_history_still_predict():
    first_message = "Tôi là nam, 20 tuổi, huyết áp 100/80, không có bệnh tim, glucose 100, BMI 25"
    second_message = "Tôi là nam, 65 tuổi, Huyết áp là 150/100, có bệnh tim. glucose 150, bmi là 30"
    workflow, prediction = make_workflow(
        {
            first_message: ScreeningProfileUpdate(
                age=20,
                gender="male",
                systolic_bp=100,
                diastolic_bp=80,
                heart_disease=False,
                avg_glucose_level=100,
                bmi=25,
            ),
            second_message: ScreeningProfileUpdate(
                age=65,
                gender="male",
                systolic_bp=150,
                diastolic_bp=100,
                heart_disease=True,
                avg_glucose_level=150,
                bmi=30,
                corrections=["age corrected from 20 to 65"],
            ),
        }
    )

    first = workflow.invoke(message=first_message, current_profile={}, recent_messages=[])
    second = workflow.invoke(
        message=second_message,
        current_profile=first["current_profile"],
        recent_messages=[
            {"role": "assistant", "content": "Các dấu hiệu bạn mô tả có thể cần cấp cứu."},
        ],
    )

    assert len(prediction.calls) == 2
    assert prediction.calls[-1]["age"] == 65
    assert prediction.calls[-1]["heart_disease"] is True
    assert second["assistant_card"] is not None


def test_graph_does_not_treat_negated_weakness_as_emergency():
    message = "Tôi đau đầu nhẹ nhưng không yếu tay"
    workflow, prediction = make_workflow({message: ScreeningProfileUpdate()})

    result = workflow.invoke(message=message, current_profile={}, recent_messages=[])

    assert result["safety_level"] == "normal"
    assert result["reply"] == "general response"
    assert prediction.calls == []


def test_reply_context_routes_to_general_without_extracting_quoted_values():
    from app.chat.graph import ScreeningWorkflow

    captured: dict[str, object] = {}

    class RejectingExtractor:
        def extract(self, **kwargs):
            raise AssertionError(f"reply quote reached extractor: {kwargs}")

    def responder(message, history):
        captured["message"] = message
        captured["history"] = history
        return "Giải thích theo đoạn đã chọn"

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=RejectingExtractor(),
        prediction_service=prediction,
        general_responder=responder,
    )

    result = workflow.invoke(
        message="Con số này có nghĩa là gì?",
        reply_context={
            "message_id": "assistant-1",
            "quote": "Glucose 170 và BMI 29 là các giá trị cần theo dõi.",
        },
        current_profile={
            "age": 65,
            "gender": "male",
            "hypertension": True,
            "heart_disease": True,
            "avg_glucose_level": 150,
            "bmi": 30,
        },
        recent_messages=[],
    )

    assert result["reply"] == "Giải thích theo đoạn đã chọn"
    assert "Glucose 170" in str(captured["message"])
    assert "Con số này có nghĩa là gì?" in str(captured["message"])
    assert "<reply_quote>" not in str(captured["message"])
    assert prediction.calls == []


def test_complete_profile_routes_non_screening_symptom_to_general_text():
    from app.chat.graph import ScreeningWorkflow

    class RejectingExtractor:
        def extract(self, **kwargs):
            raise AssertionError("a non-screening symptom must not enter the extractor")

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=RejectingExtractor(),
        prediction_service=prediction,
        general_responder=lambda message, history: "Tư vấn triệu chứng dạng text",
    )
    complete_profile = {
        "age": 20,
        "gender": "male",
        "hypertension": False,
        "heart_disease": False,
        "avg_glucose_level": 70,
        "bmi": 25,
    }

    result = workflow.invoke(
        message="Mình dạo này hay đau đầu không có triệu chứng rõ ràng",
        current_profile=complete_profile,
        recent_messages=[],
    )

    assert result["reply"] == "Tư vấn triệu chứng dạng text"
    assert result.get("assistant_card") is None
    assert prediction.calls == []


def test_general_symptom_route_can_attach_optional_education_image():
    from app.chat.graph import ScreeningWorkflow

    class RejectingExtractor:
        def extract(self, **kwargs):
            raise AssertionError("a non-screening symptom must not enter the extractor")

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=RejectingExtractor(),
        prediction_service=prediction,
        general_responder=lambda message, history: "Tư vấn triệu chứng dạng text",
        media_searcher=lambda message: [
            {
                "type": "image",
                "url": "https://example.org/stroke-fast.png",
                "title": "Dấu hiệu FAST",
                "source_url": "https://example.org/stroke-fast",
            }
        ],
    )
    complete_profile = {
        "age": 20,
        "gender": "male",
        "hypertension": False,
        "heart_disease": False,
        "avg_glucose_level": 70,
        "bmi": 25,
    }

    result = workflow.invoke(
        message="Mình dạo này hay đau đầu không có triệu chứng rõ ràng",
        current_profile=complete_profile,
        recent_messages=[],
    )

    assert result["assistant_card"] is None
    assert result["media"][0]["type"] == "image"
    assert prediction.calls == []


def test_general_route_adds_web_source_citations_when_search_returns_results():
    from app.chat.graph import ScreeningWorkflow

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=object(),
        prediction_service=prediction,
        general_responder=lambda message, history: "Giải thích dựa trên nguồn đã tìm.",
        web_searcher=lambda message: [
            {
                "title": "Stroke warning signs",
                "url": "https://www.cdc.gov/stroke/signs_symptoms.htm",
                "content": "Sudden severe headache can be a warning sign.",
            }
        ],
    )

    result = workflow.invoke(
        message="Có bài báo nào nói về dấu hiệu này không?",
        current_profile={
            "age": 65,
            "gender": "male",
            "hypertension": True,
            "heart_disease": True,
            "avg_glucose_level": 150,
            "bmi": 30,
        },
        recent_messages=[],
    )

    assert result["sources"][0]["url"].startswith("https://")
    assert "[Stroke warning signs](https://www.cdc.gov/stroke/signs_symptoms.htm)" in result["reply"]


def test_risk_factor_question_does_not_render_prediction_card_when_profile_is_complete():
    from app.chat.graph import ScreeningWorkflow

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=object(),
        prediction_service=prediction,
        general_responder=lambda message, history: "Hút thuốc và rượu có thể ảnh hưởng đến nguy cơ.",
        web_searcher=lambda message: [],
    )
    result = workflow.invoke(
        message="hút thuốc, uống rượu có ảnh hưởng đến nguy cơ đột quỵ không?",
        current_profile={
            "age": 60,
            "gender": "male",
            "hypertension": True,
            "heart_disease": True,
            "avg_glucose_level": 150,
            "bmi": 25,
        },
        recent_messages=[],
    )

    assert result["assistant_card"] is None
    assert result["reply"].startswith("Hút thuốc")
    assert prediction.calls == []


def test_natural_follow_up_about_alcohol_and_heart_history_uses_general_route():
    from app.chat.graph import ScreeningWorkflow

    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=object(),
        prediction_service=prediction,
        general_responder=lambda message, history: "Tôi sẽ giải thích nguy cơ rượu và suy tim.",
        web_searcher=lambda message: [],
    )
    result = workflow.invoke(
        message="thế còn uống rượu, và có tiền sử bệnh suy tim",
        current_profile={
            "age": 20,
            "gender": "male",
            "hypertension": False,
            "heart_disease": False,
            "avg_glucose_level": 94,
            "bmi": 25,
        },
        recent_messages=[],
    )

    assert result["assistant_card"] is None
    assert result["reply"].startswith("Tôi sẽ giải thích")
    assert prediction.calls == []


def test_prediction_card_contains_llm_synthesis_for_optional_risk_factors():
    from app.chat.graph import ScreeningWorkflow

    message = "Nam, 20 tuổi, huyết áp 120/80, không bệnh tim, glucose 94, BMI 25, tôi có hút thuốc"
    update = ScreeningProfileUpdate(
        age=20,
        gender="male",
        systolic_bp=120,
        diastolic_bp=80,
        heart_disease=False,
        avg_glucose_level=94,
        bmi=25,
        smoke_flag=True,
        evidence=[
            EvidenceItem(field="age", quote="20 tuổi"),
            EvidenceItem(field="gender", quote="Nam"),
            EvidenceItem(field="systolic_bp", quote="120/80"),
            EvidenceItem(field="diastolic_bp", quote="120/80"),
            EvidenceItem(field="heart_disease", quote="không bệnh tim"),
            EvidenceItem(field="avg_glucose_level", quote="glucose 94"),
            EvidenceItem(field="bmi", quote="BMI 25"),
            EvidenceItem(field="smoke_flag", quote="hút thuốc"),
        ],
    )
    prediction = FakePredictionService()
    workflow = ScreeningWorkflow(
        extractor=FakeExtractor({message: update}),
        prediction_service=prediction,
        general_responder=lambda message, history: "general",
        synthesis_responder=lambda card, profile: "Hút thuốc là yếu tố nguy cơ cần lưu ý.",
    )

    result = workflow.invoke(message=message, current_profile={}, recent_messages=[])

    assert result["assistant_card"]["interpretation"] == "Hút thuốc là yếu tố nguy cơ cần lưu ý."
    assert {item["label"] for item in result["assistant_card"]["feature_cards"]} >= {"Hút thuốc"}
    assert "Hút thuốc là yếu tố nguy cơ" in result["reply"]


def test_completed_profile_requires_confirmation_before_new_prediction():
    message = "Nam, 20 tuổi, huyết áp ở mức 120/80, không có bệnh tim, glucose khoảng 94, bmi là 25"
    confirm = "Có"
    update = ScreeningProfileUpdate(
        age=20,
        gender="male",
        systolic_bp=120,
        diastolic_bp=80,
        heart_disease=False,
        avg_glucose_level=94,
        bmi=25,
    )
    workflow, prediction = make_workflow({message: update, confirm: ScreeningProfileUpdate()})
    profile = update.profile_updates() | {"hypertension": False}
    previous_card = {"risk_score_10": 2.0, "risk_label": "Thấp"}

    pending = workflow.invoke(
        message=message,
        current_profile=profile,
        recent_messages=[{"role": "assistant", "content": "Kết quả trước", "assistant_card": previous_card}],
    )

    assert pending["assistant_card"] is None
    assert "tạo một dự đoán mới" in pending["reply"]
    assert prediction.calls == []

    completed = workflow.invoke(
        message=confirm,
        current_profile=profile,
        recent_messages=[
            {"role": "assistant", "content": "Kết quả trước", "assistant_card": previous_card},
            {"role": "assistant", "content": pending["reply"]},
        ],
    )

    assert completed["assistant_card"] is not None
    assert len(prediction.calls) == 1


def test_graph_returns_controlled_response_when_extraction_is_unavailable():
    message = "Tôi 20 tuổi"
    workflow, prediction = make_workflow(
        {message: ExtractionUnavailable("provider timeout")}
    )

    result = workflow.invoke(message=message, current_profile={}, recent_messages=[])

    assert "tạm thời" in result["reply"]
    assert result["missing_required_fields"] == []
    assert prediction.calls == []


def test_adult_blood_pressure_classification_and_special_populations():
    from app.chat.blood_pressure import classify_blood_pressure

    assert classify_blood_pressure(20, 100, 70).category == "normal"
    assert classify_blood_pressure(20, 100, 70).hypertension is False
    assert classify_blood_pressure(20, 130, 80).category == "stage_1"
    assert classify_blood_pressure(20, 130, 80).hypertension is True
    assert classify_blood_pressure(17, 130, 80).category == "unknown"
    assert classify_blood_pressure(20, 130, None).category == "unknown"
    assert classify_blood_pressure(20, 130, 80, pregnant=True).category == "unknown"


def test_severe_blood_pressure_routes_to_safety_and_never_predicts():
    message = "Tôi 20 tuổi, huyết áp 190/130"
    workflow, prediction = make_workflow(
        {message: ScreeningProfileUpdate(age=20, systolic_bp=190, diastolic_bp=130)}
    )

    result = workflow.invoke(message=message, current_profile={}, recent_messages=[])

    assert result["safety_level"] == "emergency"
    assert "cấp cứu" in result["reply"].lower()
    assert prediction.calls == []


def test_severe_blood_pressure_in_pregnancy_routes_to_safety():
    message = "Tôi đang mang thai và huyết áp 190/130"
    workflow, prediction = make_workflow(
        {
            message: ScreeningProfileUpdate(
                age=30,
                pregnant=True,
                systolic_bp=190,
                diastolic_bp=130,
            )
        }
    )

    result = workflow.invoke(message=message, current_profile={}, recent_messages=[])

    assert result["safety_level"] == "emergency"
    assert prediction.calls == []


def test_age_correction_to_minor_clears_previously_inferred_hypertension():
    message = "Tôi xin sửa lại là 17 tuổi"
    workflow, prediction = make_workflow(
        {message: ScreeningProfileUpdate(age=17, corrections=["age"])}
    )
    current = {
        "age": 20,
        "gender": "male",
        "systolic_bp": 130,
        "diastolic_bp": 80,
        "hypertension": True,
        "heart_disease": False,
        "avg_glucose_level": 70,
        "bmi": 25,
    }

    result = workflow.invoke(message=message, current_profile=current, recent_messages=[])

    assert "hypertension" not in result["current_profile"]
    assert result["missing_required_fields"] == ["hypertension"]
    assert prediction.calls == []
