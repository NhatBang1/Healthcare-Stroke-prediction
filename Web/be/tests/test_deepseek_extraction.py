from __future__ import annotations

import json

import httpx
import pytest


SAMPLE = "Nam, huyết áp 100/70, glucose tầm 70, 20 tuổi, BMI là 25, không có bệnh tim"


def tool_response(arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_extract_1",
                            "type": "function",
                            "function": {
                                "name": "extract_screening_profile",
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ]
    }


def make_extractor(arguments: dict, *, expected_message: str = SAMPLE):
    from app.extraction.deepseek_tool import DeepSeekProfileExtractor

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tool_choice"]["function"]["name"] == "extract_screening_profile"
        assert payload["tools"][0]["function"]["name"] == "extract_screening_profile"
        assert expected_message in payload["messages"][-1]["content"]
        return httpx.Response(200, json=tool_response(arguments))

    return DeepSeekProfileExtractor(
        api_key="test-key",
        base_url="https://deepseek.invalid",
        model="deepseek-test",
        transport=httpx.MockTransport(handler),
    )


def test_extractor_accepts_natural_vietnamese_tool_arguments():
    extractor = make_extractor(
        {
            "age": 20,
            "gender": "male",
            "systolic_bp": 100,
            "diastolic_bp": 70,
            "hypertension_history": None,
            "heart_disease": False,
            "avg_glucose_level": 70,
            "bmi": 25,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [
                {"field": "age", "quote": "20 tuổi"},
                {"field": "gender", "quote": "Nam"},
                {"field": "systolic_bp", "quote": "100/70"},
                {"field": "diastolic_bp", "quote": "100/70"},
                {"field": "avg_glucose_level", "quote": "glucose tầm 70"},
                {"field": "bmi", "quote": "BMI là 25"},
                {"field": "heart_disease", "quote": "không có bệnh tim"},
            ],
        },
    )

    result = extractor.extract(new_message=SAMPLE, current_profile={}, recent_messages=[])

    assert result.age == 20
    assert result.gender == "male"
    assert result.systolic_bp == 100
    assert result.diastolic_bp == 70
    assert result.avg_glucose_level == 70
    assert result.bmi == 25


def test_extractor_retries_once_after_transient_validation_failure(caplog):
    from app.extraction.deepseek_tool import DeepSeekProfileExtractor

    valid = {
        "age": 20,
        "gender": "male",
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "hypertension_history": None,
        "heart_disease": False,
        "avg_glucose_level": 94,
        "bmi": 25,
        "pregnant": None,
        "explicit_unknown_fields": [],
        "corrections": [],
        "evidence": [
            {"field": "age", "quote": "20 tuổi"},
            {"field": "gender", "quote": "Nam"},
            {"field": "systolic_bp", "quote": "120/80"},
            {"field": "diastolic_bp", "quote": "120/80"},
            {"field": "heart_disease", "quote": "không có bệnh tim"},
            {"field": "avg_glucose_level", "quote": "glucose khoảng 94"},
            {"field": "bmi", "quote": "bmi là 25"},
        ],
    }
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            broken = dict(valid)
            broken["age"] = "not-a-number"
            return httpx.Response(200, json=tool_response(broken))
        return httpx.Response(200, json=tool_response(valid))

    extractor = DeepSeekProfileExtractor(
        api_key="test-key",
        base_url="https://deepseek.invalid",
        model="deepseek-test",
        transport=httpx.MockTransport(handler),
    )

    with caplog.at_level("WARNING"):
        result = extractor.extract(
            new_message="Nam, 20 tuổi, huyết áp 120/80, không có bệnh tim, glucose khoảng 94, bmi là 25",
            current_profile={},
            recent_messages=[],
        )

    assert calls == 2
    assert result.age == 20
    assert "retrying once" in caplog.text


def test_extractor_accepts_descriptive_corrections_when_user_replaces_profile():
    extractor = make_extractor(
        {
            "age": 65,
            "gender": "male",
            "systolic_bp": 150,
            "diastolic_bp": 100,
            "hypertension_history": None,
            "heart_disease": True,
            "avg_glucose_level": 150,
            "bmi": 30,
            "pregnant": None,
            "explicit_unknown_fields": [],
            "corrections": ["age corrected from 20 to 65", "heart disease corrected to true"],
            "evidence": [
                {"field": "age", "quote": "65 tuổi"},
                {"field": "gender", "quote": "Tôi là nam"},
                {"field": "systolic_bp", "quote": "Huyết áp là 150/100"},
                {"field": "diastolic_bp", "quote": "Huyết áp là 150/100"},
                {"field": "heart_disease", "quote": "có bệnh tim"},
                {"field": "avg_glucose_level", "quote": "glucose 150"},
                {"field": "bmi", "quote": "bmi là 30"},
            ],
        },
        expected_message="Tôi là nam, 65 tuổi, Huyết áp là 150/100, có bệnh tim. glucose 150, bmi là 30",
    )

    result = extractor.extract(
        new_message="Tôi là nam, 65 tuổi, Huyết áp là 150/100, có bệnh tim. glucose 150, bmi là 30",
        current_profile={"age": 20, "gender": "male"},
        recent_messages=[],
    )

    assert result.age == 65
    assert result.corrections == ["age corrected from 20 to 65", "heart disease corrected to true"]


def test_extractor_keeps_unknown_heart_disease_distinct_from_false():
    message = "Tôi không chắc mình có bệnh tim"
    extractor = make_extractor(
        {
            "age": None,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": None,
            "avg_glucose_level": None,
            "bmi": None,
            "explicit_unknown_fields": ["heart_disease"],
            "corrections": [],
            "evidence": [{"field": "heart_disease", "quote": "không chắc"}],
        },
        expected_message=message,
    )

    result = extractor.extract(new_message=message, current_profile={}, recent_messages=[])

    assert result.heart_disease is None
    assert result.explicit_unknown_fields == ["heart_disease"]


def test_extractor_normalizes_provider_string_null_without_losing_false():
    message = "Tôi không có bệnh tim"
    extractor = make_extractor(
        {
            "age": None,
            "gender": "null",
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": False,
            "avg_glucose_level": None,
            "bmi": None,
            "pregnant": None,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [{"field": "heart_disease", "quote": "không có bệnh tim"}],
        },
        expected_message=message,
    )

    result = extractor.extract(new_message=message, current_profile={}, recent_messages=[])

    assert result.gender is None
    assert result.heart_disease is False


def test_extractor_rejects_uncertainty_mislabeled_as_false():
    message = "Tôi không chắc mình có bệnh tim"
    extractor = make_extractor(
        {
            "age": None,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": False,
            "avg_glucose_level": None,
            "bmi": None,
            "pregnant": None,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [{"field": "heart_disease", "quote": "không chắc"}],
        },
        expected_message=message,
    )

    with pytest.raises(ValueError, match="uncertainty"):
        extractor.extract(new_message=message, current_profile={}, recent_messages=[])


def test_extractor_rejects_evidence_missing_from_new_message():
    extractor = make_extractor(
        {
            "age": 65,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": None,
            "avg_glucose_level": None,
            "bmi": None,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [{"field": "age", "quote": "65 tuổi"}],
        }
    )

    with pytest.raises(ValueError, match="evidence"):
        extractor.extract(new_message=SAMPLE, current_profile={}, recent_messages=[])


def test_extractor_rejects_extracted_value_without_evidence():
    extractor = make_extractor(
        {
            "age": 20,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": None,
            "avg_glucose_level": None,
            "bmi": None,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [{"field": "hypertension", "quote": "chưa biết huyết áp"}],
        }
    )

    with pytest.raises(ValueError, match="Missing evidence"):
        extractor.extract(new_message=SAMPLE, current_profile={}, recent_messages=[])


def test_extractor_rejects_numeric_value_inconsistent_with_evidence_quote():
    extractor = make_extractor(
        {
            "age": 65,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": None,
            "avg_glucose_level": None,
            "bmi": None,
            "pregnant": None,
            "explicit_unknown_fields": [],
            "corrections": [],
            "evidence": [{"field": "age", "quote": "20 tuổi"}],
        }
    )

    with pytest.raises(ValueError, match="does not support value"):
        extractor.extract(new_message=SAMPLE, current_profile={}, recent_messages=[])


def test_extractor_does_not_fall_back_without_api_key():
    from app.extraction.deepseek_tool import DeepSeekProfileExtractor, ExtractionUnavailable

    extractor = DeepSeekProfileExtractor(
        api_key=None,
        base_url="https://api.deepseek.com",
        model="deepseek-test",
    )

    with pytest.raises(ExtractionUnavailable, match="not configured"):
        extractor.extract(new_message=SAMPLE, current_profile={}, recent_messages=[])


def test_extractor_normalizes_hypertension_history_unknown_field_alias():
    extractor = make_extractor(
        {
            "age": None,
            "gender": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "hypertension_history": None,
            "heart_disease": None,
            "avg_glucose_level": None,
            "bmi": None,
            "pregnant": None,
            "smoke_flag": None,
            "alco_flag": None,
            "explicit_unknown_fields": ["hypertension_history"],
            "corrections": [],
            "evidence": [{"field": "hypertension", "quote": "chưa biết huyết áp"}],
        },
        expected_message="tôi chưa biết huyết áp",
    )

    result = extractor.extract(
        new_message="tôi chưa biết huyết áp",
        current_profile={},
        recent_messages=[],
    )

    assert result.explicit_unknown_fields == ["hypertension"]
