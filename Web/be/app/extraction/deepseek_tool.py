from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import httpx
from pydantic import ValidationError

from app.extraction.schemas import ScreeningProfileUpdate


logger = logging.getLogger(__name__)


EXTRACT_SCREENING_PROFILE_TOOL: dict[str, Any] = {
    "name": "extract_screening_profile",
    "description": (
        "Extract only health screening facts explicitly stated or corrected "
        "in the newest Vietnamese or English user message. Never guess."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "age": {"type": ["number", "null"]},
            "gender": {"type": ["string", "null"], "enum": ["male", "female", None]},
            "systolic_bp": {"type": ["number", "null"]},
            "diastolic_bp": {"type": ["number", "null"]},
            "hypertension_history": {"type": ["boolean", "null"]},
            "heart_disease": {"type": ["boolean", "null"]},
            "avg_glucose_level": {"type": ["number", "null"]},
            "bmi": {"type": ["number", "null"]},
            "pregnant": {"type": ["boolean", "null"]},
            "smoke_flag": {
                "type": ["boolean", "null"],
                "description": "Whether the user currently smokes or has a smoking history, only when explicitly stated.",
            },
            "alco_flag": {
                "type": ["boolean", "null"],
                "description": "Whether the user drinks alcohol, only when explicitly stated.",
            },
            "explicit_unknown_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
            "corrections": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "field": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["field", "quote"],
                },
            },
        },
        "required": [
            "age",
            "gender",
            "systolic_bp",
            "diastolic_bp",
            "hypertension_history",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
            "pregnant",
            "smoke_flag",
            "alco_flag",
            "explicit_unknown_fields",
            "corrections",
            "evidence",
        ],
    },
}


class ExtractionUnavailable(RuntimeError):
    pass


class DeepSeekProfileExtractor:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.transport = transport

    def configured(self) -> bool:
        return bool(self.api_key)

    def extract(
        self,
        *,
        new_message: str,
        current_profile: dict[str, Any],
        recent_messages: list[dict[str, str]],
    ) -> ScreeningProfileUpdate:
        if not self.api_key:
            raise ExtractionUnavailable("DeepSeek extraction is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract stroke-screening profile facts. Return facts "
                        "only through the required tool. Use null for facts absent "
                        "from the newest message. Preserve explicit false values. "
                        "First distinguish intent: a question asking whether smoking, "
                        "alcohol, heart disease, or another factor affects stroke risk is "
                        "general education, not a new screening measurement. For such a "
                        "question return null values, empty explicit_unknown_fields, and "
                        "empty evidence; never turn the question into a prediction. "
                        "Only extract a factor when the user states it as their own fact "
                        "(for example: 'tôi hút thuốc', 'tôi có suy tim'). "
                        "Put unknown facts in explicit_unknown_fields. Every extracted "
                        "fact must include a verbatim evidence quote from the newest message."
                    ),
                },
                {
                    "role": "user",
                    "content": self._user_prompt(
                        new_message=new_message,
                        current_profile=current_profile,
                        recent_messages=recent_messages,
                    ),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": EXTRACT_SCREENING_PROFILE_TOOL,
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "extract_screening_profile"},
            },
            "temperature": 0,
            "stream": False,
            "thinking": {"type": "disabled"},
        }

        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    arguments = self._tool_arguments(response.json())
                    arguments = self._normalize_null_sentinels(arguments)
                    result = ScreeningProfileUpdate.model_validate(arguments)
            except (
                ExtractionUnavailable,
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc
                if attempt < 2:
                    logger.warning(
                        "DeepSeek profile extraction attempt %d failed (%s); retrying once",
                        attempt,
                        type(exc).__name__,
                    )
                    continue
                logger.error(
                    "DeepSeek profile extraction failed after %d attempts (%s)",
                    attempt,
                    type(exc).__name__,
                )
                continue
            try:
                self._validate_evidence(result, new_message)
            except ValueError as exc:
                logger.warning(
                    "DeepSeek profile evidence validation failed (%s)",
                    type(exc).__name__,
                )
                raise
            return result
        raise ExtractionUnavailable(
            f"DeepSeek extraction failed: {type(last_error).__name__ if last_error else 'unknown'}"
        ) from last_error

    @staticmethod
    def _user_prompt(
        *,
        new_message: str,
        current_profile: dict[str, Any],
        recent_messages: list[dict[str, str]],
    ) -> str:
        context = recent_messages[-6:]
        return (
            "Current accepted profile JSON:\n"
            f"{json.dumps(current_profile, ensure_ascii=False, default=str)}\n"
            "Recent conversation JSON:\n"
            f"{json.dumps(context, ensure_ascii=False)}\n"
            "Newest user message (extract only from this text):\n"
            f"{new_message}"
        )

    @staticmethod
    def _tool_arguments(data: dict[str, Any]) -> dict[str, Any]:
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        for call in tool_calls:
            function = call.get("function") or {}
            if function.get("name") != "extract_screening_profile":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
            else:
                parsed = arguments
            if not isinstance(parsed, dict):
                raise ExtractionUnavailable("DeepSeek returned invalid tool arguments")
            return parsed
        raise ExtractionUnavailable("DeepSeek did not call extract_screening_profile")

    @staticmethod
    def _normalize_null_sentinels(arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        nullable_scalar_fields = {
            "age",
            "gender",
            "systolic_bp",
            "diastolic_bp",
            "hypertension_history",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
            "pregnant",
        }
        for field in nullable_scalar_fields:
            value = normalized.get(field)
            if isinstance(value, str) and value.strip().casefold() in {"null", "none"}:
                normalized[field] = None
        # The prompt/tool exposes the model-facing name
        # ``hypertension_history`` while the merged profile and evidence
        # schema use the shorter canonical name ``hypertension``. DeepSeek
        # may echo either spelling in explicit_unknown_fields; normalize it
        # before Pydantic validation so an otherwise valid extraction is not
        # discarded.
        unknown_fields = normalized.get("explicit_unknown_fields")
        if isinstance(unknown_fields, list):
            normalized["explicit_unknown_fields"] = [
                "hypertension" if item == "hypertension_history" else item
                for item in unknown_fields
            ]
        return normalized

    @classmethod
    def _validate_evidence(
        cls,
        result: ScreeningProfileUpdate,
        new_message: str,
    ) -> None:
        normalized_message = cls._normalize_text(new_message)
        evidenced_fields = {item.field for item in result.evidence}
        # Evidence is mandatory for values that will be merged into the
        # accepted profile. ``explicit_unknown_fields`` are intentionally
        # non-values; DeepSeek may list a field as unknown without producing
        # a quote (for example, pregnancy or alcohol use was not mentioned),
        # and that must not discard otherwise valid screening measurements.
        required_fields = set(result.profile_updates())
        missing_evidence = required_fields - evidenced_fields
        if missing_evidence:
            raise ValueError(f"Missing evidence for: {sorted(missing_evidence)}")
        evidence_by_field = {item.field: item.quote for item in result.evidence}
        numeric_values = {
            "age": result.age,
            "systolic_bp": result.systolic_bp,
            "diastolic_bp": result.diastolic_bp,
            "avg_glucose_level": result.avg_glucose_level,
            "bmi": result.bmi,
        }
        for field, value in numeric_values.items():
            if value is None:
                continue
            quote_numbers = [
                float(item.replace(",", "."))
                for item in re.findall(r"\d+(?:[.,]\d+)?", evidence_by_field[field])
            ]
            if not any(math.isclose(float(value), item, rel_tol=0, abs_tol=0.01) for item in quote_numbers):
                raise ValueError(f"Evidence for {field} does not support value {value}")
        if result.gender is not None:
            gender_quote = cls._normalize_text(evidence_by_field["gender"])
            allowed = ("nam", "male", "man") if result.gender == "male" else ("nữ", "nu", "female", "woman")
            if not any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", gender_quote) for term in allowed):
                raise ValueError(f"Evidence for gender does not support value {result.gender}")
        boolean_values = {
            "hypertension": result.hypertension_history,
            "heart_disease": result.heart_disease,
            "pregnant": result.pregnant,
            "smoke_flag": result.smoke_flag,
            "alco_flag": result.alco_flag,
        }
        for field, value in boolean_values.items():
            if value is None:
                continue
            quote = cls._normalize_text(evidence_by_field[field])
            if any(
                phrase in quote
                for phrase in (
                    "không biết",
                    "chưa biết",
                    "không rõ",
                    "không chắc",
                    "chưa chắc",
                    "khó nói",
                    "not sure",
                    "unsure",
                    "uncertain",
                    "don't know",
                    "do not know",
                )
            ):
                raise ValueError(f"Evidence for {field} expresses uncertainty")
            is_negated = bool(re.search(r"(?<!\w)(?:không|chưa|chẳng|no|not)(?!\w)", quote))
            if value is False and not is_negated:
                raise ValueError(f"Evidence for {field} does not support false")
            if value is True and is_negated:
                raise ValueError(f"Evidence for {field} does not support true")
        for item in result.evidence:
            if cls._normalize_text(item.quote) not in normalized_message:
                raise ValueError(f"Invalid evidence for {item.field}")

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", value.casefold()).strip()
