from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.prediction.preprocess import build_feature_frame, missing_screening_fields


DISCLAIMER = "Kết quả này chỉ dùng để sàng lọc nguy cơ, không phải chẩn đoán y khoa."


@dataclass
class PredictionArtifacts:
    metadata: dict[str, Any]
    clinical_model: Any | None
    calories_model: Any | None
    model_status: str


class PredictionService:
    def __init__(self) -> None:
        self._artifacts: PredictionArtifacts | None = None

    def ready_status(self) -> dict[str, Any]:
        artifacts = self._load()
        return {
            "model_status": artifacts.model_status,
            "clinical_model_path": str(settings.clinical_model_path),
            "metadata_path": str(settings.preprocess_metadata_path),
            "threshold": artifacts.metadata.get("decision_threshold_clinical"),
        }

    def predict(self, profile: dict[str, Any]) -> dict[str, Any]:
        artifacts = self._load()
        frame, detected, filled = build_feature_frame(profile, artifacts.metadata, artifacts.calories_model)
        estimated_calories = float(frame.iloc[0].to_dict().get("Estimated_calories", 0.0))
        threshold = float(artifacts.metadata.get("decision_threshold_clinical", 0.03))

        try:
            if artifacts.clinical_model is None:
                raise RuntimeError("clinical model unavailable")
            probability = float(artifacts.clinical_model.predict_proba(frame)[0][1])
            model_status = artifacts.model_status
        except Exception:
            probability = self._fallback_probability(frame.iloc[0].to_dict())
            model_status = "fallback"

        return {
            "model_used": "stroke_model_clinical.joblib",
            "model_status": model_status,
            "model_version": "local-artifacts-2026-07-19",
            "risk_probability": round(max(0.0, min(1.0, probability)), 6),
            "risk_label": self._risk_label(probability, threshold),
            "threshold": threshold,
            "detected_features": detected,
            "filled_features": filled,
            "estimated_calories": estimated_calories,
            "missing_user_fields": missing_screening_fields(profile),
            "disclaimer": DISCLAIMER,
        }

    def _load(self) -> PredictionArtifacts:
        if self._artifacts is not None:
            return self._artifacts

        metadata = self._default_metadata()
        clinical_model = None
        calories_model = None
        model_status = "loaded"
        try:
            metadata = json.loads(settings.preprocess_metadata_path.read_text(encoding="utf-8"))
            from joblib import load

            clinical_model = load(settings.clinical_model_path)
            try:
                calories_model = load(settings.calories_model_path)
            except Exception:
                calories_model = None
        except Exception:
            model_status = "fallback"

        self._artifacts = PredictionArtifacts(
            metadata=metadata,
            clinical_model=clinical_model,
            calories_model=calories_model,
            model_status=model_status,
        )
        return self._artifacts

    @staticmethod
    def _default_metadata() -> dict[str, Any]:
        numeric_fill = {
            "gender": 0.0,
            "age": 0.0,
            "hypertension": 0.0,
            "heart_disease": 0.0,
            "avg_glucose_level": 0.0,
            "bmi": 0.0,
            "avg_RestingBP": 0.0,
            "avg_MaxHR": 0.0,
            "avg_Oldpeak": 0.0,
            "heart_disease_rate": 0.0,
            "cardio_cholesterol": 0.0,
            "cardio_gluc": 0.0,
            "cardio_rate": 0.0,
            "smoke_flag": 0.0,
            "alco_flag": 0.0,
            "Estimated_calories": 0.0,
        }
        categorical_fill = {
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "unknown",
        }
        selected_features = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
            "avg_RestingBP",
            "avg_MaxHR",
            "avg_Oldpeak",
            "heart_disease_rate",
            "cardio_cholesterol",
            "cardio_gluc",
            "cardio_rate",
            "smoke_flag",
            "alco_flag",
            "Estimated_calories",
            "ever_married_Yes",
            "work_type_Never_worked",
            "work_type_Private",
            "work_type_Self-employed",
            "work_type_children",
            "Residence_type_Urban",
            "smoking_status_never smoked",
            "smoking_status_smokes",
            "smoking_status_unknown",
        ]
        return {
            "numeric_fill": numeric_fill,
            "categorical_fill": categorical_fill,
            "selected_features": selected_features,
            "decision_threshold_clinical": 0.03,
        }

    @staticmethod
    def _risk_label(probability: float, threshold: float) -> str:
        if probability >= threshold:
            return "high_risk"
        if probability >= threshold * 0.5:
            return "medium_risk"
        return "low_risk"

    @staticmethod
    def _fallback_probability(features: dict[str, Any]) -> float:
        score = 0.01
        score += max(0.0, (float(features.get("age", 45)) - 45.0) / 1000.0)
        score += 0.015 if float(features.get("hypertension", 0)) >= 1 else 0.0
        score += 0.018 if float(features.get("heart_disease", 0)) >= 1 else 0.0
        score += max(0.0, (float(features.get("avg_glucose_level", 91)) - 120.0) / 5000.0)
        score += max(0.0, (float(features.get("bmi", 28)) - 28.0) / 2000.0)
        return score


prediction_service = PredictionService()
