from __future__ import annotations

from typing import Any


REQUIRED_SCREENING_FIELDS = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]


class _SimpleRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class _SimpleILoc:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, index: int) -> _SimpleRow:
        if index != 0:
            raise IndexError(index)
        return _SimpleRow(self._data)


class SimpleFeatureFrame:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)
        self.iloc = _SimpleILoc(self._data)


def normalize_gender(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"male", "nam", "m", "man"}:
        return 0.0
    if normalized in {"female", "nữ", "nu", "f", "woman"}:
        return 1.0
    return None


def normalize_binary(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 1 else 0.0
    normalized = str(value).strip().lower()
    if normalized in {"yes", "có", "co", "true", "1"}:
        return 1.0
    if normalized in {"no", "không", "khong", "false", "0"}:
        return 0.0
    return None


def clean_bmi(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 10 or numeric > 80:
        return None
    return numeric


def build_feature_frame(profile: dict[str, Any], metadata: dict[str, Any], calories_model: Any = None):
    numeric_fill = metadata["numeric_fill"]
    categorical_fill = metadata["categorical_fill"]
    selected_features = metadata["selected_features"]
    detected: dict[str, Any] = {}
    filled: dict[str, Any] = {}

    normalized: dict[str, Any] = dict(profile)
    normalized["gender"] = normalize_gender(profile.get("gender"))
    normalized["hypertension"] = normalize_binary(profile.get("hypertension"))
    normalized["heart_disease"] = normalize_binary(profile.get("heart_disease"))
    normalized["smoke_flag"] = normalize_binary(profile.get("smoke_flag"))
    normalized["alco_flag"] = normalize_binary(profile.get("alco_flag"))
    normalized["bmi"] = clean_bmi(profile.get("bmi"))

    for key, value in normalized.items():
        if value is not None:
            detected[key] = value

    row: dict[str, Any] = {}
    numeric_columns = [
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
    ]
    for column in numeric_columns:
        value = normalized.get(column)
        if value is None:
            value = numeric_fill.get(column, 0.0)
            filled[column] = value
        row[column] = float(value)

    categorical_values: dict[str, str] = {}
    for column in ["ever_married", "work_type", "Residence_type", "smoking_status"]:
        value = normalized.get(column)
        if value is None or str(value).strip() == "":
            value = categorical_fill[column]
            filled[column] = value
        categorical_values[column] = str(value)

    row["Estimated_calories"] = _estimate_calories(row, calories_model)
    if calories_model is None:
        filled["Estimated_calories"] = row["Estimated_calories"]

    feature_row = {feature: 0.0 for feature in selected_features}
    for key, value in row.items():
        if key in feature_row:
            feature_row[key] = value

    _apply_one_hot(feature_row, categorical_values)
    ordered_row = {feature: float(feature_row.get(feature, 0.0)) for feature in selected_features}
    try:
        import pandas as pd

        # The clinical stacking artifact is scikit-learn based and requires a
        # real array/DataFrame at inference time. Keep the tiny fallback frame
        # only for environments where pandas is unavailable.
        frame = pd.DataFrame([ordered_row], columns=selected_features)
    except ImportError:
        frame = SimpleFeatureFrame(ordered_row)
    return frame, detected, filled


def missing_screening_fields(profile: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_SCREENING_FIELDS:
        value = profile.get(field)
        if field == "gender":
            value = normalize_gender(value)
        if field in {"hypertension", "heart_disease"}:
            value = normalize_binary(value)
        if field == "bmi":
            value = clean_bmi(value)
        if value is None:
            missing.append(field)
    return missing


def _estimate_calories(row: dict[str, float], calories_model: Any = None) -> float:
    if calories_model is not None:
        try:
            import pandas as pd

            features = pd.DataFrame(
                [{"Gender": row["gender"], "Age": row["age"], "BMI": row["bmi"]}],
                columns=["Gender", "Age", "BMI"],
            )
            return float(max(0.0, float(calories_model.predict(features)[0])))
        except Exception:
            pass
    return float(max(40.0, 45.0 + row["age"] * 0.55 + row["bmi"] * 1.3))


def _apply_one_hot(feature_row: dict[str, float], categorical_values: dict[str, str]) -> None:
    if categorical_values["ever_married"].strip().lower() == "yes":
        feature_row["ever_married_Yes"] = 1.0

    work_type = categorical_values["work_type"].strip().lower()
    work_map = {
        "never_worked": "work_type_Never_worked",
        "never worked": "work_type_Never_worked",
        "private": "work_type_Private",
        "self-employed": "work_type_Self-employed",
        "self employed": "work_type_Self-employed",
        "children": "work_type_children",
    }
    if work_type in work_map:
        feature_row[work_map[work_type]] = 1.0

    if categorical_values["Residence_type"].strip().lower() == "urban":
        feature_row["Residence_type_Urban"] = 1.0

    smoking_status = categorical_values["smoking_status"].strip().lower()
    smoke_map = {
        "never smoked": "smoking_status_never smoked",
        "smokes": "smoking_status_smokes",
        "unknown": "smoking_status_unknown",
    }
    if smoking_status in smoke_map:
        feature_row[smoke_map[smoking_status]] = 1.0
