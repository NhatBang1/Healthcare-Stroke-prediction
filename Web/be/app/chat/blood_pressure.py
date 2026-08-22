from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BloodPressureClassification:
    category: str
    hypertension: bool | None


def classify_blood_pressure(
    age: float | None,
    systolic: float | None,
    diastolic: float | None,
    *,
    pregnant: bool = False,
) -> BloodPressureClassification:
    """Classify complete adult, non-pregnancy readings using AHA thresholds."""
    if age is None or age < 18 or systolic is None or diastolic is None or pregnant:
        return BloodPressureClassification("unknown", None)
    if systolic > 180 or diastolic > 120:
        return BloodPressureClassification("severe", True)
    if systolic >= 140 or diastolic >= 90:
        return BloodPressureClassification("stage_2", True)
    if systolic >= 130 or diastolic >= 80:
        return BloodPressureClassification("stage_1", True)
    if systolic >= 120 and diastolic < 80:
        return BloodPressureClassification("elevated", False)
    return BloodPressureClassification("normal", False)
