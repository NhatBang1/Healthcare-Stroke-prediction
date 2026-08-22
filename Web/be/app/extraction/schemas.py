from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ScreeningField = Literal[
    "age",
    "gender",
    "systolic_bp",
    "diastolic_bp",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "pregnant",
    "smoke_flag",
    "alco_flag",
]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: ScreeningField
    quote: str = Field(min_length=1, max_length=240)


class ScreeningProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float | None = Field(default=None, ge=0, le=100)
    gender: Literal["male", "female"] | None = None
    systolic_bp: float | None = Field(default=None, ge=50, le=300)
    diastolic_bp: float | None = Field(default=None, ge=30, le=200)
    hypertension_history: bool | None = None
    heart_disease: bool | None = None
    avg_glucose_level: float | None = Field(default=None, ge=40, le=400)
    bmi: float | None = Field(default=None, ge=10, le=80)
    pregnant: bool | None = None
    smoke_flag: bool | None = None
    alco_flag: bool | None = None
    explicit_unknown_fields: list[ScreeningField] = Field(default_factory=list)
    # DeepSeek may describe a correction in natural language (for example,
    # "age corrected from 20 to 65"). Keep the audit trail descriptive rather
    # than constraining it to a field-name enum.
    corrections: list[str] = Field(default_factory=list, max_length=20)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    def profile_updates(self) -> dict[str, object]:
        values = self.model_dump(
            include={
                "age",
                "gender",
                "systolic_bp",
                "diastolic_bp",
                "heart_disease",
                "avg_glucose_level",
                "bmi",
                "pregnant",
                "smoke_flag",
                "alco_flag",
            },
            exclude_none=True,
        )
        if self.hypertension_history is not None:
            values["hypertension"] = self.hypertension_history
        return values

    def evidence_by_field(self) -> dict[str, dict[str, str]]:
        return {
            item.field: {"quote": item.quote}
            for item in self.evidence
        }
