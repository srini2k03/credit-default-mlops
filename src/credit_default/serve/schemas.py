"""Request/response models. Field names match the UCI columns so payloads map 1:1 to the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_PAY = dict(
    ge=-2,
    le=8,
    description="Repayment status: -2 no use, -1 paid duly, 0 revolving, 1-8 months late",
)
_BILL = dict(
    ge=-1_000_000,
    le=2_000_000,
    description="Bill statement amount (NT$); negative = credit balance",
)
_PAYAMT = dict(ge=0, le=2_000_000, description="Amount paid (NT$)")

EXAMPLE = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 46990,
    "BILL_AMT2": 48233,
    "BILL_AMT3": 49291,
    "BILL_AMT4": 28314,
    "BILL_AMT5": 28959,
    "BILL_AMT6": 29547,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 2019,
    "PAY_AMT3": 1200,
    "PAY_AMT4": 1100,
    "PAY_AMT5": 1069,
    "PAY_AMT6": 1000,
}


class CreditApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    LIMIT_BAL: int = Field(ge=1_000, le=5_000_000, description="Credit limit (NT$)")
    SEX: Literal[1, 2] = Field(description="1 male, 2 female")
    EDUCATION: int = Field(
        ge=0,
        le=6,
        description="1 graduate, 2 university, 3 high school, 4 others (0/5/6 unlabelled)",
    )
    MARRIAGE: int = Field(ge=0, le=3, description="1 married, 2 single, 3 others (0 unlabelled)")
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(**_PAY)
    PAY_2: int = Field(**_PAY)
    PAY_3: int = Field(**_PAY)
    PAY_4: int = Field(**_PAY)
    PAY_5: int = Field(**_PAY)
    PAY_6: int = Field(**_PAY)
    BILL_AMT1: int = Field(**_BILL)
    BILL_AMT2: int = Field(**_BILL)
    BILL_AMT3: int = Field(**_BILL)
    BILL_AMT4: int = Field(**_BILL)
    BILL_AMT5: int = Field(**_BILL)
    BILL_AMT6: int = Field(**_BILL)
    PAY_AMT1: int = Field(**_PAYAMT)
    PAY_AMT2: int = Field(**_PAYAMT)
    PAY_AMT3: int = Field(**_PAYAMT)
    PAY_AMT4: int = Field(**_PAYAMT)
    PAY_AMT5: int = Field(**_PAYAMT)
    PAY_AMT6: int = Field(**_PAYAMT)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[CreditApplication] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    request_id: str
    probability_of_default: float = Field(description="Isotonic-calibrated P(default next month)")
    default_prediction: bool
    decision: Literal["approve", "decline"]
    threshold: float = Field(description="Cost-optimal threshold chosen on validation data")
    model_version: str
    model_source: Literal["registry", "local"]


class BatchResponse(BaseModel):
    n: int
    predictions: list[Prediction]


class Contribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class Explanation(BaseModel):
    request_id: str
    probability_of_default: float
    default_prediction: bool
    base_value: float
    contributions: list[Contribution]
    explanation_space: str = (
        "SHAP values in the base model's log-odds (before isotonic calibration)"
    )
    model_version: str


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_source: str | None = None
    model_version: str | None = None
    model_family: str | None = None
    threshold: float | None = None
    detail: str | None = None
