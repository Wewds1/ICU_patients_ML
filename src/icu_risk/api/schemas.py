from __future__ import annotations

from pydantic import BaseModel, Field


class PatientObservation(BaseModel):
    patient_age: int = Field(..., ge=18, le=100)
    gender: str
    weight_kg: float
    height_cm: float
    bmi: float
    icu_unit: str
    admission_type: str
    icu_los_days_at_obs: float
    has_diabetes: int
    has_hypertension: int
    has_heart_failure: int
    has_copd: int
    has_ckd: int
    has_immunosuppression: int
    charlson_index: int
    hr_mean: float
    hr_min: float
    hr_max: float
    hr_std: float
    sbp_mean: float
    sbp_min: float
    sbp_max: float
    map_mean: float
    map_min: float
    rr_mean: float
    rr_max: float
    temp_mean: float
    temp_max: float
    spo2_mean: float
    spo2_min: float
    gcs_score: float | None = None
    sofa_score: int
    news2_score: int
    lactate: float | None = None
    wbc: float
    creatinine: float
    bilirubin: float | None = None
    platelet_count: int
    sodium: float
    potassium: float
    hemoglobin: float
    pao2_fio2_ratio: float | None = None
    on_vasopressor: int
    on_mechanical_vent: int
    on_rrt: int
    fluid_balance_24h: float | None = None


class PredictionResponse(BaseModel):
    deterioration_probability: float
    alert_triggered: bool
    alert_threshold: float
    risk_level: str
    model_version: str

