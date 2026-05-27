from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "icu_patients.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
TARGET_COLUMN = "deterioration_12h"
ID_COLUMN = "patient_id"

MISSING_VALUE_COLUMNS = [
    "gcs_score",
    "lactate",
    "pao2_fio2_ratio",
    "bilirubin",
    "fluid_balance_24h",
]

CATEGORICAL_COLUMNS = ["gender", "icu_unit", "admission_type"]

CORE_API_FIELDS = [
    "patient_age",
    "gender",
    "weight_kg",
    "height_cm",
    "bmi",
    "icu_unit",
    "admission_type",
    "icu_los_days_at_obs",
    "has_diabetes",
    "has_hypertension",
    "has_heart_failure",
    "has_copd",
    "has_ckd",
    "has_immunosuppression",
    "charlson_index",
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "sbp_mean",
    "sbp_min",
    "sbp_max",
    "map_mean",
    "map_min",
    "rr_mean",
    "rr_max",
    "temp_mean",
    "temp_max",
    "spo2_mean",
    "spo2_min",
    "gcs_score",
    "sofa_score",
    "news2_score",
    "lactate",
    "wbc",
    "creatinine",
    "bilirubin",
    "platelet_count",
    "sodium",
    "potassium",
    "hemoglobin",
    "pao2_fio2_ratio",
    "on_vasopressor",
    "on_mechanical_vent",
    "on_rrt",
    "fluid_balance_24h",
]

