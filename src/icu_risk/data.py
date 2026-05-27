from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from icu_risk.config import ID_COLUMN, MISSING_VALUE_COLUMNS, RAW_DATA_PATH


@dataclass
class ValidationReport:
    dropped_duplicates: int
    invalid_bmi_rows: int
    invalid_spo2_rows: int
    invalid_hr_rows: int
    invalid_map_rows: int
    invalid_potassium_rows: int

    def to_dict(self) -> dict[str, int]:
        return {
            "dropped_duplicates": self.dropped_duplicates,
            "invalid_bmi_rows": self.invalid_bmi_rows,
            "invalid_spo2_rows": self.invalid_spo2_rows,
            "invalid_hr_rows": self.invalid_hr_rows,
            "invalid_map_rows": self.invalid_map_rows,
            "invalid_potassium_rows": self.invalid_potassium_rows,
        }


ICU_UNIT_LOOKUP = {
    "medical icu": "Medical ICU",
    "medical icu ": "Medical ICU",
    "medical icu unit": "Medical ICU",
    "micu": "Medical ICU",
    "medical icu.": "Medical ICU",
    "surgical icu": "Surgical ICU",
    "sicu": "Surgical ICU",
    "cardiac icu": "Cardiac ICU",
    "cicu": "Cardiac ICU",
    "neuro icu": "Neuro ICU",
    "nicu": "Neuro ICU",
    "trauma icu": "Trauma ICU",
    "ticu": "Trauma ICU",
}


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    csv_path = RAW_DATA_PATH if path is None else path
    return pd.read_csv(csv_path)


def canonicalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df["icu_unit"] = (
        clean_df["icu_unit"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: ICU_UNIT_LOOKUP.get(value, value.title()))
    )
    clean_df["admission_type"] = clean_df["admission_type"].astype(str).str.strip().str.title()
    clean_df["gender"] = clean_df["gender"].astype(str).str.strip().str.title()
    return clean_df


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    for column in MISSING_VALUE_COLUMNS:
        flagged[f"{column}_missing"] = flagged[column].isna().astype(int)
    return flagged


def impute_clinical_missingness(df: pd.DataFrame) -> pd.DataFrame:
    imputed = add_missingness_flags(df)

    imputed["gcs_score"] = imputed["gcs_score"].fillna(3.0)
    imputed["lactate"] = imputed.groupby("admission_type")["lactate"].transform(
        lambda series: series.fillna(series.median())
    )
    imputed["pao2_fio2_ratio"] = imputed["pao2_fio2_ratio"].fillna(400.0)
    imputed["bilirubin"] = (
        imputed.groupby(["has_heart_failure", "has_ckd"])["bilirubin"]
        .transform(lambda series: series.fillna(series.median()))
        .fillna(imputed["bilirubin"].median())
    )
    imputed["fluid_balance_24h"] = imputed["fluid_balance_24h"].fillna(0.0)
    return imputed


def run_clinical_validations(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    clean_df = df.drop_duplicates().copy()
    dropped_duplicates = len(df) - len(clean_df)

    bmi_check = clean_df["weight_kg"] / (clean_df["height_cm"] / 100.0) ** 2
    bmi_mismatch = (clean_df["bmi"] - bmi_check).abs() > 2
    clean_df.loc[bmi_mismatch, "bmi"] = bmi_check[bmi_mismatch].round(1)

    spo2_invalid = clean_df["spo2_min"] > clean_df["spo2_mean"]
    clean_df.loc[spo2_invalid, "spo2_min"] = clean_df.loc[spo2_invalid, "spo2_mean"]

    hr_invalid = clean_df["hr_min"] > clean_df["hr_mean"]
    clean_df.loc[hr_invalid, "hr_min"] = clean_df.loc[hr_invalid, "hr_mean"]

    map_invalid = (clean_df["map_mean"] > clean_df["sbp_mean"]) | (
        clean_df["map_mean"] < clean_df["sbp_min"] * 0.5
    )
    clean_df.loc[map_invalid, "map_mean"] = (
        (clean_df.loc[map_invalid, "sbp_mean"] + clean_df.loc[map_invalid, "sbp_min"]) / 2.0
    )

    potassium_invalid = (clean_df["potassium"] < 2.0) | (clean_df["potassium"] > 7.0)
    clean_df.loc[potassium_invalid, "potassium"] = np.nan
    clean_df["potassium"] = clean_df["potassium"].fillna(clean_df["potassium"].median())

    report = ValidationReport(
        dropped_duplicates=dropped_duplicates,
        invalid_bmi_rows=int(bmi_mismatch.sum()),
        invalid_spo2_rows=int(spo2_invalid.sum()),
        invalid_hr_rows=int(hr_invalid.sum()),
        invalid_map_rows=int(map_invalid.sum()),
        invalid_potassium_rows=int(potassium_invalid.sum()),
    )
    return clean_df, report


def build_modeling_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    canonical = canonicalize_categories(df)
    validated, report = run_clinical_validations(canonical)
    enriched = impute_clinical_missingness(validated)
    modeling_df = enriched.drop(columns=[ID_COLUMN]).copy()
    return modeling_df, report

