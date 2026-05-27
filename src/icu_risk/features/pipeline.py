from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from icu_risk.config import CATEGORICAL_COLUMNS, TARGET_COLUMN


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["map_below_65"] = (features["map_min"] < 65).astype(int)
    features["spo2_below_90"] = (features["spo2_min"] < 90).astype(int)
    features["rr_above_25"] = (features["rr_mean"] > 25).astype(int)
    features["hr_above_130"] = (features["hr_max"] > 130).astype(int)
    features["lactate_elevated"] = (features["lactate"] > 2.0).astype(int)
    features["lactate_critical"] = (features["lactate"] > 4.0).astype(int)
    features["temp_fever"] = (features["temp_max"] > 38.3).astype(int)
    features["temp_hypothermia"] = (features["temp_mean"] < 36.0).astype(int)
    features["gcs_severe"] = (features["gcs_score"] < 9).astype(int)
    features["pf_ratio_ards"] = (features["pao2_fio2_ratio"] < 200).astype(int)
    features["creatinine_aki"] = (features["creatinine"] > 2.0).astype(int)
    features["platelets_low"] = (features["platelet_count"] < 100).astype(int)

    features["hr_range"] = features["hr_max"] - features["hr_min"]
    features["sbp_range"] = features["sbp_max"] - features["sbp_min"]
    features["temp_range"] = features["temp_max"] - features["temp_mean"]
    features["shock_index"] = features["hr_mean"] / features["sbp_mean"].clip(lower=1)
    features["fluid_overload_flag"] = (
        (features["fluid_balance_24h"] > 2000) & (features["on_mechanical_vent"] == 1)
    ).astype(int)

    organ_flags = [
        features["map_min"] < 65,
        features["pao2_fio2_ratio"] < 200,
        features["creatinine"] > 2.0,
        features["bilirubin"] > 2.0,
        features["platelet_count"] < 100,
        features["gcs_score"] < 9,
    ]
    features["organ_failure_count"] = np.column_stack(organ_flags).astype(int).sum(axis=1)
    features["vasopress_x_aki"] = features["on_vasopressor"] * features["creatinine_aki"]

    for column in ["lactate", "creatinine", "bilirubin", "wbc"]:
        numeric_series = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
        features[column] = numeric_series
        features[f"log_{column}"] = np.log1p(numeric_series.clip(lower=0))

    return features


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in feature_frame.columns]
    numerical_columns = [column for column in feature_frame.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", Pipeline([("scaler", StandardScaler())]), numerical_columns),
        ],
        remainder="drop",
    )
