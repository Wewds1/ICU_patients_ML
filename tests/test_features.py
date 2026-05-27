import pandas as pd

from icu_risk.data import canonicalize_categories, impute_clinical_missingness
from icu_risk.features.pipeline import engineer_features


def test_feature_engineering_adds_expected_columns():
    frame = pd.DataFrame(
        [
            {
                "gender": "female",
                "icu_unit": "micu",
                "admission_type": "emergency",
                "map_min": 60,
                "spo2_min": 88,
                "rr_mean": 28,
                "hr_max": 135,
                "temp_max": 38.5,
                "temp_mean": 37.8,
                "gcs_score": None,
                "lactate": None,
                "pao2_fio2_ratio": None,
                "bilirubin": None,
                "fluid_balance_24h": None,
                "on_mechanical_vent": 1,
                "creatinine": 2.4,
                "platelet_count": 90,
                "on_vasopressor": 1,
                "wbc": 12.0,
                "hr_min": 80,
                "sbp_max": 130,
                "sbp_min": 90,
                "hr_mean": 100,
                "sbp_mean": 100,
                "has_heart_failure": 0,
                "has_ckd": 0,
            }
        ]
    )

    frame = canonicalize_categories(frame)
    frame = impute_clinical_missingness(frame)
    engineered = engineer_features(frame)

    assert engineered.loc[0, "icu_unit"] == "Medical ICU"
    assert engineered.loc[0, "gcs_score_missing"] == 1
    assert engineered.loc[0, "map_below_65"] == 1
    assert engineered.loc[0, "fluid_overload_flag"] == 0
    assert "shock_index" in engineered.columns

