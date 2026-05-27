# Source Package Notes

The `src` directory contains the reusable application and machine learning code.

## Layout

- `icu_risk/config.py` centralizes file-system paths and shared constants
- `icu_risk/data.py` handles loading, category cleanup, missingness treatment, and clinical validation
- `icu_risk/features/pipeline.py` creates engineered features and preprocessing objects
- `icu_risk/training/train.py` runs training, evaluation, artifact persistence, and figure generation
- `icu_risk/api/` contains the FastAPI application and request/response schemas

## Design intent

The package is separated so the notebooks, scripts, tests, and API all reuse the same logic instead of duplicating preprocessing code in different places. That reduces drift between training and inference.

