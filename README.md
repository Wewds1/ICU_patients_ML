# ICU Patient Deterioration Early Warning

This repository is a complete end-to-end machine learning project built around a synthetic ICU deterioration dataset. It includes data preparation, clinical validation, feature engineering, model training, saved artifacts, evaluation figures, documented notebooks, a FastAPI backend, a simple black-and-white web UI, Docker deployment support, and tests.

## What is included

### UI

The frontend uses plain HTML, CSS, and JavaScript. It is intentionally minimal, monochrome, and easy to deploy. The interface allows a user to submit one ICU observation and receive:

- deterioration probability
- alert status
- deployed threshold
- risk level
- model version

Files:

- [templates/index.html]
- [static/css/styles.css]
- [static/js/app.js]

### Machine Learning workflow


The ML pipeline follows a clinically motivated structure:

1. load the raw ICU CSV
2. drop duplicate observations
3. canonicalize messy categorical values
4. run clinical plausibility checks
5. create missingness flags before imputation
6. engineer threshold-based and composite features
7. train logistic regression, random forest, and XGBoost
8. compare them against SOFA and NEWS2 baselines
9. calibrate the selected deployment model
10. save figures, processed data, and serialized artifacts

Core code:

- [src/icu_risk/data.py](/D:/Projects/ICU_Patients/src/icu_risk/data.py)
- [src/icu_risk/features/pipeline.py](/D:/Projects/ICU_Patients/src/icu_risk/features/pipeline.py)
- [src/icu_risk/training/train.py](/D:/Projects/ICU_Patients/src/icu_risk/training/train.py)
- [scripts/train_pipeline.py](/D:/Projects/ICU_Patients/scripts/train_pipeline.py)

### Dataset

The raw dataset lives at [data/raw/icu_patients.csv](/D:/Projects/ICU_Patients/data/raw/icu_patients.csv). It contains 8,530 rows and 48 original columns. The pipeline identified and removed 30 duplicate rows, then exported a model-ready dataset to [data/processed/icu_patients_model_ready.csv](/D:/Projects/ICU_Patients/data/processed/icu_patients_model_ready.csv).

Important data characteristics:

- target: `deterioration_12h`
- duplicated rows removed: `30`
- major missing columns: `gcs_score`, `lactate`, `bilirubin`, `pao2_fio2_ratio`, `fluid_balance_24h`
- validation issue fixed most often: implausible `map_mean` values

### Figures

The training run exports figures to [figures]:

- `target_distribution.png`
- `clinical_score_boxplots.png`
- `vital_distributions.png`
- `model_curves.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `calibration_curve.png`
- `shap_summary.png`

These figures support both the notebooks and the written documentation.

### Backend

The backend is a FastAPI application that serves both the UI and the prediction API.

Endpoints:

- `GET /` renders the frontend
- `GET /health` returns a health check payload
- `GET /model_info` returns the saved model card
- `POST /predict` scores one observation

Files:

- [app.py](/D:/Projects/ICU_Patients/app.py)
- [src/icu_risk/api/app.py](/D:/Projects/ICU_Patients/src/icu_risk/api/app.py)
- [src/icu_risk/api/schemas.py](/D:/Projects/ICU_Patients/src/icu_risk/api/schemas.py)

### Saved model artifacts

The deployment artifacts are in [models](/D:/Projects/ICU_Patients/models):

- `preprocessor.pkl`
- `deterioration_model.pkl`
- `calibrated_model.pkl`
- `model_card.json`

The `model_card.json` file records metadata such as the chosen threshold, train/validation/test split sizes, baseline metrics, and data validation summary.

### Notebooks

Documented notebooks are in [notebooks](/D:/Projects/ICU_Patients/notebooks):

- `01_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_modeling.ipynb`
- `04_explainability.ipynb`

Each notebook includes markdown narrative so the project reads like a guided analysis rather than a loose collection of code cells.

## Current model results

The generated metrics are stored in [data/processed/model_metrics.csv](/D:/Projects/ICU_Patients/data/processed/model_metrics.csv).

Summary from the latest run:

- best AUROC: logistic regression at about `0.604`
- best recall-oriented alerting threshold in the deployed card: `0.25`
- SOFA baseline AUROC: about `0.552`
- NEWS2 baseline AUROC: about `0.545`

This means the current pipeline modestly outperformed the two score-only baselines on this synthetic dataset, but the absolute performance is still limited. That is an honest and useful outcome: it tells us the project is operationally complete while still leaving room for stronger modeling, threshold tuning, and better feature refinement.

## Project structure

```text
ICU_Patients/
├── app.py
├── Dockerfile
├── README.md
├── scratch.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── notebooks/
├── scripts/
├── src/
├── static/
├── templates/
└── tests/
```

## How to run locally

### 1. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Rebuild training artifacts

```powershell
$env:PYTHONPATH='src'
python scripts\train_pipeline.py
python scripts\generate_notebooks.py
```

### 3. Start the web application

```powershell
$env:PYTHONPATH='src'
uvicorn app:app --host 0.0.0.0 --port 7000
```

Then open `http://127.0.0.1:7000`.

## Docker deployment

The project is ready for container deployment and uses port `7000` as requested.

Build:

```powershell
docker build -t icu-deterioration-app .
```

Run:

```powershell
docker run --rm -p 7000:7000 icu-deterioration-app
```

Notes:

- the container exposes port `7000`
- the health check calls `GET /health`
- `PYTHONPATH` is set to `/app/src` inside the image

## Testing

The repository includes lightweight tests for the feature pipeline and API health endpoint.

Run:

```powershell
$env:PYTHONPATH='src'
pytest
```

Tests live in [tests](/D:/Projects/ICU_Patients/tests).

## What I learned

This project reinforces a few things that matter in healthcare-flavored ML:

- strong documentation is part of the deliverable, not an extra
- missingness can be a signal, especially in clinically ordered tests
- explainability and calibration are important when predictions inform action
- deployment readiness includes structure, health checks, serialization, and reproducibility
- even when the model is not exceptionally strong, the engineering discipline around it still matters

## How to apply this to future projects

The same approach transfers well to future work:

- for other healthcare datasets, keep clinical validation rules close to the preprocessing code
- for business or operations projects, replace clinical thresholds with domain thresholds and retain the same pipeline pattern
- for time-series or streaming settings, keep the API layer and model card pattern but expand the feature store and drift monitoring
- for team projects, the current split across `src`, `scripts`, `tests`, notebooks, and deployment files is a reusable template

## Supporting notes

- The original project brief was preserved as [scratch.md](/D:/Projects/ICU_Patients/scratch.md).
- The repository ignores `scratch.md`, `venv/`, `__pycache__/`, and serialized model files through [.gitignore](/D:/Projects/ICU_Patients/.gitignore).

