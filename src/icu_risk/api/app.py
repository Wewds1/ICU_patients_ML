from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from icu_risk.config import MODELS_DIR
from icu_risk.data import canonicalize_categories, impute_clinical_missingness
from icu_risk.features.pipeline import engineer_features
from icu_risk.api.schemas import PatientObservation, PredictionResponse

app = FastAPI(
    title="ICU Deterioration Risk API",
    description="Black-and-white UI and prediction API for ICU patient deterioration risk.",
    version="1.0.0",
)

project_root = Path(__file__).resolve().parents[3]
templates = Jinja2Templates(directory=str(project_root / "templates"))
app.mount("/static", StaticFiles(directory=str(project_root / "static")), name="static")

model = None
model_card = None


def _load_artifacts() -> tuple[object, dict]:
    global model, model_card
    if model is None:
        model = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    if model_card is None:
        with open(MODELS_DIR / "model_card.json", "r", encoding="utf-8") as file:
            model_card = json.load(file)
    return model, model_card


def prepare_payload(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    frame = canonicalize_categories(frame)
    frame = impute_clinical_missingness(frame)
    frame = engineer_features(frame)
    return frame


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model_info")
def model_info() -> dict:
    _, card = _load_artifacts()
    return card


@app.post("/predict", response_model=PredictionResponse)
def predict(observation: PatientObservation) -> PredictionResponse:
    calibrated_model, card = _load_artifacts()
    prepared = prepare_payload(observation.model_dump())
    probability = float(calibrated_model.predict_proba(prepared)[0][1])
    threshold = float(card["alert_threshold"])

    if probability >= 0.50:
        risk_level = "HIGH"
    elif probability >= threshold:
        risk_level = "ELEVATED"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        deterioration_probability=round(probability, 4),
        alert_triggered=probability >= threshold,
        alert_threshold=threshold,
        risk_level=risk_level,
        model_version=str(card["model_version"]),
    )

