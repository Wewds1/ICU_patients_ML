from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from icu_risk.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from icu_risk.data import build_modeling_dataset, load_raw_data
from icu_risk.features.pipeline import build_preprocessor, engineer_features, split_features_target

sns.set_theme(style="whitegrid")


@dataclass
class ModelMetrics:
    model_name: str
    roc_auc: float
    pr_auc: float
    recall: float
    specificity: float
    precision: float
    threshold: float


def specificity_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp))


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.1, 0.55, 0.05)
    best_threshold = 0.25
    best_specificity = -1.0

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions)
        specificity = specificity_score(y_true, predictions)
        if recall >= 0.80 and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = specificity
    return best_threshold


def compute_metrics(
    model_name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> ModelMetrics:
    predictions = (probabilities >= threshold).astype(int)
    return ModelMetrics(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        recall=float(recall_score(y_true, predictions)),
        specificity=float(specificity_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions)),
        threshold=float(threshold),
    )


def make_figures(
    clean_df: pd.DataFrame,
    y_test: pd.Series,
    sofa_scores: pd.Series,
    news2_scores: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    xgb_pipeline: Pipeline,
    calibrated_model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    threshold: float,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(
        x="deterioration_12h",
        hue="deterioration_12h",
        data=clean_df,
        palette=["white", "black"],
        legend=False,
    )
    plt.title("Target Distribution")
    plt.xlabel("Deterioration Within 12 Hours")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=clean_df, x="deterioration_12h", y="sofa_score", color="white", ax=axes[0])
    axes[0].set_title("SOFA Score by Outcome")
    sns.boxplot(data=clean_df, x="deterioration_12h", y="news2_score", color="lightgray", ax=axes[1])
    axes[1].set_title("NEWS2 Score by Outcome")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "clinical_score_boxplots.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_columns = ["hr_max", "spo2_min", "map_min", "lactate"]
    for axis, column in zip(axes.flatten(), plot_columns):
        sns.histplot(
            data=clean_df,
            x=column,
            hue="deterioration_12h",
            bins=30,
            stat="density",
            common_norm=False,
            palette=["white", "black"],
            ax=axis,
        )
        axis.set_title(f"{column} Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vital_distributions.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, probabilities in {"SOFA": sofa_scores, "NEWS2": news2_scores, **model_probabilities}.items():
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, probabilities, name=name, ax=axes[1])
    axes[0].set_title("ROC Curves")
    axes[1].set_title("Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_curves.png", dpi=200)
    plt.close(fig)

    xgb_probabilities = model_probabilities["xgboost"]
    predictions = (xgb_probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Greys", ax=ax)
    ax.set_title("XGBoost Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    transformed = xgb_pipeline.named_steps["preprocessor"].transform(X_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
    booster = xgb_pipeline.named_steps["model"]
    importances = pd.Series(booster.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, color="black")
    plt.title("Top XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close()

    calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(y_test, calibrated_probabilities, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(mean_predicted, fraction_positive, marker="o", color="black")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Deterioration Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=200)
    plt.close()

    # SHAP can be expensive; sample enough rows for a readable summary plot.
    sample_size = min(300, X_test.shape[0])
    X_sample = X_test.iloc[:sample_size]
    transformed_sample = xgb_pipeline.named_steps["preprocessor"].transform(X_sample)
    if hasattr(transformed_sample, "toarray"):
        transformed_sample = transformed_sample.toarray()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(transformed_sample)
    shap.summary_plot(
        shap_values,
        transformed_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
        plot_size=(10, 8),
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def train_all_models() -> dict[str, object]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data()
    clean_df, validation_report = build_modeling_dataset(raw_df)
    feature_df = engineer_features(clean_df)
    feature_df.to_csv(PROCESSED_DIR / "icu_patients_model_ready.csv", index=False)

    X, y = split_features_target(feature_df)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
    }

    fitted_pipelines: dict[str, Pipeline] = {}
    validation_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        validation_scores[name] = pipeline.predict_proba(X_val)[:, 1]
        test_scores[name] = pipeline.predict_proba(X_test)[:, 1]

    threshold = choose_threshold(y_val, validation_scores["xgboost"])
    metrics = [
        compute_metrics(name, y_test, probabilities, threshold if name == "xgboost" else 0.5)
        for name, probabilities in test_scores.items()
    ]

    sofa_scores = X_test["sofa_score"]
    news2_scores = X_test["news2_score"]
    metrics.extend(
        [
            compute_metrics("sofa_baseline", y_test, sofa_scores.to_numpy(), threshold),
            compute_metrics("news2_baseline", y_test, news2_scores.to_numpy(), threshold),
        ]
    )

    metrics_df = pd.DataFrame([asdict(metric) for metric in metrics]).sort_values(
        by="roc_auc", ascending=False
    )
    metrics_df.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)

    best_pipeline = fitted_pipelines["xgboost"]
    calibrated_model = CalibratedClassifierCV(
        estimator=fitted_pipelines["xgboost"],
        method="isotonic",
        cv=3,
    )
    calibrated_model.fit(X_train_val, y_train_val)

    model_card = {
        "project": "ICU Patient Deterioration Early Warning",
        "model_version": "1.0.0",
        "alert_threshold": threshold,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "validation_report": validation_report.to_dict(),
        "baselines": {
            "sofa_roc_auc": float(roc_auc_score(y_test, sofa_scores)),
            "news2_roc_auc": float(roc_auc_score(y_test, news2_scores)),
        },
        "best_model": metrics_df.iloc[0].to_dict(),
    }

    joblib.dump(best_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")
    joblib.dump(best_pipeline, MODELS_DIR / "deterioration_model.pkl")
    joblib.dump(calibrated_model, MODELS_DIR / "calibrated_model.pkl")
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as file:
        json.dump(model_card, file, indent=2)

    make_figures(
        clean_df,
        y_test,
        sofa_scores,
        news2_scores,
        test_scores,
        best_pipeline,
        calibrated_model,
        X_test,
        threshold,
    )

    return {
        "metrics": metrics_df,
        "model_card": model_card,
        "threshold": threshold,
    }
