"""Generate richly documented project notebooks from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

from icu_risk.config import FIGURES_DIR, MODELS_DIR, NOTEBOOKS_DIR, PROCESSED_DIR


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def code_cell(text: str):
    return nbf.v4.new_code_cell(text)


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        nbf.write(notebook, file)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(PROCESSED_DIR / "model_metrics.csv")
    with open(MODELS_DIR / "model_card.json", "r", encoding="utf-8") as file:
        model_card = json.load(file)

    eda_cells = [
        markdown_cell(
            "# 01 EDA\n"
            "This notebook documents the exploratory phase for the ICU deterioration dataset. "
            "The focus is on clinical baselines, missingness behavior, and early signals that "
            "support downstream feature engineering."
        ),
        markdown_cell(
            "## Why this notebook matters\n"
            "- The dataset is clinically sensitive even though it is synthetic.\n"
            "- SOFA and NEWS2 are meaningful baselines, not placeholder features.\n"
            "- Missingness is informative, especially for `gcs_score`, `lactate`, and `pao2_fio2_ratio`."
        ),
        code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "df = pd.read_csv(Path('../data/processed/icu_patients_model_ready.csv'))\n"
            "df.head()"
        ),
        markdown_cell(
            "## Saved visuals\n"
            "The pipeline exports reusable figures to keep results consistent across notebooks, the README, and deployment documentation."
        ),
        markdown_cell(
            "![Target Distribution](../figures/target_distribution.png)\n\n"
            "![Clinical Score Boxplots](../figures/clinical_score_boxplots.png)\n\n"
            "![Vital Distributions](../figures/vital_distributions.png)"
        ),
    ]

    feature_cells = [
        markdown_cell(
            "# 02 Feature Engineering\n"
            "This notebook explains the clinical reasoning behind the engineered features and the sequence used in the preprocessing pipeline."
        ),
        markdown_cell(
            "## Design notes\n"
            "- Missingness flags are created before imputation.\n"
            "- Threshold flags map to real ICU practice, such as `map_below_65` and `lactate_critical`.\n"
            "- Composite features capture instability beyond any single measurement."
        ),
        code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "df = pd.read_csv(Path('../data/processed/icu_patients_model_ready.csv'))\n"
            "engineered_columns = [c for c in df.columns if c.endswith('_missing') or c in ['shock_index', 'organ_failure_count', 'fluid_overload_flag']]\n"
            "df[engineered_columns].head()"
        ),
    ]

    modeling_cells = [
        markdown_cell(
            "# 03 Modeling\n"
            "This notebook compares logistic regression, random forest, XGBoost, and the clinical baselines."
        ),
        markdown_cell(
            f"## Best model summary\n"
            f"- Selected threshold: `{model_card['alert_threshold']:.2f}`\n"
            f"- SOFA baseline AUROC: `{model_card['baselines']['sofa_roc_auc']:.3f}`\n"
            f"- NEWS2 baseline AUROC: `{model_card['baselines']['news2_roc_auc']:.3f}`"
        ),
        code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "metrics = pd.read_csv(Path('../data/processed/model_metrics.csv'))\n"
            "metrics"
        ),
        markdown_cell("![ROC and PR Curves](../figures/model_curves.png)\n\n![Confusion Matrix](../figures/confusion_matrix.png)"),
    ]

    explain_cells = [
        markdown_cell(
            "# 04 Explainability And Deployment Notes\n"
            "This notebook connects the explainability outputs to production concerns such as calibration, alert thresholds, and model governance."
        ),
        markdown_cell(
            "## Production-oriented interpretation\n"
            "- SHAP is used for clinician-friendly local and global explanations.\n"
            "- Calibration matters because the score is expected to trigger real decisions.\n"
            "- Model metadata is stored in `model_card.json` for auditability."
        ),
        code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "with open(Path('../models/model_card.json'), 'r', encoding='utf-8') as file:\n"
            "    model_card = json.load(file)\n"
            "model_card"
        ),
        markdown_cell(
            "![Feature Importance](../figures/feature_importance.png)\n\n"
            "![Calibration Curve](../figures/calibration_curve.png)\n\n"
            "![SHAP Summary](../figures/shap_summary.png)"
        ),
        markdown_cell(
            "## Metrics snapshot\n"
            + dataframe_to_markdown(metrics_df.round(4))
        ),
    ]

    write_notebook(NOTEBOOKS_DIR / "01_eda.ipynb", eda_cells)
    write_notebook(NOTEBOOKS_DIR / "02_feature_engineering.ipynb", feature_cells)
    write_notebook(NOTEBOOKS_DIR / "03_modeling.ipynb", modeling_cells)
    write_notebook(NOTEBOOKS_DIR / "04_explainability.ipynb", explain_cells)


if __name__ == "__main__":
    main()
