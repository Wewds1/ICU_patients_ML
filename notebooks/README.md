# Notebook Notes

The notebooks are generated from the saved artifacts so they stay aligned with the latest training run.

## Sequence

- `01_eda.ipynb` introduces the dataset and baseline distributions
- `02_feature_engineering.ipynb` explains the engineered features
- `03_modeling.ipynb` compares the trained models with score baselines
- `04_explainability.ipynb` connects feature importance, SHAP, and calibration to deployment

## Usage

Regenerate the notebooks after retraining:

```powershell
$env:PYTHONPATH='src'
python scripts\generate_notebooks.py
```

