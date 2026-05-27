"""Run the full ICU deterioration training pipeline and persist artifacts."""

from icu_risk.training.train import train_all_models


if __name__ == "__main__":
    results = train_all_models()
    print(results["metrics"].to_string(index=False))

