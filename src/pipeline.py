"""End-to-end forecasting pipeline with optional MLflow tracking."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.data.validation import validate_frame
from src.models.forecasting import chronological_split, evaluate, make_lag_features, train_models


def main() -> None:
    path = Path(os.getenv("DATA_PATH", "Datasets/dataframe.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    validate_frame(frame)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric target column found")

    requested_target = os.getenv("TARGET_COLUMN")
    target_name = requested_target or numeric.columns[-1]
    if target_name not in numeric.columns:
        raise ValueError(
            f"TARGET_COLUMN={target_name!r} is not numeric or does not exist. "
            f"Available numeric columns: {list(numeric.columns)}"
        )

    target = numeric[target_name].dropna()
    if len(target) < 30:
        raise ValueError("At least 30 observations are required")

    lags = min(int(os.getenv("LAGS", "12")), max(1, len(target) // 5))
    X, y = make_lag_features(target, lags=lags)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    models = train_models(X_train, y_train)

    results: dict[str, dict[str, float]] = {}
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "agricultural-price-forecasting")

    if tracking_uri:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    print(f"Target: {target_name}")
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        if tracking_uri:
            registered_name = f"agri-forecast-{name.lower()}"
            with mlflow.start_run(run_name=f"{target_name}-{name}"):
                mlflow.log_params(
                    {"model": name, "target": target_name, "lags": lags, "test_size": 0.2}
                )
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model", registered_model_name=registered_name)

                versions = MlflowClient().search_model_versions(f"name='{registered_name}'")
                latest = max(versions, key=lambda version: int(version.version))
                MlflowClient().set_registered_model_alias(
                    registered_name, "candidate", latest.version
                )
                print(f"MLflow candidate: {registered_name}@candidate -> {latest.version}")
        print(f"{name:16s} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")

    best_model = min(results, key=lambda model_name: results[model_name]["rmse"])
    if tracking_uri:
        from src.mlops.model_promotion import promote_if_better

        registered_name = f"agri-forecast-{best_model.lower()}"
        versions = MlflowClient().search_model_versions(f"name='{registered_name}'")
        latest = max(versions, key=lambda version: int(version.version))
        status = promote_if_better(registered_name, latest.version, results[best_model]["rmse"])
        print(f"Champion decision: {status} ({registered_name}@{latest.version})")

    report = {
        "target": target_name,
        "observations": len(target),
        "lags": lags,
        "best_model": best_model,
        "metrics": results,
    }
    report_path = Path(os.getenv("METRICS_PATH", "reports/model_metrics.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
