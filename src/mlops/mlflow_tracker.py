"""MLflow experiment and model-registry integration."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "agricultural-price-forecasting"))


def log_run(model, model_name: str, metrics: dict[str, float], params: dict | None = None):
    configure_mlflow()
    with mlflow.start_run(run_name=model_name) as run:
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        return run.info.run_id
