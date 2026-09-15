"""Promote a candidate MLflow model when it beats the current champion."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient


def promote_if_better(model_name: str, candidate_version: str, candidate_rmse: float) -> str:
    """Assign candidate/champion aliases using RMSE as the promotion criterion."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "candidate", candidate_version)
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        champion = None

    if champion is None:
        client.set_registered_model_alias(model_name, "champion", candidate_version)
        return "promoted"

    champion_run = client.get_run(champion.run_id)
    champion_rmse = float(champion_run.data.metrics.get("rmse", float("inf")))
    if candidate_rmse < champion_rmse:
        client.set_registered_model_alias(model_name, "champion", candidate_version)
        return "promoted"
    return "rejected"


def load_champion(model_name: str):
    """Load the current champion through its stable MLflow alias."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
