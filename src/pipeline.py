"""End-to-end forecasting pipeline with optional MLflow tracking."""
from pathlib import Path
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from src.models.forecasting import make_lag_features, chronological_split, train_models, evaluate


def main():
    path = Path(os.getenv("DATA_PATH", "Datasets/dataframe.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric target column found")
    target_name = numeric.columns[-1]
    target = numeric[target_name].dropna()
    if len(target) < 30:
        raise ValueError("At least 30 observations are required")

    X, y = make_lag_features(target, lags=min(12, max(1, len(target) // 5)))
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    models = train_models(X_train, y_train)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "agricultural-price-forecasting"))

    print(f"Target: {target_name}")
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "lags": X.shape[1], "test_size": 0.2})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
        print(f"{name:16s} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
