from pathlib import Path
import pandas as pd
from src.models.forecasting import make_lag_features, chronological_split, train_models, evaluate


def main():
    path = Path("Datasets/dataframe.csv")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric target column found")
    target = numeric.iloc[:, -1].dropna()
    if len(target) < 30:
        raise ValueError("At least 30 observations are required")
    X, y = make_lag_features(target, lags=min(12, max(1, len(target) // 5)))
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    models = train_models(X_train, y_train)
    print("Model performance")
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test)
        print(f"{name:16s} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
