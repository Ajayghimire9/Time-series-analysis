"""Reusable forecasting models and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_lag_features(series: pd.Series, lags: int = 12):
    frame = pd.DataFrame({"target": series})
    for lag in range(1, lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    frame = frame.dropna()
    return frame.drop(columns="target"), frame["target"]


def chronological_split(X, y, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split = int(len(X) * (1 - test_size))
    if split <= 0 or split >= len(X):
        raise ValueError("test_size leaves an empty split")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_models(X_train, y_train):
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate(model, X_test, y_test):
    prediction = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
    }
