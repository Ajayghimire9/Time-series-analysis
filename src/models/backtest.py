"""Expanding-window evaluation with an explicit persistence baseline."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.models.forecasting import make_lag_features


def rolling_backtest(series, estimator, lags=12, folds=3, gap=0):
    values = np.asarray(series, dtype=float)
    if not np.isfinite(values).all() or lags < 1:
        raise ValueError("Provide finite observations and positive lags")
    x, y = make_lag_features(pd.Series(values), lags)
    reports = []
    for train, test in TimeSeriesSplit(n_splits=folds, gap=gap).split(x):
        model = clone(estimator).fit(x.iloc[train], y.iloc[train])
        pred = model.predict(x.iloc[test])
        reports.append(
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "mae": float(mean_absolute_error(y.iloc[test], pred)),
                "rmse": float(np.sqrt(mean_squared_error(y.iloc[test], pred))),
                "persistence_mae": float(mean_absolute_error(y.iloc[test], x.iloc[test]["lag_1"])),
            }
        )
    return reports
