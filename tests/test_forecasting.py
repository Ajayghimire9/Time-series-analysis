import pandas as pd
import pytest
from src.models.forecasting import make_lag_features, chronological_split


def test_lag_features_preserve_alignment():
    X, y = make_lag_features(pd.Series([1, 2, 3, 4, 5]), lags=2)
    assert list(X.columns) == ["lag_1", "lag_2"]
    assert y.tolist() == [3, 4, 5]


def test_split_preserves_time_order():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    X_train, X_test, y_train, y_test = chronological_split(X, y, 0.2)
    assert X_train.index.max() < X_test.index.min()
    assert y_train.iloc[-1] == 7
    assert y_test.iloc[0] == 8


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"x": [1, 2]}), pd.Series([1, 2]), 1.0)
