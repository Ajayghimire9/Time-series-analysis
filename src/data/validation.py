"""Lightweight data-quality gates used before model training."""
from __future__ import annotations
import pandas as pd

def validate_frame(frame: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    if frame.empty: raise ValueError("Dataset is empty")
    if frame.columns.duplicated().any(): raise ValueError("Duplicate column names detected")
    if required_columns:
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing: raise ValueError(f"Missing required columns: {missing}")
    numeric = frame.select_dtypes(include="number")
    if numeric.isin([float("inf"), float("-inf")]).any().any(): raise ValueError("Infinite numeric values detected")
