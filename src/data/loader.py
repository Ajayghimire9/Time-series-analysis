from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Input dataset is empty")
    return frame


def clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].replace([float("inf"), float("-inf")], pd.NA)
    return result
