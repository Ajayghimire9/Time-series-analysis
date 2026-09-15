# HarvestCast

Agricultural price forecasting.

This project studies one-step forecasts from a historical price series. The focus is temporal evaluation: a prediction must only use observations that were available at its forecast origin.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m src.pipeline
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

The API constructs the next observation from the most recent 12 lags. It no longer returns a fitted prediction for the final training row.

Ridge and random forest baselines use chronological splits. `src/models/backtest.py` adds expanding-window folds, a configurable gap and a persistence baseline.

MLflow tracking and registry helpers are available when a tracking URI is configured. Prometheus records request counts and latency.

## Technology

Python, pandas, scikit-learn, FastAPI, MLflow, Prometheus; Docker and Kubernetes configuration.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The API fits a local model for each request; it is a small experiment service, not a high-throughput serving system. The main pipeline compares models on its chronological holdout; use the rolling backtest for broader validation. Historical price files require an explicit target selection with TARGET_COLUMN when their meaning is ambiguous.
