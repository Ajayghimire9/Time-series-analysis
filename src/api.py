"""HTTP inference service with health and Prometheus metrics."""

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, FiniteFloat

from src.models.forecasting import make_lag_features, train_models
from src.monitoring.metrics import LATENCY, PREDICTIONS, REQUESTS

app = FastAPI(title="Agricultural Price Forecasting API", version="2.2.0")


class PredictionRequest(BaseModel):
    history: list[FiniteFloat] = Field(min_length=20, max_length=10000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(request: PredictionRequest):
    REQUESTS.inc()
    with LATENCY.time():
        if len(request.history) < 20:
            raise HTTPException(
                status_code=400, detail="Provide at least 20 historical observations"
            )
        series = pd.Series(request.history, dtype=float)
        X, y = make_lag_features(series, lags=12)
        model = train_models(X, y)["random_forest"]
        future = pd.DataFrame([{f"lag_{lag}": series.iloc[-lag] for lag in range(1, 13)}])
        prediction = float(model.predict(future)[0])
        PREDICTIONS.inc()
        return {"prediction": prediction, "model": "random_forest"}
