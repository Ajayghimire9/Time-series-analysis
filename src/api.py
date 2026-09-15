"""HTTP inference service with health and Prometheus metrics."""
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import pandas as pd
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.models.forecasting import make_lag_features, train_models
from src.monitoring.metrics import REQUESTS, LATENCY, PREDICTIONS

app = FastAPI(title="Agricultural Price Forecasting API", version="2.2.0")

class PredictionRequest(BaseModel):
    history: list[float]

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
            raise HTTPException(status_code=400, detail="Provide at least 20 historical observations")
        series = pd.Series(request.history, dtype=float)
        X, y = make_lag_features(series, lags=12)
        model = train_models(X, y)["random_forest"]
        prediction = float(model.predict(X.iloc[[-1]])[0])
        PREDICTIONS.inc()
        return {"prediction": prediction, "model": "random_forest"}
