"""Minimal HTTP inference service for container/Kubernetes deployment."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models.forecasting import make_lag_features, train_models

app = FastAPI(title="Agricultural Price Forecasting API", version="2.1.0")

class PredictionRequest(BaseModel):
    history: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.history) < 20:
        raise HTTPException(status_code=400, detail="Provide at least 20 historical observations")
    series = pd.Series(request.history, dtype=float)
    X, y = make_lag_features(series, lags=12)
    model = train_models(X, y)["random_forest"]
    prediction = float(model.predict(X.iloc[[-1]])[0])
    return {"prediction": prediction, "model": "random_forest"}
