from fastapi.testclient import TestClient

from src import api


def test_forecast_uses_most_recent_observation(monkeypatch):
    class LastValue:
        def predict(self, frame):
            return frame["lag_1"].to_numpy()

    monkeypatch.setattr(api, "train_models", lambda x, y: {"random_forest": LastValue()})
    client = TestClient(api.app)
    response = client.post("/predict", json={"history": list(range(30))})
    assert response.json()["prediction"] == 29
    assert client.post("/predict", json={"history": [1]}).status_code == 422
