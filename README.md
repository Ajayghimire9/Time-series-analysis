# Agricultural Price Forecasting & Time-Series MLOps

**End-to-end machine-learning system for agricultural price forecasting, experiment tracking, data versioning, monitored inference, and Kubernetes deployment.**

[![CI](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml)

## System architecture

```text
Agricultural data
      ↓
DVC versioning → data-quality validation
      ↓
Feature engineering → chronological validation
      ↓
Ridge / Random Forest
      ↓
MAE / RMSE
      ↓
MLflow experiments + model registry
      ↓
Docker → FastAPI
      ↓
Kubernetes (2 replicas)
      ↓
Prometheus → Grafana
```

## Implemented capabilities

### ML
- Lag-based time-series feature engineering
- Leakage-aware chronological splitting
- Ridge baseline and Random Forest benchmark
- MAE/RMSE evaluation

### MLOps
- **MLflow** experiment tracking and registered models
- **DVC** pipeline definition for reproducibility
- Explicit data-quality validation gates
- Reproducible dependency management
- pytest + Ruff quality gates

### Production serving
- FastAPI `/predict` endpoint
- `/health` health endpoint
- `/metrics` Prometheus endpoint
- Request count, prediction count, and latency metrics
- Dockerized inference
- Kubernetes Deployment and Service
- 2 replicas, resource requests/limits, health probes

### Observability
- Prometheus scraping configuration
- Grafana Prometheus datasource provisioning
- Metrics designed for operational dashboards

## Repository structure

```text
.
├── Datasets/                  # Source dataset
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── validation.py
│   ├── models/
│   │   └── forecasting.py
│   ├── mlops/
│   │   └── mlflow_tracker.py
│   ├── monitoring/
│   │   └── metrics.py
│   ├── api.py
│   └── pipeline.py
├── tests/
├── k8s/
├── monitoring/
│   └── grafana/
├── .github/workflows/
├── dvc.yaml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
python -m src.pipeline
```

## Full local MLOps stack

```bash
docker compose up --build
```

Services:

| Service | Port | Purpose |
|---|---:|---|
| FastAPI | 8000 | Model inference |
| MLflow | 5000 | Experiments/models |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics visualization |

## MLflow

Set the tracking server through environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT=agricultural-price-forecasting
```

The MLflow integration records parameters, evaluation metrics, and scikit-learn model artifacts and registers models by name.

## DVC

The repository contains a DVC pipeline definition so data dependencies and pipeline outputs can be versioned as the project grows.

```bash
dvc repro
```

For a team/cloud workflow, configure a DVC remote such as S3 and commit the resulting `.dvc` metadata rather than generated datasets.

## Kubernetes

```bash
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get services
```

For local development with Minikube or Kind, build/load the image into the cluster and port-forward the service:

```bash
kubectl port-forward service/agricultural-price-forecasting 8000:8000
curl http://localhost:8000/health
```

## Monitoring

FastAPI exposes Prometheus-compatible metrics at `/metrics`. Prometheus is configured to scrape the forecasting service, while Grafana is provisioned with Prometheus as its datasource.

Example PromQL queries:

```text
forecast_requests_total
forecast_predictions_total
rate(forecast_request_latency_seconds_count[5m])
```

## CI and retraining

GitHub Actions runs tests and linting on changes. The repository also contains a scheduled retraining workflow that can execute the forecasting pipeline and publish the resulting run to MLflow. In a real production environment, the same workflow can be replaced by an orchestrator such as Airflow, Dagster, or a managed cloud scheduler.

## Technology stack

**Python · Pandas · NumPy · scikit-learn · MLflow · DVC · FastAPI · Docker · Docker Compose · Kubernetes · Prometheus · Grafana · pytest · Ruff · GitHub Actions · Git**

These technologies are included because the repository contains corresponding implementation/configuration—not merely CV keywords.

## Engineering principles

1. Prevent temporal leakage.
2. Establish simple baselines before complex models.
3. Generate metrics from executable code.
4. Version data and model lineage.
5. Separate training from serving.
6. Instrument production inference.
7. Keep infrastructure reproducible.

## License

MIT
