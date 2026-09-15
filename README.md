# Agricultural Price Forecasting & Time-Series MLOps

**End-to-end machine-learning system for agricultural price forecasting, experiment tracking, data versioning, automated retraining, model promotion, monitored inference, and Kubernetes deployment.**

[![CI](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml)

## System architecture

```text
Agricultural data
      ↓
DVC versioning → data-quality validation → drift analysis
      ↓
Feature engineering → chronological validation
      ↓
Ridge / Random Forest
      ↓
MAE / RMSE
      ↓
MLflow tracking → candidate model → champion alias
      ↓
Docker → FastAPI
      ↓
Kubernetes (2 replicas)
      ↓
Prometheus → Grafana → operational alerts

GitHub Actions → scheduled retraining → training report → MLflow
```

## Implemented capabilities

### ML
- Lag-based time-series feature engineering
- Leakage-aware chronological splitting
- Ridge baseline and Random Forest benchmark
- MAE/RMSE evaluation
- Configurable target column and lag count

### MLOps
- **MLflow** experiment tracking and registered models
- MLflow `candidate` and `champion` aliases with RMSE-based promotion
- **DVC** pipeline definition for reproducibility
- Explicit data-quality validation gates
- Lightweight PSI-based distribution drift detection
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
- Prometheus alert rules for API availability and high latency
- Grafana datasource provisioning
- Prebuilt Grafana dashboard for request rate, latency, uptime, and predictions

### Automation
- GitHub Actions CI on pushes and pull requests
- Weekly scheduled retraining workflow
- Manual retraining through `workflow_dispatch`
- Training report uploaded as a GitHub Actions artifact
- Optional MLflow tracking through `MLFLOW_TRACKING_URI` secret

## Repository structure

```text
.
├── Datasets/                  # Source dataset
├── src/
│   ├── data/                 # Loading and validation
│   ├── models/               # Feature engineering, training, evaluation
│   ├── mlops/                # MLflow tracking and model promotion
│   ├── monitoring/           # Prometheus metrics and drift detection
│   ├── api.py                # FastAPI inference service
│   └── pipeline.py           # End-to-end training pipeline
├── tests/
├── k8s/                      # Kubernetes manifests
├── monitoring/               # Prometheus, alerts, Grafana provisioning
├── .github/workflows/        # CI and scheduled retraining
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

Choose a target explicitly when needed:

```bash
TARGET_COLUMN=Radish python -m src.pipeline
```

## Full local MLOps stack

```bash
docker compose up --build
```

Services:

| Service | Port | Purpose |
|---|---:|---|
| FastAPI | 8000 | Model inference |
| MLflow | 5000 | Experiments and model registry |
| Prometheus | 9090 | Metrics collection and alerts |
| Grafana | 3000 | Operational dashboard |

Grafana automatically provisions the forecasting dashboard from `monitoring/grafana/dashboards/`.

## MLflow model lifecycle

Training registers each model under a stable name such as `agri-forecast-randomforest`. Every new version receives the `candidate` alias. The best model is compared with the existing `champion` using RMSE; it replaces the champion only when its RMSE is lower.

This uses MLflow aliases rather than hard-coded model-version numbers, so serving systems can reference a stable `champion` endpoint.

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

## Monitoring and drift

FastAPI exposes Prometheus-compatible metrics at `/metrics`. Prometheus scrapes the forecasting service and loads alert rules from `monitoring/alerts.yml`.

The project also includes a dependency-light Population Stability Index implementation for comparing reference and current numeric distributions:

```python
from src.monitoring.drift import population_stability_index, drift_status

psi = population_stability_index(reference_values, current_values)
print(psi, drift_status(psi))
```

Operational thresholds are intentionally explicit: below `0.10` is stable, `0.10–0.25` is a warning, and `>=0.25` is critical.

## CI and retraining

GitHub Actions runs tests and linting on changes. A weekly scheduled workflow can retrain the models and publish the generated training report. If `MLFLOW_TRACKING_URI` is configured as a repository secret, the workflow also records runs and model versions in the shared MLflow server.

For larger production environments, the same training entry point can be orchestrated by Airflow, Dagster, or a managed cloud scheduler without changing the core ML code.

## Technology stack

**Python · Pandas · NumPy · scikit-learn · MLflow · DVC · FastAPI · Docker · Docker Compose · Kubernetes · Prometheus · Grafana · pytest · Ruff · GitHub Actions · Git**

These technologies are included because the repository contains corresponding implementation/configuration.

## Engineering principles

1. Prevent temporal leakage.
2. Establish simple baselines before complex models.
3. Generate metrics from executable code.
4. Version data and model lineage.
5. Separate training from serving.
6. Instrument production inference.
7. Detect distribution changes before they become silent failures.
8. Promote models using measurable evaluation criteria.
9. Keep infrastructure reproducible.

## License

MIT
