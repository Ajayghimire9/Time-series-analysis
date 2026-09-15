# Agricultural Price Forecasting & Time-Series MLOps

**End-to-end machine-learning system for agricultural price forecasting, experiment tracking, containerized inference, and Kubernetes deployment.**

[![CI](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml)

## Overview

This project analyzes Japanese agricultural price observations and builds a leakage-aware forecasting workflow. The original exploratory analysis has been reorganized into reusable Python components and extended into an MLOps-oriented system.

### Architecture

```text
                         +------------------+
                         | Agricultural CSV |
                         +--------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    | Validation / Preprocessing|
                    +-------------+-------------+
                                  |
                                  v
                       +----------+----------+
                       | Lag Feature Engine  |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       | Chronological Split |
                       +----------+----------+
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                      Ridge         Random Forest
                         |                 |
                         +--------+--------+
                                  |
                                  v
                           MAE / RMSE
                                  |
                                  v
                             MLflow
                       experiments + models
                                  |
                                  v
                        Docker / FastAPI
                                  |
                                  v
                           Kubernetes
                       replicas + probes
```

## What is implemented

### Machine Learning
- Lag-based time-series feature engineering
- Leakage-aware chronological train/test split
- Ridge regression baseline
- Random Forest benchmark
- MAE and RMSE evaluation

### MLOps
- **MLflow** experiment tracking and model logging
- Configurable `MLFLOW_TRACKING_URI`
- Reproducible dependency configuration
- Automated tests with pytest
- Ruff code-quality checks

### Deployment
- Dockerized FastAPI inference service
- `/health` readiness/liveness endpoint
- `/predict` prediction endpoint
- Docker Compose local MLflow stack
- Kubernetes Deployment with **2 replicas**
- Kubernetes resource requests/limits
- Kubernetes readiness and liveness probes
- Kubernetes service discovery for MLflow

## Repository structure

```text
.
├── Datasets/
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── models/
│   │   └── forecasting.py
│   ├── api.py
│   └── pipeline.py
├── tests/
├── k8s/
│   ├── deployment.yaml
│   ├── mlflow.yaml
│   └── README.md
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Run locally

```bash
git clone https://github.com/Ajayghimire9/Time-series-analysis.git
cd Time-series-analysis
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m src.pipeline
```

## MLflow

Start the local tracking server with Docker Compose:

```bash
docker compose up mlflow
```

In another terminal, run the pipeline against MLflow:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.pipeline
```

The pipeline logs model parameters, MAE/RMSE metrics, and trained scikit-learn models to MLflow.

## API

Build and start the inference service:

```bash
docker build -t agricultural-forecasting .
docker run --rm -p 8000:8000 agricultural-forecasting
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"history":[171,198,196,207,221,226,253,260,275,281,290,301,315,320,330,340,350,360,370,380]}'
```

## Kubernetes

The manifests demonstrate a production-oriented deployment pattern:

```bash
docker build -t agricultural-forecasting:latest .
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get services
```

For a local cluster:

```bash
kubectl port-forward service/agricultural-forecasting 8000:8000
curl http://localhost:8000/health
```

The forecasting service runs two replicas and includes resource limits plus health probes. MLflow is available to the application through the Kubernetes service name `mlflow:5000`.

## CI/CD

GitHub Actions automatically installs the project, runs Ruff, and executes pytest on pushes and pull requests.

## Technology stack

**Python · Pandas · NumPy · scikit-learn · MLflow · FastAPI · Docker · Docker Compose · Kubernetes · pytest · Ruff · GitHub Actions · Git**

Every technology listed above is backed by implementation in the repository.

## Engineering principles

1. **No temporal leakage** — training never uses future observations.
2. **Baseline before complexity** — model improvements are measured against Ridge.
3. **Reproducibility over screenshots** — metrics are generated by code.
4. **Operational thinking** — models are tracked, packaged, served, and deployable.
5. **Simple infrastructure first** — the local stack can be run without a cloud account.

## Roadmap

- [ ] DVC dataset versioning
- [ ] Automated data-quality checks
- [ ] Model registry promotion workflow
- [ ] Cloud deployment
- [ ] Prometheus/Grafana monitoring
- [ ] Scheduled retraining workflow

## License

MIT
