# Agricultural Price Forecasting & Time-Series Analytics

**Production-style machine learning project for agricultural price forecasting, time-series feature engineering, reproducible evaluation, and software delivery.**

[![CI](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Time-series-analysis/actions/workflows/ci.yml)

## Overview

This project analyzes Japanese agricultural price observations and builds a leakage-aware forecasting workflow. The original exploratory analysis has been reorganized into reusable Python components so the repository demonstrates both **machine-learning knowledge and engineering discipline**.

### Pipeline

`Raw data → validation → cleaning → lag features → chronological split → baseline models → evaluation → reproducible delivery`

## Key engineering decisions

- **Chronological validation:** future observations are never shuffled into training data.
- **Baseline-first modeling:** Ridge provides a simple benchmark before a non-linear Random Forest model.
- **Explicit metrics:** MAE and RMSE are calculated from predictions rather than copied into documentation.
- **Reusable code:** forecasting logic lives in `src/` instead of a single notebook-style script.
- **Automated quality:** pytest and Ruff run in GitHub Actions.
- **Containerized execution:** Docker provides a consistent runtime.

## Repository structure

```text
.
├── Datasets/                  # Original project datasets
├── src/
│   ├── data/                  # Loading and cleaning
│   ├── models/                # Forecasting and evaluation
│   └── pipeline.py            # End-to-end entry point
├── tests/                     # Automated tests
├── .github/workflows/         # CI pipeline
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Quick start

```bash
git clone https://github.com/Ajayghimire9/Time-series-analysis.git
cd Time-series-analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest
python -m src.pipeline
```

## Docker

```bash
docker build -t agricultural-price-forecasting .
docker run --rm agricultural-price-forecasting
```

## Development commands

```bash
make install
make test
make lint
make run
```

## Technology

**Python · Pandas · NumPy · scikit-learn · pytest · Ruff · Docker · GitHub Actions · Git**

These technologies are included because they are implemented in the repository—not simply listed as portfolio keywords.

## Data

The project uses the Japanese agricultural price dataset already included in the repository. The original analysis investigated missing observations and multiple imputation strategies. The refactored pipeline adds a cleaner foundation for reproducible modeling.

## Model evaluation

The pipeline reports:

- **MAE** — average absolute prediction error
- **RMSE** — penalizes larger errors more strongly

No performance numbers are hard-coded into this README. Run the pipeline to generate the current metrics from the repository data.

## Roadmap

- [ ] Add formal data-quality validation
- [ ] Add DVC dataset versioning
- [ ] Add MLflow experiment tracking
- [ ] Persist trained model artifacts
- [ ] Add forecasting visualization/report generation
- [ ] Add scheduled pipeline execution

## Why this project matters for Data Engineering / MLOps

The focus is deliberately broader than model training. The project demonstrates the workflow expected around a production ML system: structured code, reproducible environments, automated tests, CI, data handling, chronological evaluation, and a clear path toward experiment and artifact management.

## License

MIT
