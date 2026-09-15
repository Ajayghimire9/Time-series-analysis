"""Prometheus metrics for the inference service."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("forecast_requests_total", "Total forecast API requests")
LATENCY = Histogram("forecast_request_latency_seconds", "Forecast API latency")
PREDICTIONS = Counter("forecast_predictions_total", "Successful predictions")
