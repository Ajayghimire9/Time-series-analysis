# Kubernetes deployment

This directory demonstrates container orchestration for the forecasting API and its MLflow tracking service.

```bash
docker build -t agricultural-forecasting:latest .
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get services
```

For a local cluster such as Minikube or Docker Desktop Kubernetes, the forecasting service can be exposed with:

```bash
kubectl port-forward service/agricultural-forecasting 8000:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

The deployment includes two replicas, resource requests/limits, readiness probes and liveness probes. MLflow is exposed internally as `http://mlflow:5000`.
