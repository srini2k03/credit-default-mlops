# MLflow tracking server + model registry (SQLite backend, artifacts proxied by the server).
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MLFLOW_DISABLE_TELEMETRY=true
RUN pip install "mlflow==3.17.2"
RUN mkdir -p /mlflow/artifacts
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
