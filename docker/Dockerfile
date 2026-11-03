# syntax=docker/dockerfile:1.7
# Two targets from one base:
#   api   - FastAPI service (default for CI/GHCR):  docker build -f docker/Dockerfile --target api .
#   train - training/flow image used by docker-compose's "trainer" profile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=1
# libgomp: OpenMP runtime for LightGBM/XGBoost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

FROM base AS api
RUN pip install -e ".[api]"
# The bundled model (committed by train.yml) is the fallback when no registry is reachable.
COPY models ./models
RUN useradd --create-home --uid 10001 app && mkdir -p /app/logs && chown -R app /app/logs
USER app
ENV PORT=8000 \
    MODEL_SOURCE=auto \
    MODEL_BUNDLE_PATH=/app/models/bundled/model.joblib \
    PREDICTION_LOG_PATH=/app/logs/predictions.sqlite
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn credit_default.serve.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]

FROM base AS train
RUN pip install -e ".[train,monitor,orchestrate]"
COPY configs ./configs
COPY flows ./flows
COPY models ./models
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false
CMD ["python", "flows/training_flow.py"]
