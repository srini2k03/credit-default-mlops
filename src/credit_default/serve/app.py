"""FastAPI app. Run: uvicorn credit_default.serve.app:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from credit_default import __version__
from credit_default.serve.model_loader import REPO_ROOT, LoadedModel, load_model
from credit_default.serve.prediction_log import PredictionLogger
from credit_default.serve.schemas import (
    BatchRequest,
    BatchResponse,
    Contribution,
    CreditApplication,
    Explanation,
    Health,
    Prediction,
)

log = logging.getLogger("credit_default.serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.pred_log = PredictionLogger(
        os.environ.get("PREDICTION_LOG_PATH", REPO_ROOT / "logs" / "predictions.sqlite"),
        enabled=os.environ.get("PREDICTION_LOG_ENABLED", "true").lower() == "true",
    )
    yield


app = FastAPI(
    title="Credit default risk API",
    version=__version__,
    description=(
        "Calibrated probability of credit-card default next month, a cost-optimal approve/decline "
        "decision, and per-request SHAP explanations. Model: `models:/credit-default@production` "
        "from the MLflow registry, or the bundled CI-trained artifact when no registry is reachable."
    ),
    lifespan=lifespan,
)


def _model(request: Request) -> LoadedModel:
    m: LoadedModel = request.app.state.model
    if m.bundle is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {m.error}")
    return m


def _frame(apps: list[CreditApplication]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in apps])


def _predict(request: Request, apps: list[CreditApplication], endpoint: str) -> list[Prediction]:
    m = _model(request)
    X = _frame(apps)
    proba = m.bundle.predict_proba(X)
    out, rows = [], []
    for app_, p in zip(apps, proba, strict=True):
        rid = uuid.uuid4().hex
        decline = bool(p >= m.bundle.threshold)
        out.append(
            Prediction(
                request_id=rid,
                probability_of_default=float(p),
                default_prediction=decline,
                decision="decline" if decline else "approve",
                threshold=m.bundle.threshold,
                model_version=m.version,
                model_source=m.source,
            )
        )
        rows.append(
            {
                "request_id": rid,
                "endpoint": endpoint,
                "model_source": m.source,
                "model_version": m.version,
                "probability": float(p),
                "prediction": decline,
                "threshold": m.bundle.threshold,
                "features": app_.model_dump(),
            }
        )
    try:
        request.app.state.pred_log.log(rows)
    except Exception as exc:  # logging must never fail a prediction
        log.warning("prediction log write failed: %s", exc)
    return out


@app.get("/", include_in_schema=False)
def root():
    return {"service": "credit-default-mlops", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=Health)
def health(request: Request):
    m: LoadedModel = request.app.state.model
    if m.bundle is None:
        body = Health(status="unavailable", model_loaded=False, detail=m.error)
        return JSONResponse(status_code=503, content=body.model_dump())
    return Health(
        status="ok",
        model_loaded=True,
        model_source=m.source,
        model_version=m.version,
        model_family=m.bundle.family,
        threshold=m.bundle.threshold,
    )


@app.get("/model")
def model_info(request: Request):
    m = _model(request)
    return {
        "source": m.source,
        "version": m.version,
        "uri": m.uri,
        "family": m.bundle.family,
        "threshold": m.bundle.threshold,
        "metadata": m.bundle.metadata,
    }


@app.post("/predict", response_model=Prediction)
def predict(application: CreditApplication, request: Request):
    return _predict(request, [application], "predict")[0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    preds = _predict(request, batch.applications, "predict_batch")
    return BatchResponse(n=len(preds), predictions=preds)


@app.post("/explain", response_model=Explanation)
def explain(application: CreditApplication, request: Request, top_k: int = Query(5, ge=1, le=31)):
    m = _model(request)
    result = m.bundle.explain(_frame([application]), top_k=top_k)[0]
    return Explanation(
        request_id=uuid.uuid4().hex,
        probability_of_default=result["probability"],
        default_prediction=bool(result["probability"] >= m.bundle.threshold),
        base_value=result["base_value"],
        contributions=[Contribution(**c) for c in result["contributions"]],
        model_version=m.version,
    )
