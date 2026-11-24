"""FastAPI app. Run: uvicorn credit_default.serve.app:app --host 0.0.0.0 --port 8000

Endpoints (all backward compatible with the first release; new fields are additive):
  GET  /health          liveness + model summary (200, or 503 without a model)
  GET  /ready           readiness: model loaded, fast path verified, explainer warmed up
  GET  /model           metadata of the served model (version, data checksum, test metrics)
  POST /predict         one application -> calibrated probability + approve/decline
  POST /predict_batch   up to 1,000 applications
  POST /explain         probability + ranked SHAP contributions

Hardening: request IDs (X-Request-ID in and out), JSON access logs without applicant data,
request-size limit (413), per-client rate limit (429), JSON errors with the request ID, CORS
for the project dashboard only. Settings come from environment variables (see .env.example).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from credit_default import __version__
from credit_default.serve.fast import FastScorer, build_fast_scorer
from credit_default.serve.model_loader import REPO_ROOT, LoadedModel, load_model
from credit_default.serve.prediction_log import PredictionLogger
from credit_default.serve.schemas import (
    EXAMPLE,
    LOW_RISK_EXAMPLE,
    OPENAPI_BATCH_EXAMPLES,
    OPENAPI_EXAMPLES,
    BatchRequest,
    BatchResponse,
    Contribution,
    CreditApplication,
    Explanation,
    Health,
    Prediction,
    Ready,
)

DISCLAIMER = (
    "Educational decision-support demo on a public 2005 dataset; not an approved lending or "
    "credit-scoring system."
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={"fields": {...}}`` adds structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(out, default=str)


log = logging.getLogger("credit_default.serve")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        JsonFormatter()
        if os.environ.get("LOG_FORMAT", "json") == "json"
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(_handler)
    log.propagate = False
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
UNLIMITED_PATHS = {"/health", "/ready", "/", "/docs", "/openapi.json", "/redoc"}


class RateLimiter:
    """Sliding one-minute window per client key. ``limit <= 0`` disables it."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(60.0 - (now - q[0])) + 1
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory: forget idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


def _client_key(request: Request) -> str:
    # Render terminates TLS in a proxy and appends the caller to X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _warm_up(app: FastAPI) -> None:
    m: LoadedModel = app.state.model
    app.state.fast = None
    app.state.explainer_ready = False
    if m.bundle is None:
        return
    verify = pd.concat(
        [m.bundle.background, pd.DataFrame([EXAMPLE, LOW_RISK_EXAMPLE])], ignore_index=True
    )
    if os.environ.get("FAST_SCORING", "true").lower() == "true":
        app.state.fast = build_fast_scorer(m.bundle, verify)
    if os.environ.get("EXPLAINER_WARMUP", "true").lower() == "true":
        try:
            m.bundle.explain(pd.DataFrame([EXAMPLE]), top_k=1)
            app.state.explainer_ready = True
        except Exception as exc:  # explanations are optional; predictions still work
            log.warning("explainer warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    app.state.model = load_model()
    app.state.pred_log = PredictionLogger(
        os.environ.get("PREDICTION_LOG_PATH", REPO_ROOT / "logs" / "predictions.sqlite"),
        enabled=os.environ.get("PREDICTION_LOG_ENABLED", "true").lower() == "true",
    )
    app.state.limiter = RateLimiter(_env_int("RATE_LIMIT_PER_MINUTE", 120))
    app.state.max_body = _env_int("MAX_BODY_BYTES", 2_000_000)
    _warm_up(app)
    app.state.started = time.time()
    log.info(
        "startup complete",
        extra={
            "fields": {
                "model_version": app.state.model.version,
                "model_source": app.state.model.source,
                "fast_scoring": app.state.fast is not None,
                "explainer_ready": app.state.explainer_ready,
                "startup_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        },
    )
    yield


app = FastAPI(
    title="Credit default risk API",
    version=__version__,
    description=(
        "Calibrated probability of credit-card default next month, a cost-optimal approve/decline "
        "decision, and per-request SHAP explanations. Model: `models:/credit-default@production` "
        "from the MLflow registry, or the bundled CI-trained artifact when no registry is reachable.\n\n"
        "**Try it:** open `POST /predict`, click **Try it out**, pick an example applicant from the "
        "*Examples* dropdown (or edit the JSON body), then click **Execute**. The response appears "
        "under *Responses* below the button.\n\n"
        "**Hosted demo note:** the free hosting tier sleeps when idle, so the first request after a "
        "pause can take 30-60 s while the service wakes up. Later requests return in well under a "
        "second.\n\n"
        "Optional query parameters on the prediction endpoints: `threshold` (a cutoff in (0, 1)) or "
        "`fn_cost` and `fp_cost` (the Bayes cutoff fp/(fp+fn) for calibrated probabilities). "
        "Requests are limited per client per minute (HTTP 429) and in size (HTTP 413); every "
        "response carries an `X-Request-ID` header.\n\n" + DISCLAIMER
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "https://srini2k03.github.io").split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-request-id"],
    expose_headers=["x-request-id"],
    max_age=600,
)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def _error(request: Request, status: int, detail, headers: dict | None = None) -> JSONResponse:
    rid = _rid(request)
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "request_id": rid},
        headers={"x-request-id": rid, **(headers or {})},
    )


@app.middleware("http")
async def hardening(request: Request, call_next):
    t0 = time.perf_counter()
    incoming = request.headers.get("x-request-id", "")
    request.state.request_id = incoming if REQUEST_ID.match(incoming) else uuid.uuid4().hex
    path = request.url.path
    status = 500
    try:
        if request.method == "POST":
            length = request.headers.get("content-length")
            limit = request.app.state.max_body
            if length is not None and length.isdigit() and int(length) > limit:
                status = 413
                return _error(request, 413, f"request body larger than {limit} bytes")
            if length is None and len(await request.body()) > limit:
                status = 413
                return _error(request, 413, f"request body larger than {limit} bytes")
        if path not in UNLIMITED_PATHS and request.method != "OPTIONS":
            ok, retry = request.app.state.limiter.allow(_client_key(request))
            if not ok:
                status = 429
                return _error(request, 429, "rate limit exceeded", {"retry-after": str(retry)})
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request.state.request_id
        return response
    finally:
        model = getattr(request.app.state, "model", None)
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "model_version": getattr(model, "version", None),
                }
            },
        )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Same 422 body as FastAPI's default ("detail": [...]) plus the request ID.
    return _error(request, 422, json.loads(json.dumps(exc.errors(), default=str)))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return _error(request, exc.status_code, exc.detail)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.error(
        "unhandled error",
        exc_info=exc,
        extra={"fields": {"request_id": _rid(request), "path": request.url.path}},
    )
    return _error(request, 500, "internal error; quote the request_id when reporting it")


# ---------------------------------------------------------------------------- helpers
def _model(request: Request) -> LoadedModel:
    m: LoadedModel = request.app.state.model
    if m.bundle is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {m.error}")
    return m


def _frame(apps: list[CreditApplication]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in apps])


def _cutoff(
    m: LoadedModel, threshold: float | None, fn_cost: float | None, fp_cost: float | None
) -> tuple[float, str]:
    if threshold is not None and (fn_cost is not None or fp_cost is not None):
        raise HTTPException(422, "pass either threshold or fn_cost and fp_cost, not both")
    if threshold is not None:
        return float(threshold), "request threshold"
    if (fn_cost is None) != (fp_cost is None):
        raise HTTPException(422, "fn_cost and fp_cost must be given together")
    if fn_cost is not None:
        return float(fp_cost / (fp_cost + fn_cost)), "request costs"
    return float(m.bundle.threshold), "policy"


def _proba(request: Request, m: LoadedModel, apps: list[CreditApplication]) -> np.ndarray:
    fast: FastScorer | None = request.app.state.fast
    rows = [a.model_dump() for a in apps]
    if fast is not None:
        return fast.predict_proba(FastScorer.rows_to_array(rows))
    return m.bundle.predict_proba(pd.DataFrame(rows))


def _predict(
    request: Request,
    apps: list[CreditApplication],
    endpoint: str,
    cutoff: tuple[float, str],
) -> list[Prediction]:
    m = _model(request)
    proba = _proba(request, m, apps)
    threshold, source = cutoff
    base = _rid(request)
    out, rows = [], []
    for i, (app_, p) in enumerate(zip(apps, proba, strict=True)):
        rid = base if len(apps) == 1 else f"{base}-{i}"
        decline = bool(p >= threshold)
        out.append(
            Prediction(
                request_id=rid,
                probability_of_default=float(p),
                default_prediction=decline,
                decision="decline" if decline else "approve",
                threshold=threshold,
                model_version=m.version,
                model_source=m.source,
                threshold_source=source,
                policy_threshold=float(m.bundle.threshold),
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
                "threshold": threshold,
                "features": app_.model_dump(),
            }
        )
    try:
        request.app.state.pred_log.log(rows)
    except Exception as exc:  # logging must never fail a prediction
        log.warning("prediction log write failed: %s", exc)
    return out


ThresholdQ = Annotated[
    float | None,
    Query(gt=0, lt=1, description="Optional cutoff; decline when P(default) >= threshold"),
]
FnCostQ = Annotated[
    float | None, Query(gt=0, le=1000, description="Cost of approving a defaulter (with fp_cost)")
]
FpCostQ = Annotated[
    float | None, Query(gt=0, le=1000, description="Cost of declining a good customer")
]


# ---------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


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


@app.get("/ready", response_model=Ready)
def ready(request: Request):
    m: LoadedModel = request.app.state.model
    body = Ready(
        ready=m.bundle is not None,
        model_version=m.version,
        fast_scoring=request.app.state.fast is not None,
        explainer_ready=request.app.state.explainer_ready,
        detail=None if m.bundle is not None else m.error,
    )
    return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())


@app.get("/model")
def model_info(request: Request):
    m = _model(request)
    meta = m.bundle.metadata
    return {
        "source": m.source,
        "version": m.version,
        "uri": m.uri,
        "family": m.bundle.family,
        "threshold": m.bundle.threshold,
        "calibration": getattr(m.bundle, "calibration", "isotonic"),
        "excluded_features": meta.get("excluded_features", []),
        "fast_scoring": request.app.state.fast is not None,
        "explanation_space": m.bundle.explanation_space,
        "loaded_at": m.loaded_at,
        "disclaimer": DISCLAIMER,
        "metadata": meta,
    }


@app.post("/predict", response_model=Prediction)
def predict(
    application: Annotated[CreditApplication, Body(openapi_examples=OPENAPI_EXAMPLES)],
    request: Request,
    threshold: ThresholdQ = None,
    fn_cost: FnCostQ = None,
    fp_cost: FpCostQ = None,
):
    m = _model(request)
    return _predict(request, [application], "predict", _cutoff(m, threshold, fn_cost, fp_cost))[0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(
    batch: Annotated[BatchRequest, Body(openapi_examples=OPENAPI_BATCH_EXAMPLES)],
    request: Request,
    threshold: ThresholdQ = None,
    fn_cost: FnCostQ = None,
    fp_cost: FpCostQ = None,
):
    m = _model(request)
    cutoff = _cutoff(m, threshold, fn_cost, fp_cost)
    preds = _predict(request, batch.applications, "predict_batch", cutoff)
    return BatchResponse(n=len(preds), predictions=preds)


@app.post("/explain", response_model=Explanation)
def explain(
    application: Annotated[CreditApplication, Body(openapi_examples=OPENAPI_EXAMPLES)],
    request: Request,
    top_k: int = Query(5, ge=1, le=31, description="Number of top SHAP contributions to return"),
    threshold: ThresholdQ = None,
    fn_cost: FnCostQ = None,
    fp_cost: FpCostQ = None,
):
    m = _model(request)
    cutoff, _ = _cutoff(m, threshold, fn_cost, fp_cost)
    result = m.bundle.explain(_frame([application]), top_k=top_k)[0]
    request.app.state.explainer_ready = True
    decline = bool(result["probability"] >= cutoff)
    space = m.bundle.explanation_space
    return Explanation(
        request_id=_rid(request),
        probability_of_default=result["probability"],
        default_prediction=decline,
        base_value=result["base_value"],
        contributions=[Contribution(**c) for c in result["contributions"]],
        explanation_space=f"SHAP values in the base model's {space} (before calibration)",
        model_version=m.version,
        decision="decline" if decline else "approve",
        threshold=cutoff,
        raw_score=result["raw_score"],
        uncalibrated_probability=result["uncalibrated_probability"],
        other_features_contribution=result["other_features_contribution"],
        calibration=getattr(m.bundle, "calibration", "isotonic"),
    )
