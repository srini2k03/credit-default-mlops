import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_default.serve.schemas import EXAMPLE


@pytest.fixture
def client(bundle_path, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MODEL_SOURCE", "auto")
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "pred.sqlite"))
    from credit_default.serve.app import app

    with TestClient(app) as c:
        c.log_path = tmp_path / "pred.sqlite"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_source"] == "local" and 0 < body["threshold"] < 1


def test_predict_and_log(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] == ("decline" if body["default_prediction"] else "approve")
    with sqlite3.connect(client.log_path) as con:
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1


def test_predict_batch(client):
    r = client.post("/predict_batch", json={"applications": [EXAMPLE, {**EXAMPLE, "PAY_0": -1}]})
    assert r.status_code == 200
    assert r.json()["n"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        {**EXAMPLE, "SEX": 3},
        {**EXAMPLE, "PAY_0": 12},
        {**EXAMPLE, "AGE": "old"},
        {k: v for k, v in EXAMPLE.items() if k != "LIMIT_BAL"},
        {**EXAMPLE, "unexpected": 1},
    ],
)
def test_invalid_payloads_are_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_limits(client):
    assert client.post("/predict_batch", json={"applications": []}).status_code == 422


def test_explain(client):
    r = client.post("/explain?top_k=3", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) == 3
    assert {"feature", "value", "shap_value"} <= set(body["contributions"][0])


def test_unreachable_registry_falls_back_to_local(bundle_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    from credit_default.serve.model_loader import load_model

    m = load_model()
    assert m.bundle is not None and m.source == "local"


def test_registry_load_by_alias(trained, cfg, workdir, bundle_path, monkeypatch):
    from credit_default.registry import mlflow_registry as reg

    reg.setup_tracking(cfg)
    reg.set_alias(cfg, trained.model_version, cfg.registry.production_alias)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{workdir}/mlflow.db")
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(workdir / "missing.joblib"))
    from credit_default.serve.model_loader import load_model

    m = load_model()
    assert m.source == "registry" and m.version == trained.model_version


def test_no_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("PREDICTION_LOG_ENABLED", "false")
    from credit_default.serve.app import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=EXAMPLE).status_code == 503


def test_request_id_is_echoed_or_generated(client):
    r = client.post("/predict", json=EXAMPLE, headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.json()["request_id"] == "abc-123"
    r = client.post("/predict", json=EXAMPLE, headers={"x-request-id": "bad id with spaces"})
    assert len(r.headers["x-request-id"]) == 32  # replaced by a generated uuid4 hex
    batch = client.post("/predict_batch", json={"applications": [EXAMPLE, EXAMPLE]})
    rid = batch.headers["x-request-id"]
    assert [p["request_id"] for p in batch.json()["predictions"]] == [f"{rid}-0", f"{rid}-1"]


def test_errors_carry_request_id(client):
    r = client.post("/predict", json={**EXAMPLE, "SEX": 3})
    assert r.status_code == 422 and r.json()["request_id"] == r.headers["x-request-id"]
    assert isinstance(r.json()["detail"], list)  # FastAPI's default 422 shape is kept


def test_ready_and_model_metadata(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] and r.json()["fast_scoring"]
    meta = client.get("/model").json()
    assert meta["calibration"] in {"none", "sigmoid", "isotonic"}
    assert meta["excluded_features"] == ["SEX", "MARRIAGE", "AGE"]
    assert "not an approved" in meta["disclaimer"]


def test_threshold_and_cost_parameters(client):
    base = client.post("/predict", json=EXAMPLE).json()
    assert base["threshold_source"] == "policy" and base["threshold"] == base["policy_threshold"]
    low = client.post("/predict?threshold=0.01", json=EXAMPLE).json()
    assert low["decision"] == "decline" and low["threshold_source"] == "request threshold"
    costs = client.post("/predict?fn_cost=9&fp_cost=1", json=EXAMPLE).json()
    assert costs["threshold"] == pytest.approx(0.1) and costs["threshold_source"] == "request costs"
    assert low["probability_of_default"] == base["probability_of_default"]
    assert (
        client.post("/predict?threshold=0.2&fn_cost=5&fp_cost=1", json=EXAMPLE).status_code == 422
    )
    assert client.post("/predict?fn_cost=5", json=EXAMPLE).status_code == 422
    assert client.post("/predict?threshold=1.5", json=EXAMPLE).status_code == 422


def test_explain_reports_how_the_score_was_reached(client):
    body = client.post("/explain?top_k=31", json=EXAMPLE).json()
    total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert total == pytest.approx(body["raw_score"], abs=1e-6)
    assert body["other_features_contribution"] == pytest.approx(0.0, abs=1e-9)
    assert {c["direction"] for c in body["contributions"]} <= {"increases risk", "decreases risk"}
    assert body["decision"] in {"approve", "decline"}


def test_fast_path_matches_pipeline_exactly(client, bundle_path, fixture_df):
    import numpy as np

    from credit_default.data.schema import FEATURE_COLUMNS
    from credit_default.models.bundle import ModelBundle
    from credit_default.serve.fast import build_fast_scorer

    bundle = ModelBundle.load(bundle_path)
    X = fixture_df[FEATURE_COLUMNS]
    scorer = build_fast_scorer(bundle, X.head(100))
    assert scorer is not None
    np.testing.assert_array_equal(scorer.predict_proba(X.to_numpy(float)), bundle.predict_proba(X))
    api = client.post("/predict", json=EXAMPLE).json()["probability_of_default"]
    assert api == float(bundle.predict_proba(pd.DataFrame([EXAMPLE]))[0])


def test_cors_allows_the_dashboard_only(client):
    ok = client.options(
        "/predict",
        headers={"origin": "https://srini2k03.github.io", "access-control-request-method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://srini2k03.github.io"
    other = client.options(
        "/predict",
        headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in other.headers


def test_rate_limit_and_body_size(bundle_path, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    monkeypatch.setenv("PREDICTION_LOG_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("MAX_BODY_BYTES", "5000")
    from credit_default.serve.app import app

    with TestClient(app) as c:
        codes = [c.post("/predict", json=EXAMPLE).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert c.get("/health").status_code == 200  # health checks are never limited
        big = {"applications": [EXAMPLE] * 50}
        r = c.post("/predict_batch", json=big)
        assert r.status_code == 413 and "request_id" in r.json()


def test_rate_limiter_window():
    from credit_default.serve.app import RateLimiter

    rl = RateLimiter(2)
    assert rl.allow("a", 0.0)[0] and rl.allow("a", 1.0)[0]
    allowed, retry = rl.allow("a", 2.0)
    assert not allowed and retry > 0
    assert rl.allow("b", 2.0)[0]  # per client
    assert rl.allow("a", 61.0)[0]  # the window slides
    assert RateLimiter(0).allow("a")[0]  # disabled
