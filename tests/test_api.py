import sqlite3

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
