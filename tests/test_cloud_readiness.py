"""Cloud readiness: JSON logs, request ids, the S3 model source, and a fail-soft prediction log."""

import io
import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from credit_default.serve.schemas import EXAMPLE

BUCKET = "test-models"
KEY = "credit-default/model.joblib"
S3_URI = f"s3://{BUCKET}/{KEY}"


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("MLFLOW_TRACKING_URI", "MODEL_S3_URI", "MODEL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def make_client(bundle_path, tmp_path, clean_env):
    clean_env.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    clean_env.setenv("PREDICTION_LOG_PATH", str(tmp_path / "pred.sqlite"))
    from credit_default.serve.app import app

    def _make():
        return TestClient(app)

    return _make


@pytest.fixture
def json_capture():
    """Attach the production JSON handler to the root logger, writing to a buffer."""
    from credit_default.serve.logging_setup import json_handler

    buf = io.StringIO()
    handler = json_handler(buf)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    root.removeHandler(handler)
    root.setLevel(old_level)


@pytest.fixture
def s3_bucket(bundle_path, monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    # Fake credentials for moto only; the loader itself uses the default AWS credential chain.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.upload_file(str(bundle_path), BUCKET, KEY)
        yield s3


# ---------------------------------------------------------------- logging


def test_json_formatter_emits_one_parseable_object():
    from credit_default.serve.logging_setup import JsonFormatter

    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "credit_default.test", logging.ERROR, __file__, 1, "x=%d", (3,), sys.exc_info()
        )
    record.request_id = "abc"
    record.latency_ms = 1.5
    line = JsonFormatter().format(record)
    assert "\n" not in line
    entry = json.loads(line)
    assert entry["level"] == "ERROR" and entry["logger"] == "credit_default.test"
    assert entry["message"] == "x=3" and entry["request_id"] == "abc"
    assert entry["latency_ms"] == 1.5 and "ValueError: boom" in entry["exc_info"]
    assert entry["timestamp"].endswith("+00:00")


def test_configure_logging_text_is_default_and_json_is_opt_in(monkeypatch):
    from credit_default.serve import logging_setup

    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    uv = {n: logging.getLogger(n) for n in ("uvicorn", "uvicorn.error", "uvicorn.access")}
    saved_uv = {n: (list(lg.handlers), lg.propagate, lg.disabled) for n, lg in uv.items()}
    try:
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        assert logging_setup.configure_logging() == "text"
        assert not any(getattr(h, "_credit_default_json", False) for h in root.handlers)

        monkeypatch.setenv("LOG_FORMAT", "json")
        assert logging_setup.configure_logging() == "json"
        assert logging_setup.configure_logging() == "json"  # idempotent
        flagged = [h for h in root.handlers if getattr(h, "_credit_default_json", False)]
        assert len(flagged) == 1
        assert isinstance(flagged[0].formatter, logging_setup.JsonFormatter)
        assert uv["uvicorn.error"].propagate and uv["uvicorn.access"].disabled
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for n, (handlers, propagate, disabled) in saved_uv.items():
            uv[n].handlers[:] = handlers
            uv[n].propagate, uv[n].disabled = propagate, disabled


def test_request_line_is_logged_with_fields(make_client, json_capture):
    with make_client() as c:
        r = c.get("/health", headers={"X-Request-ID": "req-123"})
    assert r.status_code == 200
    lines = [e for e in json_capture() if e["logger"] == "credit_default.serve.access"]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["method"] == "GET" and entry["path"] == "/health" and entry["status"] == 200
    assert entry["request_id"] == "req-123" and entry["latency_ms"] >= 0


def test_logs_inside_a_request_carry_its_id(make_client, json_capture):
    with make_client() as c:

        def broken(rows):
            raise RuntimeError("disk full")

        c.app.state.pred_log.log = broken
        r = c.post("/predict", json=EXAMPLE, headers={"X-Request-ID": "trace-1"})
    assert r.status_code == 200  # a failed log write never fails a prediction
    warn = [e for e in json_capture() if "prediction log write failed" in e["message"]]
    assert len(warn) == 1 and warn[0]["request_id"] == "trace-1" and warn[0]["level"] == "WARNING"


def test_request_id_is_propagated_or_generated(make_client):
    with make_client() as c:
        assert c.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == (
            "abc-123"
        )
        generated = c.get("/health").headers["x-request-id"]
        assert len(generated) == 32 and int(generated, 16) >= 0
        # an id that could corrupt a log line is replaced, not echoed
        unsafe = c.get("/health", headers={"X-Request-ID": "a b {}"}).headers["x-request-id"]
        assert unsafe != "a b {}" and len(unsafe) == 32
        # error responses carry it too
        bad = c.post("/predict", json={}, headers={"X-Request-ID": "e-1"})
        assert bad.status_code == 422 and bad.headers["x-request-id"] == "e-1"


# ---------------------------------------------------------------- model source


def test_s3_source_loads_and_is_reported(make_client, s3_bucket, clean_env):
    clean_env.setenv("MODEL_SOURCE", "s3")
    clean_env.setenv("MODEL_S3_URI", S3_URI)
    with make_client() as c:
        health = c.get("/health").json()
        info = c.get("/model").json()
        pred = c.post("/predict", json=EXAMPLE)
    assert health["model_source"] == "s3" and health["model_uri"] == S3_URI
    assert health["model_version"].startswith("s3:")
    assert info["source"] == "s3" and info["uri"] == S3_URI
    assert pred.status_code == 200 and pred.json()["model_source"] == "s3"


def test_auto_prefers_s3_when_uri_is_set(s3_bucket, bundle_path, clean_env):
    from credit_default.serve import model_loader

    clean_env.setenv("MODEL_S3_URI", S3_URI)
    clean_env.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    m = model_loader.load_model()
    assert m.source == "s3" and m.uri == S3_URI and m.bundle is not None


@pytest.mark.parametrize(
    "mode,uri",
    [
        ("s3", f"s3://{BUCKET}/missing/model.joblib"),  # object not found
        ("s3", "s3://no-such-bucket-here/model.joblib"),  # bucket not found
        ("auto", f"s3://{BUCKET}/missing/model.joblib"),
        ("s3", "https://example.com/model.joblib"),  # not an S3 URI
        ("s3", None),  # MODEL_S3_URI unset
    ],
)
def test_s3_failure_falls_back_to_local(s3_bucket, bundle_path, clean_env, caplog, mode, uri):
    from credit_default.serve import model_loader

    clean_env.setenv("MODEL_SOURCE", mode)
    if uri:
        clean_env.setenv("MODEL_S3_URI", uri)
    clean_env.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    with caplog.at_level(logging.WARNING, logger="credit_default.serve.model_loader"):
        m = model_loader.load_model()
    assert m.bundle is not None and m.source == "local" and m.uri == str(bundle_path)
    assert any("falling back to the local bundle" in r.getMessage() for r in caplog.records)


def test_s3_without_boto3_falls_back_to_local(bundle_path, clean_env, monkeypatch, caplog):
    from credit_default.serve import model_loader

    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    clean_env.setenv("MODEL_SOURCE", "s3")
    clean_env.setenv("MODEL_S3_URI", S3_URI)
    clean_env.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    with caplog.at_level(logging.WARNING, logger="credit_default.serve.model_loader"):
        m = model_loader.load_model()
    assert m.source == "local"
    assert any("boto3" in r.getMessage() for r in caplog.records)


def test_s3_client_has_bounded_timeouts():
    pytest.importorskip("boto3")
    from credit_default.serve import model_loader

    cfg = model_loader._s3_client().meta.config
    assert cfg.connect_timeout == 5 and cfg.read_timeout == 5
    assert cfg.retries["total_max_attempts"] == 2


def test_parse_s3_uri():
    from credit_default.serve.model_loader import parse_s3_uri

    assert parse_s3_uri("s3://b/k/model.joblib") == ("b", "k/model.joblib")
    for bad in ("s3://b", "s3://b/", "s3:///k", "s3://b/dir/", "gs://b/k"):
        with pytest.raises(ValueError):
            parse_s3_uri(bad)


def test_local_mode_ignores_s3_and_registry(bundle_path, clean_env, monkeypatch):
    from credit_default.serve import model_loader

    def must_not_run(*_):
        raise AssertionError("local mode must only read the bundle")

    monkeypatch.setattr(model_loader, "_from_s3", must_not_run)
    monkeypatch.setattr(model_loader, "_from_registry", must_not_run)
    clean_env.setenv("MODEL_SOURCE", "local")
    clean_env.setenv("MODEL_S3_URI", S3_URI)
    clean_env.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    clean_env.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    m = model_loader.load_model()
    assert m.source == "local" and m.uri == str(bundle_path) and m.error is None
    assert m.version.startswith("local:")


# ---------------------------------------------------------------- prediction log


def test_unwritable_prediction_log_disables_itself(make_client, tmp_path, clean_env, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the log directory should be")
    clean_env.setenv("PREDICTION_LOG_PATH", str(blocker / "logs" / "pred.sqlite"))
    level = caplog.at_level(logging.WARNING, logger="credit_default.serve.prediction_log")
    with level, make_client() as c:
        assert c.app.state.pred_log.enabled is False
        assert c.post("/predict", json=EXAMPLE).status_code == 200
        assert c.post("/predict", json=EXAMPLE).status_code == 200
    warnings = [r for r in caplog.records if "prediction log disabled" in r.getMessage()]
    assert len(warnings) == 1


# ---------------------------------------------------------------- MLflow artifact location


def test_experiment_artifact_location():
    from credit_default.config import REPO_ROOT
    from credit_default.registry.mlflow_registry import experiment_artifact_location as loc

    assert loc("sqlite:////x/mlflow.db", "s3://bucket/mlflow") == "s3://bucket/mlflow"
    assert loc("http://mlflow:5000", "s3://bucket/mlflow") == "s3://bucket/mlflow"
    assert loc("sqlite:////x/mlflow.db", "mlruns/artifacts") == (
        (REPO_ROOT / "mlruns/artifacts").resolve().as_uri()
    )
    assert loc("http://mlflow:5000", "mlruns/artifacts") is None


def test_artifact_location_env_override(monkeypatch):
    from credit_default.config import load_config

    monkeypatch.delenv("MLFLOW_ARTIFACT_LOCATION", raising=False)
    assert load_config([]).registry.artifact_location == "mlruns/artifacts"
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", "s3://bucket/mlflow")
    assert load_config([]).registry.artifact_location == "s3://bucket/mlflow"
