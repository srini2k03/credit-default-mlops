"""Load the serving model from S3, the MLflow registry or the bundled local artifact.

Environment:
  MODEL_SOURCE         auto (default) | s3 | registry | local
  MODEL_S3_URI         s3://bucket/key/model.joblib; ``auto`` tries it first when set
  MLFLOW_TRACKING_URI  registry location; ``auto`` tries the registry only when it is set
  MODEL_URI            default models:/credit-default@production
  MODEL_BUNDLE_PATH    default models/bundled/model.joblib

``auto`` order: S3 (if MODEL_S3_URI is set), then the registry (if MLFLOW_TRACKING_URI is set),
then the local bundle. ``s3`` and ``auto`` fall back to the local bundle on any failure and log
why; ``registry`` does not fall back. ``local`` only reads the bundle.

S3 credentials come from the default AWS credential chain (on ECS, the task role); this module
never reads keys itself. boto3 (extra ``aws``) is imported only when S3 is used, so images and
hosts without it keep working. The object is unpickled by joblib, so the bucket must be trusted.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from credit_default.models.bundle import BUNDLE_FILENAME, ModelBundle

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
S3_TIMEOUT_S = 5


@dataclass
class LoadedModel:
    bundle: ModelBundle | None
    source: str | None = None
    version: str | None = None
    uri: str | None = None
    error: str | None = None
    loaded_at: float = 0.0


def _from_registry(model_uri: str) -> tuple[ModelBundle, str]:
    # Fail fast when the server is unreachable: one try, short timeout.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    import mlflow
    from mlflow.tracking import MlflowClient

    version = "unknown"
    if "@" in model_uri:
        name, alias = model_uri.removeprefix("models:/").split("@", 1)
        version = str(MlflowClient().get_model_version_by_alias(name, alias).version)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        found = list(Path(local).rglob(BUNDLE_FILENAME))
        if not found:
            raise FileNotFoundError(f"{BUNDLE_FILENAME} not found in {model_uri}")
        return ModelBundle.load(found[0]), version


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key or key.endswith("/"):
        raise ValueError(f"S3 URI must name an object, s3://bucket/key: {uri!r}")
    return bucket, key


def _s3_client():
    import boto3
    from botocore.config import Config

    # Fail fast like the registry path: short timeouts, one retry (two attempts in total).
    config = Config(
        connect_timeout=S3_TIMEOUT_S,
        read_timeout=S3_TIMEOUT_S,
        retries={"total_max_attempts": 2, "mode": "standard"},
    )
    return boto3.client("s3", config=config)


def _from_s3(uri: str) -> tuple[ModelBundle, str]:
    bucket, key = parse_s3_uri(uri)
    resp = _s3_client().get_object(Bucket=bucket, Key=key)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / BUNDLE_FILENAME
        with open(path, "wb") as fh:
            for chunk in resp["Body"].iter_chunks(chunk_size=1 << 20):
                fh.write(chunk)
        bundle = ModelBundle.load(path)
    log.info(
        "downloaded %s (etag %s, version id %s)",
        uri,
        str(resp.get("ETag", "")).strip('"') or "n/a",
        resp.get("VersionId") or "n/a",
    )
    version = f"s3:{str(bundle.metadata.get('git_commit', 'unknown'))[:10]}"
    return bundle, version


def _from_local(path: Path) -> tuple[ModelBundle, str]:
    bundle = ModelBundle.load(path)
    version = f"local:{str(bundle.metadata.get('git_commit', 'unknown'))[:10]}"
    return bundle, version


def load_model() -> LoadedModel:
    source = os.environ.get("MODEL_SOURCE", "auto").lower()
    model_uri = os.environ.get("MODEL_URI", "models:/credit-default@production")
    local_path = Path(
        os.environ.get("MODEL_BUNDLE_PATH", REPO_ROOT / "models/bundled" / BUNDLE_FILENAME)
    )
    s3_uri = os.environ.get("MODEL_S3_URI", "").strip()
    errors = []

    if source == "s3" and not s3_uri:
        log.warning("MODEL_SOURCE=s3 but MODEL_S3_URI is not set; falling back to the local bundle")
        errors.append("s3: MODEL_S3_URI is not set")
    elif source in {"auto", "s3"} and s3_uri:
        try:
            bundle, version = _from_s3(s3_uri)
            log.info("loaded %s (version %s) from S3", s3_uri, version)
            return LoadedModel(bundle, "s3", version, s3_uri, loaded_at=time.time())
        except Exception as exc:
            nxt = (
                "the registry"
                if source == "auto" and os.environ.get("MLFLOW_TRACKING_URI")
                else "the local bundle"
            )
            log.warning(
                "S3 load from %s failed (%s: %s); falling back to %s",
                s3_uri,
                type(exc).__name__,
                exc,
                nxt,
            )
            errors.append(f"s3: {exc}")

    if source in {"auto", "registry"} and (
        source == "registry" or os.environ.get("MLFLOW_TRACKING_URI")
    ):
        try:
            bundle, version = _from_registry(model_uri)
            log.info("loaded %s (version %s) from the registry", model_uri, version)
            return LoadedModel(bundle, "registry", version, model_uri, loaded_at=time.time())
        except Exception as exc:
            log.warning("registry load failed (%s); falling back to the local bundle", exc)
            errors.append(f"registry: {exc}")
            if source == "registry":
                return LoadedModel(None, error="; ".join(errors))

    try:
        bundle, version = _from_local(local_path)
        log.info("loaded local bundle %s", local_path)
        return LoadedModel(bundle, "local", version, str(local_path), loaded_at=time.time())
    except Exception as exc:
        errors.append(f"local: {exc}")
        log.error("no model could be loaded: %s", "; ".join(errors))
        return LoadedModel(None, error="; ".join(errors))
