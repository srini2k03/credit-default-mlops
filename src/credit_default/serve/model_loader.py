"""Load the serving model: MLflow registry alias first, bundled local artifact as fallback.

Environment:
  MODEL_SOURCE         auto (default) | registry | local
  MLFLOW_TRACKING_URI  registry location; if unset, ``auto`` goes straight to the local bundle
  MODEL_URI            default models:/credit-default@production
  MODEL_BUNDLE_PATH    default models/bundled/model.joblib
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
    errors = []

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
