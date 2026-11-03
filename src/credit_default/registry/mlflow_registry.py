"""MLflow tracking and Model Registry helpers.

Aliases replace the deprecated registry "stages":
  Staging    -> alias ``staging``     (newly trained challenger)
  Production -> alias ``production``  (what the API serves: models:/credit-default@production)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from credit_default.config import REPO_ROOT
from credit_default.models.bundle import BUNDLE_FILENAME, ModelBundle

log = logging.getLogger(__name__)

PIP_REQUIREMENTS = [
    "credit-default-mlops",
    "scikit-learn",
    "imbalanced-learn",
    "lightgbm",
    "xgboost",
    "shap",
    "pandas",
    "joblib",
]


def _resolve_tracking_uri(uri: str) -> str:
    """Make relative sqlite paths absolute (relative to the repo root) and create parent dirs."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        path = REPO_ROOT / uri[len(prefix) :]
        path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + str(path)
    return uri


def setup_tracking(cfg) -> str:
    uri = _resolve_tracking_uri(cfg.registry.tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(cfg.registry.experiment) is None:
        artifact_location = None
        if uri.startswith("sqlite:"):  # local store: keep artifacts next to the db
            artifact_location = (REPO_ROOT / cfg.registry.artifact_location).resolve().as_uri()
        mlflow.create_experiment(cfg.registry.experiment, artifact_location=artifact_location)
    mlflow.set_experiment(cfg.registry.experiment)
    return uri


def log_bundle(bundle: ModelBundle, registered_name: str | None = None):
    """Log the bundle as a pyfunc model in the active run; optionally register it."""
    from credit_default.registry.pyfunc import CreditDefaultPyfunc

    with tempfile.TemporaryDirectory() as tmp:
        path = bundle.save(Path(tmp) / BUNDLE_FILENAME)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CreditDefaultPyfunc(),
            artifacts={"bundle": str(path)},
            pip_requirements=PIP_REQUIREMENTS,
            registered_model_name=registered_name,
        )
    return info


def set_alias(cfg, version: str | int, alias: str) -> None:
    MlflowClient().set_registered_model_alias(cfg.registry.model_name, alias, str(version))
    log.info("alias %s -> %s v%s", alias, cfg.registry.model_name, version)


def get_version(cfg, alias: str):
    """Return the ModelVersion behind an alias, or None if the alias or model does not exist."""
    try:
        return MlflowClient().get_model_version_by_alias(cfg.registry.model_name, alias)
    except mlflow.exceptions.MlflowException:
        return None


def load_bundle_from_uri(model_uri: str) -> ModelBundle:
    """Download a registered pyfunc model's artifacts and load the bundle inside it."""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        candidates = list(Path(local).rglob(BUNDLE_FILENAME))
        if not candidates:
            raise FileNotFoundError(f"no {BUNDLE_FILENAME} under {model_uri}")
        return ModelBundle.load(candidates[0])


def load_alias(cfg, alias: str) -> tuple[ModelBundle, str] | None:
    mv = get_version(cfg, alias)
    if mv is None:
        return None
    bundle = load_bundle_from_uri(f"models:/{cfg.registry.model_name}@{alias}")
    return bundle, str(mv.version)
