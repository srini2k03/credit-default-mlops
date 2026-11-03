"""Shared fixtures. Everything runs on the synthetic fixture inside a temp dir: no network."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from credit_default.config import load_config  # noqa: E402
from credit_default.data.synthetic import generate, write_fixture  # noqa: E402


def overrides_for(tmp) -> list[str]:
    return [
        "data=synthetic",
        "train=fast",
        "data.n_rows=3000",
        f"data.raw_path={tmp}/data/credit_default.csv",
        f"paths.artifacts_dir={tmp}/artifacts",
        f"paths.results_dir={tmp}/results",
        f"paths.bundle_dir={tmp}/bundled",
        f"paths.slices_dir={tmp}/slices",
        f"paths.reports_dir={tmp}/reports",
        f"paths.prediction_log={tmp}/logs/predictions.sqlite",
        f"registry.tracking_uri='sqlite:///{tmp}/mlflow.db'",
        f"registry.artifact_location={tmp}/mlartifacts",
        "validation.min_rows=500",
        "monitor.slice_rows=400",
    ]


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("cdm")


@pytest.fixture(scope="session")
def cfg(workdir):
    cfg = load_config(overrides_for(workdir))
    write_fixture(cfg)
    return cfg


@pytest.fixture(scope="session")
def fixture_df(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data.raw_path)


@pytest.fixture(scope="session")
def trained(cfg, fixture_df):
    """One fast training run (LightGBM, no tuning) registered as @staging in a temp registry."""
    from pathlib import Path

    from credit_default.pipeline.steps import data_fingerprint
    from credit_default.pipeline.train import train

    return train(cfg, fixture_df, data_fingerprint(cfg, Path(cfg.data.raw_path)), register=True)


@pytest.fixture(scope="session")
def bundle_path(trained, workdir):
    path = workdir / "good" / "model.joblib"
    trained.champion.bundle.save(path)
    return path


@pytest.fixture
def small_df():
    return generate(n_rows=1500, seed=3)
