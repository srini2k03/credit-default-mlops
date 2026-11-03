"""Pipeline steps as plain functions. The CLI and the Prefect flow both call these."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_default.config import REPO_ROOT
from credit_default.data.download import sha256_file
from credit_default.validation.schema import validate as validate_df

log = logging.getLogger(__name__)


def ingest(cfg) -> Path:
    """Fetch the configured data source (real UCI download or synthetic fixture)."""
    if cfg.data.source == "synthetic":
        from credit_default.data.synthetic import write_fixture

        path = REPO_ROOT / cfg.data.raw_path
        return path if path.exists() else write_fixture(cfg)
    from credit_default.data.download import download

    return download(cfg)


def validate(cfg, path: Path) -> pd.DataFrame:
    return validate_df(pd.read_csv(path), cfg)


def git_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def data_fingerprint(cfg, path: Path) -> dict:
    return {
        "source": cfg.data.source,
        "path": str(cfg.data.raw_path),
        "sha256": sha256_file(path),
        "synthetic": cfg.data.source == "synthetic",
    }


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path
