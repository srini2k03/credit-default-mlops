"""Register the committed bundle as @production in an empty (e.g. ephemeral CI) registry.

CI runners start with an empty MLflow store, so the git repository is the durable record of what
is in production: ``models/bundled/model.joblib``. This step re-creates that state in the registry
so the gate can compare a new challenger against it.

Usage: python -m credit_default.registry.bootstrap [--bundle=models/bundled/model.joblib]
"""

from __future__ import annotations

import logging
import sys

import mlflow

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.models.bundle import ModelBundle
from credit_default.registry import mlflow_registry as reg

log = logging.getLogger(__name__)


def bootstrap(cfg, bundle_path) -> str | None:
    reg.setup_tracking(cfg)
    if reg.get_version(cfg, cfg.registry.production_alias) is not None:
        log.info("registry already has @%s; nothing to do", cfg.registry.production_alias)
        return None
    if not bundle_path.exists():
        log.info("no bundled model at %s; registry stays empty (first run)", bundle_path)
        return None
    bundle = ModelBundle.load(bundle_path)
    with mlflow.start_run(run_name="bootstrap-production"):
        mlflow.set_tags(
            {
                "bootstrap": "true",
                "source_bundle": str(bundle_path),
                "bundle_git_commit": bundle.metadata.get("git_commit", "unknown"),
            }
        )
        info = reg.log_bundle(bundle, cfg.registry.model_name)
    version = str(info.registered_model_version)
    reg.set_alias(cfg, version, cfg.registry.production_alias)
    return version


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    path = next((f.split("=", 1)[1] for f in flags if f.startswith("--bundle=")), None)
    bundle_path = REPO_ROOT / (path or f"{cfg.paths.bundle_dir}/model.joblib")
    bootstrap(cfg, bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
