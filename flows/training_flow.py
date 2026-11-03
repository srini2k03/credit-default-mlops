"""Prefect flow: ingest -> validate -> train -> evaluate (quality gate) -> register (promote).

Runs locally without a Prefect server (Prefect starts a temporary local API), or against
a server/Cloud workspace if PREFECT_API_URL is set.

  python flows/training_flow.py data=synthetic train=fast
  python flows/training_flow.py                      # real data, full search
"""

from __future__ import annotations

import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

from credit_default.config import load_config, split_cli
from credit_default.pipeline import steps
from credit_default.pipeline.train import train as train_models
from credit_default.registry import gate as gate_mod
from credit_default.registry import mlflow_registry as reg


@task(retries=2, retry_delay_seconds=10)
def ingest(cfg) -> Path:
    return steps.ingest(cfg)


@task
def validate(cfg, path: Path):
    return steps.validate(cfg, path), steps.data_fingerprint(cfg, path)


@task
def train(cfg, df, data_info) -> dict:
    result = train_models(cfg, df, data_info, register=True)
    return {
        "version": result.model_version,
        "run_id": result.run_id,
        "champion": result.champion.family,
        "test": result.champion.test_metrics,
    }


@task
def evaluate(cfg, version: str) -> gate_mod.GateDecision:
    reg.setup_tracking(cfg)
    challenger = reg.load_bundle_from_uri(f"models:/{cfg.registry.model_name}/{version}")
    production = reg.load_alias(cfg, cfg.registry.production_alias)
    decision = gate_mod.run_gate(cfg, challenger, production[0] if production else None)
    out = steps.REPO_ROOT / cfg.paths.artifacts_dir / "run"
    gate_mod.write_report(
        decision,
        out,
        {
            "challenger_ref": f"models:/{cfg.registry.model_name}/{version}",
            "production_ref": f"v{production[1]}" if production else "none",
        },
    )
    return decision


@task
def register(cfg, version: str, decision: gate_mod.GateDecision) -> str:
    logger = get_run_logger()
    if not decision.passed:
        logger.warning("gate rejected v%s: %s", version, "; ".join(decision.reasons))
        return "rejected"
    reg.setup_tracking(cfg)
    reg.set_alias(cfg, version, cfg.registry.production_alias)
    logger.info("v%s promoted to @%s", version, cfg.registry.production_alias)
    return "promoted"


@flow(name="credit-default-training", log_prints=True)
def training_flow(overrides: list[str] | None = None) -> dict:
    cfg = load_config(overrides or [])
    path = ingest(cfg)
    df, info = validate(cfg, path)
    trained = train(cfg, df, info)
    decision = evaluate(cfg, trained["version"])
    outcome = register(cfg, trained["version"], decision)
    return {**trained, "gate_passed": decision.passed, "outcome": outcome}


if __name__ == "__main__":
    overrides, _ = split_cli(sys.argv[1:])
    print(training_flow(overrides))
