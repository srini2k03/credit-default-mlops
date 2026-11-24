"""Copy a real-data run into the committed locations.

Gate passed (the challenger is the new production model):
  artifacts/run/{metrics, gate, decision, fairness}.{json,md}, evaluation.json,
  predictions.json, figures/                   ->  results/
  artifacts/run/model.joblib                   ->  models/bundled/model.joblib (+ metadata.json)
  MLflow experiment runs                       ->  results/mlflow_summary.md
Gate failed (``--rejected``): the challenger's report goes to results/challenger/ so the failed
experiment stays visible, and production files are not touched.

Refuses to publish a synthetic-fixture run: fixture metrics must never appear as results.

Usage: python -m credit_default.pipeline.publish [overrides] [--rejected]
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.models.bundle import ModelBundle
from credit_default.pipeline.steps import write_json

log = logging.getLogger(__name__)


class SyntheticPublishError(RuntimeError):
    pass


def mlflow_summary(cfg, out: Path) -> Path | None:
    try:
        import mlflow

        from credit_default.registry import mlflow_registry as reg

        reg.setup_tracking(cfg)
        runs = mlflow.search_runs(experiment_names=[cfg.registry.experiment], max_results=200)
    except Exception as exc:
        log.warning("MLflow summary skipped: %s", exc)
        return None
    cols = [
        c
        for c in [
            "tags.mlflow.runName",
            "status",
            "start_time",
            "metrics.test_pr_auc",
            "metrics.test_roc_auc",
            "metrics.test_business_cost",
            "metrics.test_brier",
            "metrics.cv_pr_auc",
            "params.champion",
            "tags.data_source",
        ]
        if c in runs.columns
    ]
    from credit_default.pipeline.report import table as md_table

    frame = runs[cols].rename(columns=lambda c: c.split(".")[-1]).astype(object)
    rows = frame.where(frame.notna(), "").to_dict("records")
    lines = [
        "# MLflow run summary",
        "",
        f"Experiment `{cfg.registry.experiment}`, exported after the training run.",
        "",
        md_table(rows, [(c, c) for c in frame.columns]) if rows else "_no runs_",
        "",
    ]
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        name = cfg.registry.model_name
        versions = client.search_model_versions(f"name='{name}'")
        lines += ["## Registered versions", "", "| Version | Aliases | Run |", "|---|---|---|"]
        for v in sorted(versions, key=lambda v: int(v.version)):
            # search_model_versions leaves .aliases empty; get_model_version fills it.
            aliases = client.get_model_version(name, v.version).aliases
            lines.append(f"| {v.version} | {', '.join(aliases) or '-'} | `{v.run_id}` |")
    except Exception as exc:
        log.warning("registry summary skipped: %s", exc)
    out.write_text("\n".join(lines) + "\n")
    return out


RUN_FILES = (
    "metrics.json",
    "metrics.md",
    "gate.json",
    "gate.md",
    "decision.json",
    "decision.md",
    "fairness.json",
    "fairness.md",
    "evaluation.json",
    "predictions.json",
)


def _copy_run(run_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)
    if (dest / "figures").exists():
        shutil.rmtree(dest / "figures")
    shutil.copytree(run_dir / "figures", dest / "figures")


def publish(
    cfg,
    run_dir: Path | None = None,
    results_dir: Path | None = None,
    bundle_dir: Path | None = None,
    rejected: bool = False,
) -> None:
    run_dir = run_dir or REPO_ROOT / cfg.paths.artifacts_dir / "run"
    results_dir = results_dir or REPO_ROOT / cfg.paths.results_dir
    bundle_dir = bundle_dir or REPO_ROOT / cfg.paths.bundle_dir
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics["data"].get("synthetic") or metrics["data"].get("source") != "uci":
        raise SyntheticPublishError(
            "refusing to publish a run that was not trained on the real UCI data"
        )
    gate = run_dir / "gate.json"
    if not gate.exists():
        raise RuntimeError("refusing to publish: no gate.json in the run directory")
    passed = json.loads(gate.read_text())["passed"]
    if rejected:
        if passed:
            raise RuntimeError("--rejected given, but the gate passed")
        _copy_run(run_dir, results_dir / "challenger")
        log.info("published the rejected challenger's report to %s", results_dir / "challenger")
        return
    if not passed:
        raise RuntimeError("refusing to publish: the gate did not pass (use --rejected)")

    _copy_run(run_dir, results_dir)
    if (results_dir / "challenger").exists():  # the latest challenger is now production
        shutil.rmtree(results_dir / "challenger")

    bundle = ModelBundle.load(run_dir / "model.joblib")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "model.joblib", bundle_dir / "model.joblib")
    write_json({**bundle.metadata, "threshold": bundle.threshold}, bundle_dir / "metadata.json")
    mlflow_summary(cfg, results_dir / "mlflow_summary.md")
    log.info("published run to %s and %s", results_dir, bundle_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    publish(load_config(overrides), rejected="--rejected" in flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
