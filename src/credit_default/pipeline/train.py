"""Train, tune, calibrate, choose a threshold, evaluate, explain, log to MLflow, register as staging.

Usage:
  python -m credit_default.pipeline.train                     # real data, full search
  python -m credit_default.pipeline.train data=synthetic train=fast
  python -m credit_default.pipeline.train train.n_trials=10 --no-register
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import TARGET
from credit_default.data.split import Splits, make_splits
from credit_default.evaluate.metrics import evaluate_probs
from credit_default.evaluate.plots import plot_calibration, plot_cost_curve
from credit_default.models.bundle import ModelBundle
from credit_default.models.factory import build_pipeline, positive_rate
from credit_default.pipeline import report
from credit_default.pipeline.steps import (
    data_fingerprint,
    git_commit,
    ingest,
    now_iso,
    validate,
    write_json,
)
from credit_default.registry import mlflow_registry as reg
from credit_default.threshold.cost import bayes_threshold, cost_curve, max_f1_threshold
from credit_default.tuning.optuna_search import run_ablation, tune

log = logging.getLogger(__name__)


@dataclass
class FamilyResult:
    family: str
    params: dict
    cv_pr_auc: float
    bundle: ModelBundle
    val_metrics: dict
    test_metrics: dict
    uncalibrated_test_brier: float
    tuning_seconds: float
    n_trials: int


@dataclass
class TrainResult:
    champion: FamilyResult
    families: list[FamilyResult]
    ablation: list[dict]
    out_dir: Path
    metrics: dict
    run_id: str | None = None
    model_version: str | None = None
    figures: dict = field(default_factory=dict)


def fit_family(
    cfg, family: str, splits: Splits, params: dict, strategy: str
) -> tuple[ModelBundle, dict]:
    Xtr, ytr = splits.xy(splits.train)
    Xcal, ycal = splits.xy(splits.calibration)
    Xval, yval = splits.xy(splits.validation)
    pipe = build_pipeline(family, params, strategy, positive_rate(ytr), cfg.seed)
    pipe.fit(Xtr, ytr)
    calibrator = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    calibrator.fit(Xcal, ycal)
    val_proba = calibrator.predict_proba(Xval)[:, 1]
    curve = cost_curve(yval, val_proba, cfg.cost.fn_cost, cfg.cost.fp_cost, cfg.cost.threshold_grid)
    bundle = ModelBundle(
        family=family,
        pipeline=pipe,
        calibrator=calibrator,
        threshold=curve.best_threshold,
        background=Xtr.sample(min(cfg.train.background_rows, len(Xtr)), random_state=cfg.seed),
    )
    return bundle, {"curve": curve, "val_proba": val_proba}


def train(cfg, df: pd.DataFrame, data_info: dict, register: bool = True) -> TrainResult:
    try:
        return _train(cfg, df, data_info, register)
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise


def _train(cfg, df: pd.DataFrame, data_info: dict, register: bool) -> TrainResult:
    t0 = time.time()
    out_dir = REPO_ROOT / cfg.paths.artifacts_dir / "run"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(df, cfg.train.splits, cfg.seed)
    Xtr, ytr = splits.xy(splits.train)
    Xval, yval = splits.xy(splits.validation)
    Xte, yte = splits.xy(splits.test)
    fn_cost, fp_cost = cfg.cost.fn_cost, cfg.cost.fp_cost
    strategy = cfg.train.imbalance

    reg.setup_tracking(cfg)
    parent = mlflow.start_run(run_name=f"train-{data_info['source']}-{now_iso()}")
    mlflow.set_tags(
        {
            "data_source": data_info["source"],
            "synthetic": str(data_info["synthetic"]).lower(),
            "data_sha256": data_info["sha256"],
            "git_commit": git_commit(),
        }
    )
    mlflow.log_params(
        {
            "imbalance": strategy,
            "n_trials": cfg.train.n_trials,
            "cv_folds": cfg.train.cv_folds,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "seed": cfg.seed,
            "n_train": len(Xtr),
            "n_test": len(Xte),
        }
    )

    families: list[FamilyResult] = []
    for family in cfg.train.models:
        model_cfg = cfg.models[family]
        with mlflow.start_run(run_name=family, nested=True):
            ts = time.time()
            params, cv_score, study = tune(
                family,
                model_cfg,
                strategy,
                Xtr,
                ytr,
                cfg.train.n_trials,
                cfg.train.cv_folds,
                cfg.seed,
                cfg.train.timeout_s,
            )
            tuning_s = time.time() - ts
            bundle, extra = fit_family(cfg, family, splits, params, strategy)
            val_m = evaluate_probs(yval, extra["val_proba"], bundle.threshold, fn_cost, fp_cost)
            test_m = evaluate_probs(
                yte, bundle.predict_proba(Xte), bundle.threshold, fn_cost, fp_cost
            )
            uncal = bundle.pipeline.predict_proba(Xte)[:, 1]
            uncal_brier = float(np.mean((uncal - yte.to_numpy()) ** 2))
            mlflow.log_params({f"p_{k}": v for k, v in params.items()})
            mlflow.log_metrics(
                {
                    "cv_pr_auc": cv_score,
                    "tuning_seconds": tuning_s,
                    **{f"val_{k}": v for k, v in val_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                    "test_brier_uncalibrated": uncal_brier,
                }
            )
            families.append(
                FamilyResult(
                    family,
                    params,
                    cv_score,
                    bundle,
                    val_m,
                    test_m,
                    uncal_brier,
                    tuning_s,
                    len(study.trials) if study else 0,
                )
            )
            log.info(
                "%s: val cost %.4f, test PR-AUC %.4f",
                family,
                val_m["business_cost"],
                test_m["pr_auc"],
            )

    key = (
        (lambda r: (r.val_metrics["business_cost"], -r.val_metrics["pr_auc"]))
        if cfg.train.selection_metric == "business_cost"
        else (lambda r: -r.val_metrics["pr_auc"])
    )
    champion = sorted(families, key=key)[0]
    bundle = champion.bundle

    # Imbalance ablation with each family's tuned parameters.
    ablation = []
    if cfg.train.ablation.enabled:
        ablation = run_ablation(
            [r.family for r in families],
            {r.family: r.params for r in families},
            list(cfg.train.ablation.strategies),
            Xtr,
            ytr,
            cfg.train.ablation.cv_folds,
            cfg.seed,
        )

    # Threshold comparison for the champion: cost-optimal vs 0.5 vs max-F1 (chosen on validation).
    val_proba = bundle.predict_proba(Xval)
    test_proba = bundle.predict_proba(Xte)
    curve = cost_curve(yval, val_proba, fn_cost, fp_cost, cfg.cost.threshold_grid)
    thresholds = {
        "cost-optimal": bundle.threshold,
        "0.5": 0.5,
        "max-F1": max_f1_threshold(yval, val_proba),
    }
    threshold_rows = [
        {"rule": name, **evaluate_probs(yte, test_proba, thr, fn_cost, fp_cost)}
        for name, thr in thresholds.items()
    ]
    figures = {
        "cost_curve": plot_cost_curve(
            curve,
            thresholds,
            f"Cost vs threshold ({champion.family}, validation)",
            fig_dir / "cost_curve.png",
        ),
        "calibration": plot_calibration(
            yte.to_numpy(),
            {"uncalibrated": bundle.pipeline.predict_proba(Xte)[:, 1], "isotonic": test_proba},
            fig_dir / "calibration.png",
        ),
    }
    try:
        from credit_default.explain.shap_explain import global_shap_plots

        figures.update(global_shap_plots(bundle, Xte, fig_dir, cfg.train.shap_sample, cfg.seed))
    except Exception as exc:  # explanations must never block a training run
        log.warning("SHAP plots skipped: %s", exc)

    metrics = {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "data": {
            **data_info,
            "n_rows": int(len(df)),
            "default_rate": float(df[TARGET].mean()),
            "n_train": len(Xtr),
            "n_calibration": len(splits.calibration),
            "n_validation": len(Xval),
            "n_test": len(Xte),
        },
        "cost": {
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "bayes_threshold": bayes_threshold(fn_cost, fp_cost),
        },
        "imbalance": strategy,
        "champion": champion.family,
        "test": champion.test_metrics,
        "validation": champion.val_metrics,
        "families": [
            {
                "family": r.family,
                "cv_pr_auc": r.cv_pr_auc,
                "params": r.params,
                "n_trials": r.n_trials,
                "tuning_seconds": round(r.tuning_seconds, 1),
                "validation": r.val_metrics,
                "test": r.test_metrics,
                "test_brier_uncalibrated": r.uncalibrated_test_brier,
            }
            for r in families
        ],
        "thresholds": threshold_rows,
        "ablation": ablation,
        "train_seconds": round(time.time() - t0, 1),
    }
    bundle.metadata = {
        k: metrics[k] for k in ("generated_at", "git_commit", "data", "cost", "imbalance", "test")
    }
    bundle.metadata["family"] = champion.family
    bundle.metadata["params"] = champion.params

    bundle.save(out_dir / "model.joblib")
    write_json(metrics, out_dir / "metrics.json")
    report.write_markdown(metrics, out_dir)

    mlflow.log_metrics({f"test_{k}": v for k, v in champion.test_metrics.items()})
    mlflow.log_param("champion", champion.family)
    mlflow.log_artifacts(str(out_dir), artifact_path="report")
    mlflow.log_dict(OmegaConf.to_container(cfg), "config.json")
    info = reg.log_bundle(bundle, cfg.registry.model_name if register else None)
    version = (
        str(info.registered_model_version) if register and info.registered_model_version else None
    )
    if version:
        reg.set_alias(cfg, version, cfg.registry.staging_alias)
    run_id = parent.info.run_id
    mlflow.end_run()

    result = TrainResult(champion, families, ablation, out_dir, metrics, run_id, version, figures)
    write_json(
        {"run_id": run_id, "model_version": version, "champion": champion.family},
        out_dir / "run_info.json",
    )
    log.info(
        "champion %s | test PR-AUC %.4f | cost %.4f | registered v%s (alias %s)",
        champion.family,
        champion.test_metrics["pr_auc"],
        champion.test_metrics["business_cost"],
        version,
        cfg.registry.staging_alias,
    )
    return result


def run(cfg, register: bool = True) -> TrainResult:
    path = ingest(cfg)
    df = validate(cfg, path)
    return train(cfg, df, data_fingerprint(cfg, path), register=register)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    run(cfg, register="--no-register" not in flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
