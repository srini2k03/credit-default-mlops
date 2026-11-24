"""Train, tune, calibrate, choose a threshold, evaluate, explain, log to MLflow, register as staging.

One protocol for every model family (configs/train/default.yaml):
  Optuna on stratified CV PR-AUC (train split) -> fit on train -> calibrate on the calibration
  split (none / Platt / isotonic, kept only if it lowers validation Brier) -> cost-optimal
  cutoff on validation -> every metric on the untouched test split, with bootstrap CIs.
Champion = lowest validation expected cost among deployable families (PR-AUC breaks ties).

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
from credit_default.decision import engine
from credit_default.evaluate.metrics import (
    bootstrap_ci,
    curve_points,
    evaluate_probs,
    paired_bootstrap_diff,
    probability_metrics,
    reliability_curve,
)
from credit_default.evaluate.plots import plot_calibration, plot_cost_curve
from credit_default.models.bundle import ModelBundle
from credit_default.models.factory import build_pipeline, family_available, positive_rate
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
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")


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
    cv_pr_auc_std: float | None = None
    calibration: list[dict] = field(default_factory=list)
    calibration_test_proba: dict = field(default_factory=dict)
    test_proba: np.ndarray | None = None
    ci95: dict = field(default_factory=dict)
    deployable: bool = True


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


def calibrate(pipe, method: str, X, y):
    """Fit a calibrator on (X, y) on top of the frozen, already-fitted pipeline."""
    if method == "none":
        return pipe
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"unknown calibration {method!r}; expected one of {CALIBRATION_METHODS}")
    return CalibratedClassifierCV(FrozenEstimator(pipe), method=method).fit(X, y)


def _methods(cfg) -> list[str]:
    requested = cfg.train.get("calibration", "isotonic")
    if requested == "auto":
        return list(cfg.train.get("calibration_methods", CALIBRATION_METHODS))
    return [requested]


def fit_family(
    cfg, family: str, splits: Splits, params: dict, strategy: str, exclude=None
) -> tuple[ModelBundle, dict]:
    """Fit on train, calibrate on the calibration split, choose the cutoff on validation."""
    Xtr, ytr = splits.xy(splits.train)
    Xcal, ycal = splits.xy(splits.calibration)
    Xval, yval = splits.xy(splits.validation)
    exclude = tuple(cfg.features.exclude) if exclude is None else tuple(exclude)
    pipe = build_pipeline(
        family, params, strategy, positive_rate(ytr), cfg.seed, cfg.train.n_jobs, exclude
    )
    pipe.fit(Xtr, ytr)
    candidates = {}
    for method in _methods(cfg):
        cal = calibrate(pipe, method, Xcal, ycal)
        vp = cal.predict_proba(Xval)[:, 1]
        candidates[method] = {
            "calibrator": cal,
            "val_proba": vp,
            "val": probability_metrics(yval, vp),
        }
    chosen = min(
        candidates,
        key=lambda m: (round(candidates[m]["val"]["brier"], 6), candidates[m]["val"]["log_loss"]),
    )
    val_proba = candidates[chosen]["val_proba"]
    policy = engine.CostPolicy.from_config(cfg.cost)
    threshold, _ = engine.choose_threshold(
        yval, val_proba, policy, engine.threshold_grid(cfg.cost.threshold_grid)
    )
    bundle = ModelBundle(
        family=family,
        pipeline=pipe,
        calibrator=candidates[chosen]["calibrator"],
        threshold=threshold,
        background=Xtr.sample(min(cfg.train.background_rows, len(Xtr)), random_state=cfg.seed),
        calibration=chosen,
    )
    curve = cost_curve(yval, val_proba, cfg.cost.fn_cost, cfg.cost.fp_cost, cfg.cost.threshold_grid)
    return bundle, {"curve": curve, "val_proba": val_proba, "candidates": candidates}


def _calibration_rows(cfg, candidates: dict, chosen: str, splits: Splits) -> tuple[list, dict]:
    """Every calibration method on test, each at its own validation-optimal cutoff."""
    _, yval = splits.xy(splits.validation)
    Xte, yte = splits.xy(splits.test)
    policy = engine.CostPolicy(cfg.cost.fn_cost, cfg.cost.fp_cost)
    rows, probs = [], {}
    for method, c in candidates.items():
        thr, _ = engine.choose_threshold(yval, c["val_proba"], policy)
        tp = c["calibrator"].predict_proba(Xte)[:, 1]
        probs[method] = tp
        rows.append(
            {
                "method": method,
                "chosen": method == chosen,
                "validation": c["val"],
                "test": evaluate_probs(yte, tp, thr, cfg.cost.fn_cost, cfg.cost.fp_cost),
                "test_reliability": reliability_curve(yte, tp),
            }
        )
    return rows, probs


def train(cfg, df: pd.DataFrame, data_info: dict, register: bool = True) -> TrainResult:
    try:
        return _train(cfg, df, data_info, register)
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise


def _fit_and_score(cfg, family, splits, strategy, exclude, deployable) -> FamilyResult:
    Xtr, ytr = splits.xy(splits.train)
    Xval, yval = splits.xy(splits.validation)
    Xte, yte = splits.xy(splits.test)
    fn_cost, fp_cost = cfg.cost.fn_cost, cfg.cost.fp_cost
    ts = time.time()
    params, cv_score, study = tune(
        family,
        cfg.models[family],
        strategy,
        Xtr,
        ytr,
        cfg.train.n_trials,
        cfg.train.cv_folds,
        cfg.seed,
        cfg.train.timeout_s,
        cfg.train.n_jobs,
        exclude,
    )
    tuning_s = time.time() - ts
    bundle, extra = fit_family(cfg, family, splits, params, strategy, exclude)
    val_m = evaluate_probs(yval, extra["val_proba"], bundle.threshold, fn_cost, fp_cost)
    test_proba = bundle.predict_proba(Xte)
    test_m = evaluate_probs(yte, test_proba, bundle.threshold, fn_cost, fp_cost)
    uncal = bundle.pipeline.predict_proba(Xte)[:, 1]
    ci = bootstrap_ci(
        yte, test_proba, bundle.threshold, fn_cost, fp_cost, cfg.train.n_bootstrap, cfg.seed
    )
    cv_std = study.best_trial.user_attrs.get("cv_std") if study is not None else None
    cal_rows, cal_probs = _calibration_rows(cfg, extra["candidates"], bundle.calibration, splits)
    return FamilyResult(
        family=family,
        params=params,
        cv_pr_auc=cv_score,
        bundle=bundle,
        val_metrics=val_m,
        test_metrics=test_m,
        uncalibrated_test_brier=float(np.mean((uncal - yte.to_numpy()) ** 2)),
        tuning_seconds=tuning_s,
        n_trials=len(study.trials) if study else 0,
        cv_pr_auc_std=cv_std,
        calibration=cal_rows,
        calibration_test_proba=cal_probs,
        test_proba=test_proba,
        ci95=ci,
        deployable=deployable,
    )


def _select(cfg, families: list[FamilyResult]) -> FamilyResult:
    candidates = [r for r in families if r.deployable]
    if not candidates:
        raise RuntimeError("no deployable model family was trained")
    if cfg.train.selection_metric == "business_cost":
        return min(
            candidates, key=lambda r: (r.val_metrics["business_cost"], -r.val_metrics["pr_auc"])
        )
    return max(candidates, key=lambda r: r.val_metrics["pr_auc"])


def _family_summary(r: FamilyResult) -> dict:
    return {
        "family": r.family,
        "deployable": r.deployable,
        "cv_pr_auc": r.cv_pr_auc,
        "cv_pr_auc_std": r.cv_pr_auc_std,
        "params": r.params,
        "n_trials": r.n_trials,
        "tuning_seconds": round(r.tuning_seconds, 1),
        "calibration": r.bundle.calibration,
        "validation": r.val_metrics,
        "test": r.test_metrics,
        "test_ci95": r.ci95,
        "test_brier_uncalibrated": r.uncalibrated_test_brier,
    }


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
    exclude = tuple(cfg.features.exclude)
    deployable = set(cfg.train.get("deployable", cfg.train.models))

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
            "excluded_features": ",".join(exclude) or "none",
            "calibration": cfg.train.get("calibration", "isotonic"),
        }
    )

    families: list[FamilyResult] = []
    skipped: list[dict] = []
    for family in cfg.train.models:
        if not family_available(family):
            log.warning("%s skipped: its library is not installed", family)
            skipped.append({"family": family, "reason": "library not installed"})
            continue
        with mlflow.start_run(run_name=family, nested=True):
            r = _fit_and_score(cfg, family, splits, strategy, exclude, family in deployable)
            mlflow.set_tag("deployable", str(r.deployable).lower())
            mlflow.log_params({f"p_{k}": v for k, v in r.params.items()})
            mlflow.log_param("calibration_chosen", r.bundle.calibration)
            mlflow.log_metrics(
                {
                    "cv_pr_auc": r.cv_pr_auc,
                    "tuning_seconds": r.tuning_seconds,
                    **{f"val_{k}": v for k, v in r.val_metrics.items()},
                    **{f"test_{k}": v for k, v in r.test_metrics.items()},
                    "test_brier_uncalibrated": r.uncalibrated_test_brier,
                }
            )
            families.append(r)
            log.info(
                "%s: val cost %.4f, test PR-AUC %.4f, calibration %s",
                family,
                r.val_metrics["business_cost"],
                r.test_metrics["pr_auc"],
                r.bundle.calibration,
            )

    champion = _select(cfg, families)
    bundle = champion.bundle

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
            cfg.train.n_jobs,
            exclude,
        )

    # Threshold comparison for the champion: cost-optimal vs 0.5 vs max-F1 (chosen on validation).
    val_proba = bundle.predict_proba(Xval)
    test_proba = champion.test_proba
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
    }
    figures["calibration"] = plot_calibration(
        yte.to_numpy(), champion.calibration_test_proba, fig_dir / "calibration.png"
    )
    analysis = _analyse(cfg, champion, families, splits, out_dir, fig_dir, figures)

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
        "excluded_features": list(exclude),
        "selection_rule": (
            "lowest expected cost per applicant on validation among deployable families; "
            "validation PR-AUC breaks ties"
        ),
        "champion": champion.family,
        "calibration": bundle.calibration,
        "test": champion.test_metrics,
        "test_ci95": champion.ci95,
        "validation": champion.val_metrics,
        "families": [_family_summary(r) for r in families],
        "skipped_families": skipped,
        "calibration_comparison": [
            {k: v for k, v in c.items() if k != "test_reliability"} for c in champion.calibration
        ],
        "thresholds": threshold_rows,
        "ablation": ablation,
        "train_seconds": round(time.time() - t0, 1),
    }
    bundle.metadata = {
        k: metrics[k]
        for k in (
            "generated_at",
            "git_commit",
            "data",
            "cost",
            "imbalance",
            "test",
            "excluded_features",
        )
    }
    bundle.metadata.update(
        {"family": champion.family, "params": champion.params, "calibration": bundle.calibration}
    )

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
    registered = (
        f"registered v{version} as @{cfg.registry.staging_alias}" if version else "not registered"
    )
    log.info(
        "champion %s | test PR-AUC %.4f | cost %.4f | %s | analysis: %s",
        champion.family,
        champion.test_metrics["pr_auc"],
        champion.test_metrics["business_cost"],
        registered,
        ", ".join(sorted(analysis)),
    )
    return result


def _analyse(cfg, champion, families, splits, out_dir: Path, fig_dir: Path, figures: dict) -> dict:
    """Decision analysis, fairness, importance, error analysis, curves and exported predictions."""
    from credit_default.evaluate import analysis
    from credit_default.fairness.report import fairness_report
    from credit_default.fairness.report import to_markdown as fairness_md

    bundle = champion.bundle
    Xval, yval = splits.xy(splits.validation)
    Xte, yte = splits.xy(splits.test)
    p_val, p_te = bundle.predict_proba(Xval), champion.test_proba
    fn_cost, fp_cost = cfg.cost.fn_cost, cfg.cost.fp_cost
    written = {}

    decision = engine.analyse(
        yval,
        p_val,
        yte,
        p_te,
        engine.CostPolicy.from_config(cfg.cost),
        bundle.threshold,
        list(cfg.cost.sensitivity_ratios),
    )
    write_json(decision, out_dir / "decision.json")
    (out_dir / "decision.md").write_text(engine.to_markdown(decision))
    written["decision"] = out_dir / "decision.json"

    fair = fairness_report(
        Xte, yte, p_te, bundle.threshold, min(300, cfg.train.n_bootstrap), cfg.seed
    )
    fair["model"] = {"family": champion.family, "excluded_features": list(cfg.features.exclude)}
    # Same family, same hyper-parameters, with the excluded attributes as inputs: what does
    # excluding them change in accuracy and in group outcomes?
    if cfg.features.exclude:
        aware, _ = fit_family(
            cfg, champion.family, splits, champion.params, cfg.train.imbalance, ()
        )
        p_aware = aware.predict_proba(Xte)
        fair["with_excluded_features"] = {
            "test": evaluate_probs(yte, p_aware, aware.threshold, fn_cost, fp_cost),
            "report": fairness_report(Xte, yte, p_aware, aware.threshold, 100, cfg.seed),
        }
    write_json(fair, out_dir / "fairness.json")
    (out_dir / "fairness.md").write_text(fairness_md(fair))
    written["fairness"] = out_dir / "fairness.json"

    evaluation = {
        "champion": champion.family,
        "curves": {r.family: curve_points(yte, r.test_proba) for r in families},
        "reliability": {c["method"]: c["test_reliability"] for c in champion.calibration},
        "paired_vs_champion": {
            r.family: paired_bootstrap_diff(
                yte,
                p_te,
                r.test_proba,
                bundle.threshold,
                r.bundle.threshold,
                fn_cost,
                fp_cost,
                cfg.train.n_bootstrap,
                cfg.seed,
            )
            for r in families
            if r is not champion
        },
        "error_analysis": analysis.error_analysis(
            Xte, yte, p_te, bundle.threshold, fn_cost, fp_cost
        ),
    }
    try:
        from credit_default.explain.shap_explain import global_shap_plots

        figures.update(global_shap_plots(bundle, Xte, fig_dir, cfg.train.shap_sample, cfg.seed))
        imp = analysis.importance_comparison(
            bundle, Xte, yte, cfg.train.permutation_repeats, cfg.seed, cfg.train.shap_sample
        )
        evaluation["importance"] = imp
        top = [r["feature"] for r in imp["model_inputs"][:3]]
        figures.update(analysis.dependence_plots(bundle, Xte, fig_dir, top, cfg.seed))
        # One declined and one approved test applicant, explained.
        for label, mask in (
            ("declined", p_te >= bundle.threshold),
            ("approved", p_te < bundle.threshold),
        ):
            if mask.any():
                i = int(np.flatnonzero(mask)[0])
                figures[f"waterfall_{label}"] = analysis.waterfall_plot(
                    bundle,
                    Xte.iloc[[i]],
                    fig_dir / f"shap_waterfall_{label}.png",
                    f"Test applicant {label}: P(default) {p_te[i]:.3f}, cutoff {bundle.threshold:.3f}",
                )
    except Exception as exc:  # explanations must never block a training run
        log.warning("SHAP analysis skipped: %s", exc)
    write_json(evaluation, out_dir / "evaluation.json")
    written["evaluation"] = out_dir / "evaluation.json"
    write_json(
        {
            "champion": champion.family,
            "threshold": bundle.threshold,
            **analysis.export_predictions(yval, p_val, yte, p_te),
        },
        out_dir / "predictions.json",
    )
    written["predictions"] = out_dir / "predictions.json"
    return written


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
