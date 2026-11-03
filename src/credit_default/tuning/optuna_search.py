"""Optuna search (TPE, seeded) on stratified CV PR-AUC of the training split, plus the
imbalance-strategy ablation. All resampling happens inside the CV folds."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from credit_default.models.factory import build_pipeline, positive_rate

log = logging.getLogger(__name__)


def _suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    params = {}
    for name, spec in space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"unknown search-space type {spec['type']!r}")
    return params


def cv_pr_auc(
    family, params, strategy, X, y, folds: int, seed: int, n_jobs: int = 1
) -> tuple[float, float]:
    pipe = build_pipeline(family, params, strategy, positive_rate(y), seed, n_jobs)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores))


def tune(
    family: str,
    model_cfg,
    strategy: str,
    X,
    y,
    n_trials: int,
    folds: int,
    seed: int,
    timeout_s: float | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], float, optuna.Study | None]:
    """Return (best params, best CV PR-AUC, study). ``n_trials=0`` scores the config defaults."""
    defaults = dict(model_cfg.params)
    if n_trials <= 0:
        score, _ = cv_pr_auc(family, defaults, strategy, X, y, folds, seed, n_jobs)
        return defaults, score, None

    def objective(trial):
        params = {**defaults, **_suggest(trial, dict(model_cfg.search_space))}
        score, _ = cv_pr_auc(family, params, strategy, X, y, folds, seed, n_jobs)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{family}-{strategy}",
    )
    study.enqueue_trial({k: v for k, v in defaults.items() if k in model_cfg.search_space})
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    best = {**defaults, **study.best_params}
    log.info("%s: best CV PR-AUC %.4f after %d trials", family, study.best_value, len(study.trials))
    return best, float(study.best_value), study


def run_ablation(
    families, params_by_family, strategies, X, y, folds: int, seed: int, n_jobs: int = 1
) -> list[dict]:
    """CV PR-AUC for each (family, strategy) with the same hyper-parameters."""
    rows = []
    for family in families:
        for strategy in strategies:
            mean, std = cv_pr_auc(
                family, params_by_family[family], strategy, X, y, folds, seed, n_jobs
            )
            rows.append(
                {"family": family, "strategy": strategy, "cv_pr_auc": mean, "cv_pr_auc_std": std}
            )
            log.info("ablation %s/%s: PR-AUC %.4f +- %.4f", family, strategy, mean, std)
    return rows
