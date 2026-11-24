"""Model pipelines: features -> preprocessing -> (optional SMOTE) -> estimator.

Families: logistic regression (baseline), random forest, LightGBM, XGBoost, and CatBoost.
CatBoost is an optional dependency (``pip install -e ".[research]"``): it takes part in the
model comparison but is not a deployable family, which keeps the serving image small.

Imbalance strategies (compared in an ablation, chosen by config):
- ``class_weight``: ``class_weight="balanced"`` (LogReg, LightGBM), ``"balanced_subsample"``
  (random forest), ``scale_pos_weight`` (XGBoost) or ``auto_class_weights="Balanced"`` (CatBoost)
- ``smote``: SMOTE inside the imblearn pipeline, so it only ever sees the training fold
- ``none``: no correction (isotonic calibration still fixes the probability scale)

Thread count is explicit (``train.n_jobs``): LightGBM and XGBoost ship different OpenMP runtimes,
and "use every core" (-1) oversubscribes badly when both are loaded or the host is busy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_default.features.build import FeatureBuilder, build_preprocessor

FAMILIES = ("logreg", "random_forest", "lightgbm", "xgboost", "catboost")
TREE_FAMILIES = {"random_forest", "lightgbm", "xgboost", "catboost"}
OPTIONAL_FAMILIES = {"catboost": "catboost"}  # family -> import name


def family_available(family: str) -> bool:
    """True if the family's library can be imported (CatBoost is optional)."""
    module = OPTIONAL_FAMILIES.get(family)
    if module is None:
        return True
    import importlib.util

    return importlib.util.find_spec(module) is not None


STRATEGIES = ("none", "class_weight", "smote")


def build_estimator(
    family: str, params: dict[str, Any], strategy: str, pos_rate: float, seed: int, n_jobs: int = 1
):
    params = dict(params)
    weighted = strategy == "class_weight"
    if family == "logreg":
        return LogisticRegression(
            class_weight="balanced" if weighted else None, random_state=seed, **params
        )
    if family == "lightgbm":
        return LGBMClassifier(
            class_weight="balanced" if weighted else None,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            **params,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            auto_class_weights="Balanced" if weighted else None,
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
            **params,
        )
    if family == "xgboost":
        spw = (1 - pos_rate) / pos_rate if weighted else 1.0
        return XGBClassifier(
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="logloss",
            **params,
        )
    raise ValueError(f"unknown model family {family!r}; expected one of {FAMILIES}")


def build_pipeline(
    family: str,
    params: dict[str, Any],
    strategy: str = "class_weight",
    pos_rate: float = 0.22,
    seed: int = 42,
    n_jobs: int = 1,
    exclude: tuple[str, ...] | list[str] = (),
) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown imbalance strategy {strategy!r}; expected one of {STRATEGIES}")
    exclude = tuple(exclude)
    steps = [
        ("features", FeatureBuilder(drop=exclude)),
        ("preprocess", build_preprocessor(family, exclude)),
    ]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", build_estimator(family, params, strategy, pos_rate, seed, n_jobs)))
    return Pipeline(steps)


def positive_rate(y) -> float:
    return float(np.mean(np.asarray(y)))
