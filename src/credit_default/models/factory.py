"""Model pipelines: features -> preprocessing -> (optional SMOTE) -> estimator.

Imbalance strategies (compared in an ablation, chosen by config):
- ``class_weight``: ``class_weight="balanced"`` (LogReg, LightGBM) or ``scale_pos_weight`` (XGBoost)
- ``smote``: SMOTE inside the imblearn pipeline, so it only ever sees the training fold
- ``none``: no correction (isotonic calibration still fixes the probability scale)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_default.features.build import FeatureBuilder, build_preprocessor

FAMILIES = ("logreg", "lightgbm", "xgboost")
STRATEGIES = ("none", "class_weight", "smote")


def build_estimator(family: str, params: dict[str, Any], strategy: str, pos_rate: float, seed: int):
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
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if family == "xgboost":
        spw = (1 - pos_rate) / pos_rate if weighted else 1.0
        return XGBClassifier(
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
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
) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown imbalance strategy {strategy!r}; expected one of {STRATEGIES}")
    steps = [
        ("features", FeatureBuilder()),
        ("preprocess", build_preprocessor(family)),
    ]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", build_estimator(family, params, strategy, pos_rate, seed)))
    return Pipeline(steps)


def positive_rate(y) -> float:
    return float(np.mean(np.asarray(y)))
