"""Metric suite: ranking (ROC-AUC, PR-AUC), calibration (Brier), decision (F1, cost)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_default.threshold.cost import business_cost


def evaluate_probs(y_true, proba, threshold: float, fn_cost: float, fp_cost: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "business_cost": business_cost(y_true, y_pred, fn_cost, fp_cost),
        "threshold": float(threshold),
        "approval_rate": float(1 - y_pred.mean()),
        "n": int(len(y_true)),
    }
