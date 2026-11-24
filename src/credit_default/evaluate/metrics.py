"""Metric suite: ranking (ROC-AUC, PR-AUC), calibration (Brier, log loss, ECE), and the
decision metrics at a threshold (confusion matrix, precision, recall, specificity, F1,
balanced accuracy, FPR, FNR, approval rate, expected cost)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from credit_default.threshold.cost import business_cost

EPS = 1e-15


def confusion(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def decision_metrics(y_true, y_pred, fn_cost: float, fp_cost: float) -> dict:
    """Threshold-dependent metrics. "Positive" = predicted default = decline."""
    c = confusion(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        **c,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "approval_rate": _ratio(tn + fn, n),
        "business_cost": business_cost(y_true, y_pred, fn_cost, fp_cost),
    }


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """ECE with equal-width bins: sum_b |mean(y) - mean(p)| * n_b / n."""
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += abs(y_true[mask].mean() - proba[mask].mean()) * mask.sum()
    return float(total / len(proba))


def reliability_curve(y_true, proba, n_bins: int = 10) -> list[dict]:
    """Equal-mass bins (deciles by default): mean predicted vs observed rate, with counts."""
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            rows.append(
                {
                    "mean_predicted": float(proba[chunk].mean()),
                    "observed_rate": float(y_true[chunk].mean()),
                    "n": int(len(chunk)),
                }
            )
    return rows


def probability_metrics(y_true, proba) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, EPS, 1 - EPS), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, proba),
    }


def evaluate_probs(y_true, proba, threshold: float, fn_cost: float, fp_cost: float) -> dict:
    """Every metric the reports use, for one set of probabilities and one threshold."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    return {
        **probability_metrics(y_true, proba),
        **decision_metrics(y_true, y_pred, fn_cost, fp_cost),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def curve_points(y_true, proba, max_points: int = 120) -> dict:
    """Down-sampled ROC and precision-recall curves for plotting on the site."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)

    def thin(xs, ys):
        if len(xs) <= max_points:
            idx = np.arange(len(xs))
        else:
            idx = np.unique(np.linspace(0, len(xs) - 1, max_points).round().astype(int))
        return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]

    return {"roc": thin(fpr, tpr), "pr": thin(recall[::-1], precision[::-1])}


def bootstrap_ci(
    y_true,
    proba,
    threshold: float,
    fn_cost: float,
    fp_cost: float,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CIs (stratified resampling) for ROC-AUC, PR-AUC, Brier and cost."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    keys = ("roc_auc", "pr_auc", "brier", "business_cost", "recall")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        y, p = y_true[idx], proba[idx]
        pred = (p >= threshold).astype(int)
        draws["roc_auc"].append(roc_auc_score(y, p))
        draws["pr_auc"].append(average_precision_score(y, p))
        draws["brier"].append(np.mean((p - y) ** 2))
        draws["business_cost"].append(business_cost(y, pred, fn_cost, fp_cost))
        draws["recall"].append(pred[y == 1].mean())
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}


def paired_bootstrap_diff(
    y_true,
    proba_a,
    proba_b,
    thr_a: float,
    thr_b: float,
    fn_cost: float,
    fp_cost: float,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Bootstrap the difference A - B on the same resampled rows (PR-AUC and expected cost)."""
    y_true = np.asarray(y_true).astype(int)
    a, b = np.asarray(proba_a, dtype=float), np.asarray(proba_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_pr, d_cost = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = y_true[idx]
        if y.min() == y.max():
            continue
        d_pr.append(average_precision_score(y, a[idx]) - average_precision_score(y, b[idx]))
        d_cost.append(
            business_cost(y, a[idx] >= thr_a, fn_cost, fp_cost)
            - business_cost(y, b[idx] >= thr_b, fn_cost, fp_cost)
        )

    def summary(v):
        v = np.asarray(v)
        return {
            "mean": float(v.mean()),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_a_better": None,
        }

    pr, cost = summary(d_pr), summary(d_cost)
    pr["p_a_better"] = float(np.mean(np.asarray(d_pr) > 0))
    cost["p_a_better"] = float(np.mean(np.asarray(d_cost) < 0))
    return {"pr_auc": pr, "business_cost": cost}
