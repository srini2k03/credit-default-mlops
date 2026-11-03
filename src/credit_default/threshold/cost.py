"""Cost-based decision threshold.

Expected cost per applicant = (FN * fn_cost + FP * fp_cost) / n, in units of one lost-margin FP.
With calibrated probabilities the Bayes-optimal threshold is fp_cost / (fp_cost + fn_cost)
(1/6 for 5:1); we still pick it empirically on validation data, which is robust to residual
miscalibration, and report the theoretical value next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


def business_cost(y_true, y_pred, fn_cost: float, fp_cost: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float((fn * fn_cost + fp * fp_cost) / len(y_true))


@dataclass
class CostCurve:
    thresholds: np.ndarray
    costs: np.ndarray

    @property
    def best_threshold(self) -> float:
        return float(self.thresholds[int(np.argmin(self.costs))])

    @property
    def best_cost(self) -> float:
        return float(np.min(self.costs))


def cost_curve(y_true, proba, fn_cost: float, fp_cost: float, grid: int = 199) -> CostCurve:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    thresholds = np.linspace(0, 1, grid + 2)[1:-1]
    preds = proba[None, :] >= thresholds[:, None]
    fn = ((y_true[None, :] == 1) & ~preds).sum(axis=1)
    fp = ((y_true[None, :] == 0) & preds).sum(axis=1)
    costs = (fn * fn_cost + fp * fp_cost) / len(y_true)
    return CostCurve(thresholds=thresholds, costs=costs)


def optimal_threshold(y_true, proba, fn_cost: float, fp_cost: float, grid: int = 199) -> float:
    return cost_curve(y_true, proba, fn_cost, fp_cost, grid).best_threshold


def bayes_threshold(fn_cost: float, fp_cost: float) -> float:
    return fp_cost / (fp_cost + fn_cost)


def max_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])
