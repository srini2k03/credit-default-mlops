import numpy as np
import pytest

from credit_default.evaluate.metrics import evaluate_probs
from credit_default.features import ENGINEERED_COLUMNS, engineer
from credit_default.threshold.cost import (
    bayes_threshold,
    business_cost,
    cost_curve,
    max_f1_threshold,
    optimal_threshold,
)


def test_engineer_is_finite_and_folds_unknown_codes(small_df):
    df = small_df.copy()
    df.loc[:5, "EDUCATION"] = [0, 5, 6, 1, 2, 3]
    df.loc[:1, "MARRIAGE"] = [0, 2]
    out = engineer(df)
    assert set(ENGINEERED_COLUMNS) <= set(out.columns)
    assert np.isfinite(out.to_numpy()).all()
    assert set(out["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"].unique()) <= {1, 2, 3}
    assert out.loc[0, "EDUCATION"] == 4 and out.loc[0, "MARRIAGE"] == 3


def test_business_cost_counts_fn_and_fp():
    y = np.array([1, 1, 0, 0])
    pred = np.array([0, 1, 1, 0])  # one FN, one FP
    assert business_cost(y, pred, fn_cost=5, fp_cost=1) == pytest.approx(6 / 4)


def test_optimal_threshold_minimises_cost():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)  # perfectly calibrated scores
    curve = cost_curve(y, p, 5, 1, grid=199)
    assert curve.best_cost == pytest.approx(curve.costs.min())
    # With calibrated scores the empirical optimum sits near the Bayes threshold 1/6.
    assert abs(optimal_threshold(y, p, 5, 1) - bayes_threshold(5, 1)) < 0.05
    assert 0 < max_f1_threshold(y, p) < 1


def test_evaluate_probs_keys():
    m = evaluate_probs([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], 0.5, 5, 1)
    assert m["roc_auc"] == 1.0 and m["business_cost"] == 0.0
    assert {"roc_auc", "pr_auc", "f1", "brier", "business_cost"} <= set(m)
