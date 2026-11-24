"""Decision engine, metric suite, calibration selection and fairness report."""

import numpy as np
import pandas as pd
import pytest

from credit_default.data.schema import FEATURE_COLUMNS, TARGET
from credit_default.decision import engine
from credit_default.evaluate import metrics
from credit_default.fairness.report import fairness_report, group_columns


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.22
    p = np.clip(0.22 + 0.35 * (y - 0.22) + rng.normal(0, 0.15, len(y)), 0.001, 0.999)
    return y.astype(int), p


def test_decision_metrics_match_confusion_matrix(scored):
    y, p = scored
    m = metrics.evaluate_probs(y, p, 0.3, 5, 1)
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    assert tp + fp + tn + fn == len(y)
    assert m["business_cost"] == pytest.approx((5 * fn + fp) / len(y))
    assert m["fpr"] == pytest.approx(fp / (fp + tn)) and m["fnr"] == pytest.approx(1 - m["recall"])
    assert m["specificity"] == pytest.approx(1 - m["fpr"])
    assert m["balanced_accuracy"] == pytest.approx((m["recall"] + m["specificity"]) / 2)
    assert m["approval_rate"] == pytest.approx((tn + fn) / len(y))


def test_ece_is_zero_for_perfect_calibration_and_large_when_shifted():
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert metrics.expected_calibration_error(y, p) < 0.01
    assert metrics.expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_cost_optimal_threshold_beats_alternatives(scored):
    y, p = scored
    policy = engine.CostPolicy(fn_cost=5, fp_cost=1)
    thr, _ = engine.choose_threshold(y, p, policy)
    best = engine.evaluate_threshold(y, p, thr, policy)["business_cost"]
    for other in (0.1, 0.3, 0.5):
        assert best <= engine.evaluate_threshold(y, p, other, policy)["business_cost"] + 1e-12


def test_threshold_falls_as_miss_cost_rises(scored):
    y, p = scored
    rows = engine.sensitivity(y, p, y, p, [1, 2, 5, 10, 20], deployed=0.2)
    thresholds = [r["optimal_threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True)
    assert all(r["regret_deployed"] >= -1e-12 for r in rows)  # in-sample optimum is optimal
    assert rows[0]["bayes_threshold"] == pytest.approx(0.5)


def test_policy_constraint_and_fixed_threshold(scored):
    y, p = scored
    thr, reason = engine.choose_threshold(y, p, engine.CostPolicy(5, 1, min_approval_rate=0.8))
    assert engine.evaluate_threshold(y, p, thr, engine.CostPolicy())["approval_rate"] >= 0.8
    assert "80%" in reason
    assert engine.choose_threshold(y, p, engine.CostPolicy(threshold=0.42))[0] == 0.42
    with pytest.raises(ValueError):
        engine.CostPolicy(fn_cost=0)


def test_analyse_bundle_of_outputs(scored):
    y, p = scored
    a = engine.analyse(y, p, y, p, engine.CostPolicy(), 0.2, [1, 5, 10])
    assert {r["rule"].split(" ")[0] for r in a["rules"]} >= {"deployed", "0.5", "max", "Bayes"}
    assert len(a["sensitivity"]) == 3 and "not an approved" in a["disclaimer"]
    assert "Sensitivity" in engine.to_markdown(a)


def test_bootstrap_intervals_contain_point_estimate(scored):
    y, p = scored
    ci = metrics.bootstrap_ci(y, p, 0.2, 5, 1, n_boot=100, seed=0)
    point = metrics.evaluate_probs(y, p, 0.2, 5, 1)
    for k in ("roc_auc", "pr_auc", "business_cost"):
        assert ci[k][0] <= point[k] <= ci[k][1]


def test_fairness_report_groups_and_flags(fixture_df):
    X = fixture_df[FEATURE_COLUMNS].reset_index(drop=True)
    y = fixture_df[TARGET].to_numpy()
    rng = np.random.default_rng(0)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.1, len(y)), 0, 1)
    r = fairness_report(X, y, p, 0.3, n_boot=30)
    assert set(r["attributes"]) == {"SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"}
    sex = r["attributes"]["SEX"]
    assert sum(g["n"] for g in sex["groups"]) == len(y)
    assert 0 < sex["summary"]["approval_rate_ratio_min_max"] <= 1
    assert any("not evidence of legal" in t for t in r["limitations"])


def test_group_columns_fold_undocumented_codes():
    X = pd.DataFrame({"SEX": [1, 2], "EDUCATION": [0, 6], "MARRIAGE": [0, 2], "AGE": [25, 61]})
    g = group_columns(X)
    assert g["EDUCATION"].str.startswith("others").all()
    assert list(g["AGE_BAND"]) == ["21-29", "60+"]


def test_excluded_inputs_are_not_used(trained, fixture_df):
    """SEX, MARRIAGE and AGE are excluded: changing them must not change any prediction."""
    bundle = trained.champion.bundle
    X = fixture_df[FEATURE_COLUMNS].head(300).copy()
    p = bundle.predict_proba(X)
    X["SEX"], X["MARRIAGE"], X["AGE"] = 3 - X["SEX"], 2, 70
    np.testing.assert_array_equal(p, bundle.predict_proba(X))
