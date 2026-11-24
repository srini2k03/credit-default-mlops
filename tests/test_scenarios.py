"""Drift statistics and the monitoring policy on simulated scenarios."""

import numpy as np
import pytest

from credit_default.monitor import scenarios, stats


def test_psi_and_tests_behave():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert stats.psi_numeric(a, b) < 0.02
    assert stats.psi_numeric(a, b + 1.0) > 0.2
    assert stats.ks(a, b)[1] > 0.01 and stats.ks(a, b + 0.5)[1] < 1e-6
    cats = rng.choice([1, 2, 3], 5000)
    assert stats.psi_categorical(cats, cats) == pytest.approx(0.0)
    _, p, v = stats.chi2(cats, np.where(rng.random(5000) < 0.3, 4, cats))  # new category
    assert p < 1e-6 and v > 0.1
    assert stats.psi_band(0.25, 0.1, 0.2) == "major"
    z, pz = stats.proportion_z(220, 1000, 350, 1000)
    assert z > 0 and pz < 1e-6


def test_significance_alone_is_not_drift():
    """A tiny shift is significant with many rows but stays in the stable PSI band."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 200_000), rng.normal(0.03, 1, 200_000)
    assert stats.ks(a, b)[1] < 0.01
    assert stats.psi_band(stats.psi_numeric(a, b), 0.1, 0.2) == "stable"


@pytest.fixture(scope="module")
def results(cfg, fixture_df, trained):
    return scenarios.run_scenarios(cfg, fixture_df, trained.champion.bundle)


def _action(results, name, labelled=True):
    s = next(s for s in results["scenarios"] if s["scenario"] == name)
    return (s["decision"] if labelled else s["label_free_decision"])["action"]


def test_every_scenario_is_evaluated(results):
    assert [s["scenario"] for s in results["scenarios"]] == list(scenarios.SCENARIOS)
    assert "Drift scenarios" in scenarios.to_markdown(results)


def test_control_triggers_nothing(results):
    assert _action(results, "control") in {"none", "monitor"}
    assert _action(results, "control") != "retrain"


def test_data_quality_failures_block_instead_of_retraining(results):
    assert _action(results, "missing_values") == "block"
    assert _action(results, "new_categories") == "block"


def test_covariate_shift_is_flagged_without_labels(results):
    s = next(s for s in results["scenarios"] if s["scenario"] == "covariate_shift")
    assert s["n_drifted_features"] >= 2
    assert _action(results, "covariate_shift", labelled=False) == "retrain"


def test_label_shift_changes_label_rate(results):
    s = next(s for s in results["scenarios"] if s["scenario"] == "label_shift")
    assert s["labels"]["rate"] == pytest.approx(0.35, abs=0.02)
    assert s["labels"]["p_value"] < 0.01
