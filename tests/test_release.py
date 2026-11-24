"""Regression tests on the committed release: the served bundle, its metadata and the published
results must describe the same model, and the serving shortcuts must not change its output."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_default.data.schema import FEATURE_COLUMNS
from credit_default.models.bundle import ModelBundle
from credit_default.serve.fast import build_fast_scorer
from credit_default.serve.schemas import EXAMPLE, LOW_RISK_EXAMPLE

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "bundled" / "model.joblib"
METRICS = ROOT / "results" / "metrics.json"

pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and METRICS.exists()), reason="no committed release"
)


@pytest.fixture(scope="module")
def bundle():
    return ModelBundle.load(BUNDLE)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


def test_bundle_matches_published_results(bundle, metrics):
    meta = json.loads((ROOT / "models" / "bundled" / "metadata.json").read_text())
    assert metrics["data"]["source"] == "uci" and not metrics["data"]["synthetic"]
    assert bundle.family == metrics["champion"] == meta["family"]
    assert bundle.threshold == metrics["test"]["threshold"] == meta["threshold"]
    assert meta["test"] == bundle.metadata["test"] == metrics["test"]
    assert meta["data"]["sha256"] == metrics["data"]["sha256"]


def test_published_predictions_belong_to_the_bundle(bundle, metrics):
    preds = ROOT / "results" / "predictions.json"
    if not preds.exists():
        pytest.skip("no exported predictions")
    p = json.loads(preds.read_text())
    assert p["champion"] == bundle.family and p["threshold"] == bundle.threshold
    y, prob = np.array(p["test"]["y"]), np.array(p["test"]["p"])
    assert len(y) == metrics["test"]["n"]
    decline = prob >= bundle.threshold
    tp, fp = int((decline & (y == 1)).sum()), int((decline & (y == 0)).sum())
    assert (tp, fp) == (metrics["test"]["tp"], metrics["test"]["fp"])


def test_bundle_scores_examples_and_fast_path_is_exact(bundle):
    X = pd.concat([bundle.background, pd.DataFrame([EXAMPLE, LOW_RISK_EXAMPLE])], ignore_index=True)
    p = bundle.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))
    late, low = p[-2], p[-1]
    assert late > bundle.threshold > low  # the Swagger examples keep their documented outcome
    scorer = build_fast_scorer(bundle, X)
    assert scorer is not None
    np.testing.assert_array_equal(scorer.predict_proba(X[FEATURE_COLUMNS].to_numpy(float)), p)


def test_excluded_inputs_do_not_move_the_served_model(bundle):
    excluded = bundle.metadata.get("excluded_features", [])
    if not excluded:
        pytest.skip("this bundle uses every input")
    X = bundle.background.copy()
    base = bundle.predict_proba(X)
    X["SEX"], X["MARRIAGE"], X["AGE"] = 3 - X["SEX"], 2, 60
    np.testing.assert_array_equal(bundle.predict_proba(X), base)
