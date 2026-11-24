"""The benchmark's in-process part runs and asserts identical outputs on real-shaped rows."""

from credit_default.data.schema import FEATURE_COLUMNS
from credit_default.models.bundle import ModelBundle
from credit_default.perf import benchmark


def test_in_process_benchmark(bundle_path, fixture_df):
    bundle = ModelBundle.load(bundle_path)
    X = fixture_df[FEATURE_COLUMNS].head(300).reset_index(drop=True)
    r = benchmark.in_process(bundle, X, n_single=30)
    assert r["identical_on_test_split"] is True
    assert {"pipeline", "fast"} <= set(r["single"])
    assert [b["batch"] for b in r["batch"]] == [10, 100, 1000][: len(r["batch"])]
    assert r["single"]["fast"]["p50_ms"] > 0 and r["explain_single"]["n"] >= 50
