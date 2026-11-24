"""Data-quality report: duplicates, undocumented codes, leakage screen, schema failures."""

import numpy as np
import pandas as pd

from credit_default.data import quality
from credit_default.data.schema import TARGET


def _check(report, prefix):
    return next(c for c in report["leakage"]["checks"] if c["check"].startswith(prefix))


def test_report_on_fixture(cfg, fixture_df, tmp_path):
    report = quality.build_report(fixture_df, cfg, {"synthetic": True, "path": "fixture"})
    assert report["schema"]["passed"]
    assert report["n_rows"] == len(fixture_df)
    assert all(c["passed"] for c in report["leakage"]["checks"])
    rates = report["leakage"]["split_default_rates"]
    assert sum(v["n"] for v in rates.values()) == len(fixture_df)
    j, m = quality.write_report(report, tmp_path)
    assert j.exists() and "SYNTHETIC FIXTURE" in m.read_text()


def test_duplicates_and_conflicting_labels(cfg, fixture_df):
    df = fixture_df.copy()
    dup = df.iloc[[0, 1]].copy()
    dup["ID"] = df["ID"].max() + np.arange(1, 3)
    dup.loc[dup.index[0], TARGET] = 1 - dup.loc[dup.index[0], TARGET]  # conflicting label
    df2 = pd.concat([df, dup], ignore_index=True)
    d = quality._duplicates(df2)
    assert d["duplicate_ids"] == 0
    assert d["duplicate_feature_vectors"] >= 2
    assert d["groups_with_conflicting_labels"] >= 1


def test_leaky_feature_is_flagged(cfg, fixture_df):
    df = fixture_df.copy()
    df["PAY_0"] = np.where(df[TARGET] == 1, 8, -1)  # the label in disguise
    report = quality.build_report(df, cfg)
    assert not _check(report, "no single feature")["passed"]
    assert report["leakage"]["univariate_auc_top"][0]["feature"] == "PAY_0"


def test_schema_failure_is_reported_not_raised(cfg, fixture_df):
    df = fixture_df.copy()
    df.loc[0, "SEX"] = 7
    report = quality.build_report(df, cfg)
    assert report["schema"]["passed"] is False and "SEX" in report["schema"]["detail"]
