import json

import pytest
from conftest import overrides_for

from credit_default.monitor import check_drift
from credit_default.monitor.drift import run_report
from credit_default.monitor.simulate import build_slices


def test_slices_shift_only_the_drifted_slice(cfg, fixture_df):
    s = build_slices(cfg, fixture_df)
    assert len(s["control"]) == len(s["drifted"]) <= 400
    assert "default_payment_next_month" not in s["drifted"].columns
    assert s["drifted"]["LIMIT_BAL"].mean() < s["control"]["LIMIT_BAL"].mean()
    assert s["drifted"]["AGE"].mean() < s["control"]["AGE"].mean()


def test_drift_report_detects_drifted_not_control(cfg, fixture_df, tmp_path):
    s = build_slices(cfg, fixture_df)
    control = run_report(s["reference"], s["control"], "control", tmp_path, 0.3)
    drifted = run_report(s["reference"], s["drifted"], "drifted", tmp_path, 0.3)
    assert not control["drift_detected"]
    assert drifted["drift_detected"]
    assert (tmp_path / "drift_drifted.html").stat().st_size > 10_000
    assert json.loads((tmp_path / "drift_control.json").read_text())["slice"] == "control"


@pytest.mark.parametrize("slice_name, expected", [("control", 0), ("drifted", 1)])
def test_check_drift_exit_codes(workdir, cfg, slice_name, expected, tmp_path, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert check_drift.main([*overrides_for(workdir), f"--slice={slice_name}"]) == expected
    assert f"drift={'true' if expected else 'false'}" in out.read_text()
