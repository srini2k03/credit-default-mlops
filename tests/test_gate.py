"""The quality gate must reject a worse challenger and accept an equal-or-better one."""

import json

import numpy as np
from conftest import overrides_for

from credit_default.data.split import make_splits
from credit_default.pipeline.train import fit_family
from credit_default.registry import gate


def _worse_bundle(cfg, fixture_df):
    """A deliberately bad model: logistic regression trained on permuted labels."""
    splits = make_splits(fixture_df, cfg.train.splits, cfg.seed)
    rng = np.random.default_rng(0)
    for part in (splits.train, splits.calibration, splits.validation):
        part["default_payment_next_month"] = rng.permutation(
            part["default_payment_next_month"].to_numpy()
        )
    bundle, _ = fit_family(cfg, "logreg", splits, {"max_iter": 500}, "class_weight")
    return bundle


def test_compare_rules():
    prod = {"pr_auc": 0.50, "business_cost": 0.60}
    assert gate.compare({"pr_auc": 0.51, "business_cost": 0.59}, prod, 0.005, 0.01).passed
    assert gate.compare(
        {"pr_auc": 0.497, "business_cost": 0.605}, prod, 0.005, 0.01
    ).passed  # within tolerance
    assert not gate.compare({"pr_auc": 0.48, "business_cost": 0.59}, prod, 0.005, 0.01).passed
    assert not gate.compare({"pr_auc": 0.51, "business_cost": 0.70}, prod, 0.005, 0.01).passed
    assert gate.compare(
        {"pr_auc": 0.1, "business_cost": 9.0}, None, 0.005, 0.01
    ).passed  # first model


def test_gate_cli_rejects_worse_model(cfg, fixture_df, bundle_path, workdir):
    worse_path = _worse_bundle(cfg, fixture_df).save(workdir / "worse" / "model.joblib")
    rc = gate.main(
        [*overrides_for(workdir), f"--challenger={worse_path}", f"--production={bundle_path}"]
    )
    assert rc == 1
    report = json.loads((workdir / "artifacts" / "run" / "gate.json").read_text())
    assert report["passed"] is False
    assert report["challenger"]["pr_auc"] < report["production"]["pr_auc"]


def test_gate_cli_accepts_better_or_equal_model(cfg, fixture_df, bundle_path, workdir):
    worse_path = _worse_bundle(cfg, fixture_df).save(workdir / "worse2" / "model.joblib")
    args = overrides_for(workdir)
    assert gate.main([*args, f"--challenger={bundle_path}", f"--production={worse_path}"]) == 0
    assert gate.main([*args, f"--challenger={bundle_path}", f"--production={bundle_path}"]) == 0


def test_gate_registry_mode_promotes(trained, cfg, workdir):
    from credit_default.registry import mlflow_registry as reg

    assert gate.main([*overrides_for(workdir), "--promote"]) == 0
    reg.setup_tracking(cfg)
    assert str(reg.get_version(cfg, cfg.registry.production_alias).version) == trained.model_version
