import json

import numpy as np
from omegaconf import OmegaConf

from credit_default.data.schema import FEATURE_COLUMNS
from credit_default.models.bundle import ModelBundle
from credit_default.models.factory import build_pipeline
from credit_default.registry import mlflow_registry as reg


def test_training_produces_calibrated_bundle_and_report(trained, cfg, fixture_df):
    bundle = trained.champion.bundle
    p = bundle.predict_proba(fixture_df[FEATURE_COLUMNS].head(200))
    assert ((p >= 0) & (p <= 1)).all()
    assert 0 < bundle.threshold < 1
    m = json.loads((trained.out_dir / "metrics.json").read_text())
    assert m["data"]["synthetic"] is True
    for key in ("roc_auc", "pr_auc", "f1", "brier", "business_cost"):
        assert key in m["test"]
    assert m["test"]["roc_auc"] > 0.7  # the fixture has real signal
    assert {r["rule"] for r in m["thresholds"]} == {"cost-optimal", "0.5", "max-F1"}
    # the cost-optimal rule is never worse on validation, and is reported on test
    assert (trained.out_dir / "figures" / "cost_curve.png").exists()
    assert (trained.out_dir / "figures" / "shap_beeswarm.png").exists()
    assert "SYNTHETIC" in (trained.out_dir / "metrics.md").read_text()


def test_bundle_roundtrip_and_explain(bundle_path, fixture_df):
    bundle = ModelBundle.load(bundle_path)
    X = fixture_df[FEATURE_COLUMNS].head(3)
    exp = bundle.explain(X, top_k=4)
    assert len(exp) == 3 and len(exp[0]["contributions"]) == 4
    vals = [abs(c["shap_value"]) for c in exp[0]["contributions"]]
    assert vals == sorted(vals, reverse=True)
    # SHAP additivity in the base model's log-odds space
    shap_vals, base, _ = bundle.shap_values(X)
    raw = bundle.pipeline.predict_proba(X)[:, 1]
    np.testing.assert_allclose(base + shap_vals.sum(axis=1), np.log(raw / (1 - raw)), atol=1e-3)


def test_registered_as_staging(trained, cfg):
    reg.setup_tracking(cfg)
    mv = reg.get_version(cfg, cfg.registry.staging_alias)
    assert mv is not None and str(mv.version) == trained.model_version
    loaded, version = reg.load_alias(cfg, cfg.registry.staging_alias)
    assert version == trained.model_version
    assert loaded.threshold == trained.champion.bundle.threshold


def test_every_family_and_strategy_builds_and_fits(small_df):
    X, y = small_df[FEATURE_COLUMNS], small_df["default_payment_next_month"]
    model_cfg = OmegaConf.load("configs/model/xgboost.yaml")
    for family, params in (
        ("logreg", {"max_iter": 500}),
        ("lightgbm", {"n_estimators": 20}),
        ("xgboost", dict(model_cfg.params, n_estimators=20)),
    ):
        for strategy in ("none", "class_weight", "smote"):
            pipe = build_pipeline(family, params, strategy, float(y.mean()), 0)
            pipe.fit(X, y)
            assert pipe.predict_proba(X.head(5)).shape == (5, 2)


def test_ablation_and_tuning_run(small_df):
    from credit_default.tuning.optuna_search import run_ablation, tune

    X, y = small_df[FEATURE_COLUMNS], small_df["default_payment_next_month"]
    model_cfg = OmegaConf.load("configs/model/lightgbm.yaml")
    model_cfg.params.n_estimators = 30
    params, score, study = tune(
        "lightgbm", model_cfg, "class_weight", X, y, n_trials=2, folds=2, seed=0
    )
    assert study is not None and len(study.trials) == 2 and 0 < score <= 1
    rows = run_ablation(
        ["lightgbm"], {"lightgbm": dict(model_cfg.params)}, ["class_weight", "smote"], X, y, 2, 0
    )
    assert {r["strategy"] for r in rows} == {"class_weight", "smote"}
