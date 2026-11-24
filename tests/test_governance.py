"""Promotion policy, champion protection, history and rollback."""

import json
import subprocess

import pytest
from conftest import overrides_for
from omegaconf import OmegaConf

from credit_default.config import load_config
from credit_default.models.bundle import ModelBundle
from credit_default.registry import gate, governance, policy
from credit_default.registry import mlflow_registry as reg

GATE = OmegaConf.create(
    {
        "min_roc_auc": 0.70,
        "min_pr_auc": 0.45,
        "pr_auc_tolerance": 0.005,
        "max_ece": 0.05,
        "brier_tolerance": 0.002,
        "cost_tolerance": 0.01,
        "max_p95_latency_ms": 50,
        "max_bundle_mb": 50,
    }
)
GOOD = {"roc_auc": 0.78, "pr_auc": 0.55, "brier": 0.135, "ece": 0.02, "business_cost": 0.56}
OPS = {
    "contract_ok": True,
    "finite_in_range": True,
    "deterministic": True,
    "threshold": 0.18,
    "p95_latency_ms": 10.0,
    "bundle_mb": 0.5,
}


def _isolated(workdir, tmp_path):
    """Overrides for the shared fixture data, but a fresh registry and artifacts directory."""
    replace = {
        "registry.tracking_uri": f"'sqlite:///{tmp_path}/registry.db'",
        "registry.artifact_location": f"{tmp_path}/mlartifacts",
        "paths.artifacts_dir": f"{tmp_path}/artifacts",
    }
    keep = [o for o in overrides_for(workdir) if o.split("=", 1)[0] not in replace]
    return keep + [f"{k}={v}" for k, v in replace.items()]


def _failed(result):
    return {c.name for c in result.failures}


def test_policy_passes_equal_challenger():
    assert policy.evaluate(GOOD, GOOD, OPS, GATE).passed


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"pr_auc": 0.53}, {"pr_auc_vs_champion"}),
        ({"ece": 0.08}, {"ece_ceiling"}),
        ({"brier": 0.140}, {"brier_vs_champion"}),
        ({"business_cost": 0.58}, {"cost_vs_champion"}),
        ({"roc_auc": 0.65}, {"roc_auc_floor"}),
    ],
)
def test_policy_rejects_each_regression(change, failed):
    result = policy.evaluate({**GOOD, **change}, GOOD, OPS, GATE)
    assert not result.passed and _failed(result) == failed


def test_policy_operational_checks():
    slow = policy.evaluate(GOOD, GOOD, {**OPS, "p95_latency_ms": 80.0}, GATE)
    assert _failed(slow) == {"p95_latency_ms"}
    broken = policy.evaluate(GOOD, GOOD, {**OPS, "contract_ok": False}, GATE)
    assert _failed(broken) == {"input_contract"}


def test_first_model_skips_comparisons_but_keeps_floors():
    result = policy.evaluate(GOOD, None, OPS, GATE)
    assert result.passed and sum(c.skipped for c in result.checks) == 3
    assert not policy.evaluate({**GOOD, "ece": 0.2}, None, OPS, GATE).passed


def test_failed_challenger_cannot_replace_champion(cfg, fixture_df, bundle_path, workdir, tmp_path):
    """Registry mode: a worse @staging model is rejected and @production does not move."""
    import mlflow
    from test_gate import _worse_bundle

    args = _isolated(workdir, tmp_path)
    cfg = load_config(args)
    reg.setup_tracking(cfg)
    with mlflow.start_run(run_name="champion"):
        good_v = reg.log_bundle(
            ModelBundle.load(bundle_path), cfg.registry.model_name
        ).registered_model_version
    reg.promote(cfg, good_v)
    with mlflow.start_run(run_name="bad-challenger"):
        bad_v = reg.log_bundle(
            _worse_bundle(cfg, fixture_df), cfg.registry.model_name
        ).registered_model_version
    reg.set_alias(cfg, bad_v, cfg.registry.staging_alias)

    rc = gate.main([*args, "--promote"])

    assert rc == 1
    assert str(reg.get_version(cfg, cfg.registry.production_alias).version) == str(good_v)
    report = json.loads((tmp_path / "artifacts" / "run" / "gate.json").read_text())
    assert report["passed"] is False and any(not c["passed"] for c in report["checks"])


def test_promote_and_rollback_move_aliases(bundle_path, workdir, tmp_path):
    import mlflow

    cfg = load_config(_isolated(workdir, tmp_path))
    reg.setup_tracking(cfg)
    versions = []
    for name in ("a", "b"):
        with mlflow.start_run(run_name=name):
            versions.append(
                str(
                    reg.log_bundle(
                        ModelBundle.load(bundle_path), cfg.registry.model_name
                    ).registered_model_version
                )
            )
    reg.promote(cfg, versions[0])
    previous = reg.promote(cfg, versions[1])
    assert previous == versions[0]
    frm, to = reg.rollback(cfg)
    assert (frm, to) == (versions[1], versions[0])
    assert str(reg.get_version(cfg, cfg.registry.production_alias).version) == versions[0]
    assert str(reg.get_version(cfg, cfg.registry.previous_alias).version) == versions[1]


def test_history_is_append_only(tmp_path):
    path = tmp_path / "history.json"
    governance.append_event(path, {"event": "promote"})
    governance.append_event(path, {"event": "reject"})
    history = governance.load_history(path)
    assert [e["id"] for e in history] == [1, 2]
    assert [e["event"] for e in history] == ["promote", "reject"]


def test_record_gate_keeps_rejections(tmp_path, cfg):
    run = tmp_path / "run"
    run.mkdir()
    (run / "gate.json").write_text(
        json.dumps(
            {
                "passed": False,
                "reasons": ["[FAIL] performance/pr_auc_vs_champion"],
                "challenger": GOOD,
                "production": GOOD,
                "checks": [{"name": "pr_auc_vs_champion", "passed": False, "skipped": False}],
                "challenger_ref": "models:/credit-default@staging (v2)",
                "production_ref": "v1",
            }
        )
    )
    event = governance.record_gate(cfg, run, tmp_path / "h.json", "test", None)
    assert event["event"] == "reject" and event["failed_checks"] == ["pr_auc_vs_champion"]


def test_git_rollback_restores_previous_bundle(tmp_path):
    repo = tmp_path / "repo"
    (repo / "bundled").mkdir(parents=True)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    for content in ("v1", "v2"):
        (repo / "bundled" / "model.joblib").write_text(content)
        (repo / "bundled" / "metadata.json").write_text(json.dumps({"v": content}))
        git("add", ".")
        git("commit", "-q", "-m", content)
    cfg = load_config(["paths.bundle_dir=bundled"])
    event = governance.rollback_git(cfg, tmp_path / "h.json", root=repo)
    assert (repo / "bundled" / "model.joblib").read_text() == "v1"
    assert event["event"] == "rollback" and len(event["restored_from_commit"]) == 40
