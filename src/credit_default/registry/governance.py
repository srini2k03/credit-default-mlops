"""Champion/challenger governance: decision history, rollback, and a simulated lifecycle.

Commands:
  record    append the latest gate decision (artifacts/run/gate.json) to the history file
            python -m credit_default.registry.governance record --trigger="..." --run-url=...
  rollback  restore the previous production model
            registry mode: move @production back to @previous-production
              python -m credit_default.registry.governance rollback --mode=registry
            git mode (what Render serves): restore models/bundled/ from the commit that
            produced the previous production model, recorded in the history
              python -m credit_default.registry.governance rollback --mode=git [--to=<sha>]
  demo      simulated lifecycle in a throw-away MLflow registry on the real splits:
            v1 baseline -> passing challenger promoted -> failing challengers rejected
            (label-noise feed, uncalibrated scores) -> incident -> rollback
            python -m credit_default.registry.governance demo [--out=results/governance]

The history is append-only JSON (``registry.history_path``); every promotion, rejection and
rollback is kept, including failed challengers.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.pipeline.steps import git_commit, now_iso, write_json

log = logging.getLogger(__name__)
SUMMARY_KEYS = ("roc_auc", "pr_auc", "brier", "ece", "business_cost", "recall", "threshold")


# ---------------------------------------------------------------------------- history
def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def append_event(path: Path, event: dict) -> dict:
    history = load_history(path)
    event = {"id": len(history) + 1, "ts": now_iso(), **event}
    history.append(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, default=str) + "\n")
    return event


def _summary(m: dict | None) -> dict | None:
    return None if m is None else {k: m[k] for k in SUMMARY_KEYS if k in m}


def record_gate(cfg, run_dir: Path, history: Path, trigger: str, run_url: str | None) -> dict:
    gate = json.loads((run_dir / "gate.json").read_text())
    metrics_path = run_dir / "metrics.json"
    m = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return append_event(
        history,
        {
            "event": "promote" if gate["passed"] else "reject",
            "source": "train workflow",
            "trigger": trigger,
            "run_url": run_url,
            "git_commit": git_commit(),
            "challenger": {
                "ref": gate.get("challenger_ref"),
                "family": m.get("champion"),
                "calibration": m.get("calibration"),
                **(_summary(gate["challenger"]) or {}),
            },
            "production_before": {
                "ref": gate.get("production_ref"),
                **(_summary(gate.get("production")) or {}),
            },
            "failed_checks": [
                c["name"] for c in gate.get("checks", []) if not c["passed"] and not c["skipped"]
            ],
            "reasons": gate["reasons"],
        },
    )


# ---------------------------------------------------------------------------- rollback
def rollback_git(
    cfg,
    history_path: Path,
    to: str | None = None,
    root: Path = REPO_ROOT,
    reason: str = "manual rollback of the served bundle",
) -> dict:
    """Restore models/bundled/ as it was at commit ``to`` (default: the bundle version before
    the current one, found with ``git log``) and record the rollback. Render redeploys main."""
    bundle_dir = Path(cfg.paths.bundle_dir)
    if to is None:
        shas = subprocess.check_output(
            ["git", "log", "--format=%H", "-n", "2", "--", str(bundle_dir / "model.joblib")],
            cwd=root,
            text=True,
        ).split()
        if len(shas) < 2:
            raise RuntimeError("git history has no earlier bundle to roll back to")
        to = shas[1]
    for name in ("model.joblib", "metadata.json"):
        blob = subprocess.check_output(["git", "show", f"{to}:{bundle_dir / name}"], cwd=root)
        (root / bundle_dir / name).write_bytes(blob)
    return append_event(
        history_path,
        {
            "event": "rollback",
            "source": "git",
            "restored_from_commit": to,
            "git_commit": git_commit(),
            "reasons": [reason],
        },
    )


def rollback_registry(cfg, history_path: Path, reason: str) -> dict:
    from credit_default.registry import mlflow_registry as reg

    reg.setup_tracking(cfg)
    frm, to = reg.rollback(cfg)
    return append_event(
        history_path,
        {
            "event": "rollback",
            "source": "registry",
            "from_version": frm,
            "to_version": to,
            "reasons": [reason],
        },
    )


# ---------------------------------------------------------------------------- demo
def run_demo(cfg, df: pd.DataFrame, out_dir: Path) -> dict:
    """Simulated champion/challenger lifecycle in a temporary MLflow registry (real splits)."""
    import mlflow

    from credit_default.data.schema import TARGET
    from credit_default.data.split import make_splits
    from credit_default.pipeline.train import calibrate, fit_family
    from credit_default.registry import gate as gate_mod
    from credit_default.registry import mlflow_registry as reg

    splits = make_splits(df, cfg.train.splits, cfg.seed)
    holdout = splits.xy(splits.test)
    family = "lightgbm"
    tuned = dict(cfg.models[family].params)
    metrics_path = REPO_ROOT / cfg.paths.results_dir / "metrics.json"
    if metrics_path.exists() and cfg.data.source == "uci":  # reuse the published tuned params
        published = json.loads(metrics_path.read_text())
        for f in published.get("families", []):
            if f["family"] == family:
                tuned = f["params"]
    strategy = cfg.train.imbalance
    rng = np.random.default_rng(cfg.seed)

    def variant(train_frac=1.0, params=None, label_noise=0.0):
        s = make_splits(df, cfg.train.splits, cfg.seed)
        tr = s.train.sample(frac=train_frac, random_state=cfg.seed).reset_index(drop=True)
        if label_noise:
            pos = np.flatnonzero(tr[TARGET].to_numpy() == 1)
            flip = rng.choice(pos, int(label_noise * len(pos)), replace=False)
            tr.loc[flip, TARGET] = 0  # defaults recorded as repaid: a broken outcome feed
        s.train = tr
        return fit_family(cfg, family, s, params or tuned, strategy)[0]

    steps = [
        (
            "v1 baseline",
            "First model: default hyper-parameters, 50% of the training rows.",
            lambda: variant(0.5, dict(cfg.models[family].params)),
        ),
        (
            "v2 retrained",
            "Tuned hyper-parameters, all training rows (new labelled outcomes).",
            lambda: variant(1.0),
        ),
        (
            "v3 noisy labels",
            "Same recipe, but the outcome feed recorded 40% of defaults as repaid.",
            lambda: variant(1.0, label_noise=0.4),
        ),
        (
            "v4 uncalibrated",
            "Same base model as v2 served without calibration (scores inflated by class weights).",
            None,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        demo_cfg = load_config(
            [
                f"registry.tracking_uri='sqlite:///{tmp}/mlflow.db'",
                f"registry.artifact_location={tmp}/artifacts",
                "registry.experiment=governance-demo",
            ]
        )
        reg.setup_tracking(demo_cfg)
        history_path = Path(tmp) / "history.json"
        bundles, timeline = {}, []
        for name, desc, make in steps:
            if make is None:  # v4: v2's base model without a calibrator
                b2 = bundles["v2 retrained"]
                Xv, yv = splits.xy(splits.validation)
                from copy import copy

                from credit_default.decision.engine import CostPolicy, choose_threshold

                bundle = copy(b2)
                bundle.calibrator = calibrate(b2.pipeline, "none", None, None)
                bundle.calibration = "none"
                bundle.threshold, _ = choose_threshold(
                    yv, bundle.predict_proba(Xv), CostPolicy.from_config(cfg.cost)
                )
            else:
                bundle = make()
            bundles[name] = bundle
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("demo", "true")
                info = reg.log_bundle(bundle, demo_cfg.registry.model_name)
            version = str(info.registered_model_version)
            reg.set_alias(demo_cfg, version, demo_cfg.registry.staging_alias)
            current = reg.load_alias(demo_cfg, demo_cfg.registry.production_alias)
            decision = gate_mod.run_gate(cfg, bundle, current[0] if current else None, holdout)
            before = reg.get_version(demo_cfg, demo_cfg.registry.production_alias)
            if decision.passed:
                reg.promote(demo_cfg, version)
            after = reg.get_version(demo_cfg, demo_cfg.registry.production_alias)
            event = append_event(
                history_path,
                {
                    "event": "promote" if decision.passed else "reject",
                    "source": "simulation",
                    "candidate": name,
                    "description": desc,
                    "version": version,
                    "production_before": None if before is None else str(before.version),
                    "production_after": None if after is None else str(after.version),
                    "challenger": _summary(decision.challenger),
                    "champion": _summary(decision.production),
                    "checks": decision.checks,
                    "reasons": decision.reasons,
                },
            )
            timeline.append(event)
            log.info("%s -> %s (%s)", name, event["event"], "; ".join(decision.reasons))
        # Incident after the v2 promotion: roll back to the previous production version.
        rb = rollback_registry(
            demo_cfg,
            history_path,
            "simulated incident: a downstream system reports an unexpected decline-rate jump "
            "after v2 went live; roll back while it is investigated",
        )
        timeline.append(rb)
        final = reg.get_version(demo_cfg, demo_cfg.registry.production_alias)
        aliases = {
            a: str(reg.get_version(demo_cfg, a).version)
            for a in (demo_cfg.registry.production_alias, demo_cfg.registry.previous_alias)
            if reg.get_version(demo_cfg, a) is not None
        }
    result = {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "family": family,
        "note": "Simulation in a temporary MLflow registry on the real train/test splits; the "
        "served model is not touched.",
        "timeline": timeline,
        "final_production_version": str(final.version),
        "final_aliases": aliases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, out_dir / "demo.json")
    (out_dir / "demo.md").write_text(demo_markdown(result))
    return result


def demo_markdown(r: dict) -> str:
    lines = [
        "# Champion/challenger simulation",
        "",
        f"Generated {r['generated_at']} at commit `{r['git_commit'][:10]}`. {r['note']}",
        "",
        "| # | Event | Candidate | Version | Production before -> after | PR-AUC | Brier | ECE | "
        "Cost / applicant | Failed checks |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in r["timeline"]:
        if e["event"] == "rollback":
            lines.append(
                f"| {e['id']} | **rollback** | - | - | v{e['from_version']} -> v{e['to_version']} "
                "| - | - | - | - | - |"
            )
            continue
        c = e["challenger"]
        failed = [ch["name"] for ch in e["checks"] if not ch["passed"] and not ch["skipped"]]
        lines.append(
            f"| {e['id']} | **{e['event']}** | {e['candidate']} | v{e['version']} | "
            f"{'none' if e['production_before'] is None else 'v' + e['production_before']} -> "
            f"v{e['production_after']} | {c['pr_auc']:.4f} | {c['brier']:.4f} | {c['ece']:.4f} | "
            f"{c['business_cost']:.4f} | {', '.join(failed) or '-'} |"
        )
    lines += ["", "## Candidates", ""]
    for e in r["timeline"]:
        if e["event"] != "rollback":
            lines.append(f"- **{e['candidate']}**: {e['description']} -> {e['event']}")
        else:
            lines.append(f"- **rollback**: {e['reasons'][0]}")
    lines += [
        "",
        f"Final aliases: {', '.join(f'@{k} -> v{v}' for k, v in r['final_aliases'].items())}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in {"record", "rollback", "demo"}:
        print(__doc__)
        return 2
    command, rest = args[0], args[1:]
    overrides, flags = split_cli(rest)
    opts = dict(f.lstrip("-").split("=", 1) for f in flags if "=" in f)
    cfg = load_config(overrides)
    history = REPO_ROOT / cfg.registry.history_path
    if command == "record":
        run_dir = REPO_ROOT / cfg.paths.artifacts_dir / "run"
        event = record_gate(
            cfg, run_dir, history, opts.get("trigger", "manual"), opts.get("run-url")
        )
        log.info("recorded %s (#%s)", event["event"], event["id"])
        return 0
    if command == "rollback":
        if opts.get("mode", "git") == "registry":
            event = rollback_registry(cfg, history, opts.get("reason", "manual rollback"))
        else:
            event = rollback_git(
                cfg, history, opts.get("to"), reason=opts.get("reason", "manual rollback")
            )
        log.info("rollback recorded (#%s)", event["id"])
        return 0
    from credit_default.validation.schema import validate

    df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    out = Path(opts.get("out", REPO_ROOT / cfg.paths.results_dir / "governance"))
    if cfg.data.source != "uci" and "out" not in opts:
        out = REPO_ROOT / cfg.paths.artifacts_dir / "governance"
    run_demo(cfg, df, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
