"""Quality gate: the challenger must not be worse than Production on the same test holdout.

Both models are re-scored on the current test split, each at its own decision threshold.
The challenger is rejected (exit code 1) if
  PR-AUC        < production PR-AUC - pr_auc_tolerance, or
  business cost > production cost * (1 + cost_tolerance).
With no Production model yet, the challenger passes.

Usage:
  python -m credit_default.registry.gate [overrides]                 # staging vs production aliases
  python -m credit_default.registry.gate --promote                   # move @production on pass
  python -m credit_default.registry.gate --challenger=a.joblib --production=b.joblib
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.split import make_splits
from credit_default.evaluate.metrics import evaluate_probs
from credit_default.models.bundle import ModelBundle
from credit_default.pipeline.steps import write_json
from credit_default.validation.schema import validate

log = logging.getLogger(__name__)


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    challenger: dict = field(default_factory=dict)
    production: dict | None = None


def compare(
    challenger: dict, production: dict | None, pr_auc_tol: float, cost_tol: float
) -> GateDecision:
    if production is None:
        return GateDecision(
            True, ["no production model: challenger accepted as the first version"], challenger
        )
    reasons = []
    floor = production["pr_auc"] - pr_auc_tol
    if challenger["pr_auc"] < floor:
        reasons.append(
            f"PR-AUC {challenger['pr_auc']:.4f} < production {production['pr_auc']:.4f} - {pr_auc_tol}"
        )
    ceiling = production["business_cost"] * (1 + cost_tol)
    if challenger["business_cost"] > ceiling:
        reasons.append(
            f"cost {challenger['business_cost']:.4f} > production "
            f"{production['business_cost']:.4f} x (1 + {cost_tol})"
        )
    passed = not reasons
    if passed:
        reasons.append("challenger within tolerance of production on PR-AUC and business cost")
    return GateDecision(passed, reasons, challenger, production)


def holdout(cfg) -> tuple[pd.DataFrame, pd.Series]:
    df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    splits = make_splits(df, cfg.train.splits, cfg.seed)
    return splits.xy(splits.test)


def score(bundle: ModelBundle, X, y, cfg) -> dict:
    return evaluate_probs(
        y, bundle.predict_proba(X), bundle.threshold, cfg.cost.fn_cost, cfg.cost.fp_cost
    )


def run_gate(cfg, challenger: ModelBundle, production: ModelBundle | None) -> GateDecision:
    X, y = holdout(cfg)
    c = score(challenger, X, y, cfg)
    p = score(production, X, y, cfg) if production is not None else None
    g = cfg.registry.gate
    return compare(c, p, g.pr_auc_tolerance, g.cost_tolerance)


def write_report(decision: GateDecision, out_dir: Path, meta: dict) -> None:
    write_json({**asdict(decision), **meta}, out_dir / "gate.json")
    rows = ["| Metric | Challenger | Production |", "|---|---|---|"]
    for k in ("pr_auc", "roc_auc", "business_cost", "brier", "f1", "threshold"):
        p = decision.production[k] if decision.production else None
        rows.append(f"| {k} | {decision.challenger[k]:.4f} | {'-' if p is None else f'{p:.4f}'} |")
    text = "\n".join(
        [
            "# Quality gate",
            "",
            f"**Decision: {'PASS' if decision.passed else 'REJECT'}**",
            "",
            *[f"- {r}" for r in decision.reasons],
            "",
            f"Challenger: {meta.get('challenger_ref')}  ",
            f"Production: {meta.get('production_ref')}",
            "",
            *rows,
            "",
        ]
    )
    (out_dir / "gate.md").write_text(text)


def _flag(flags: list[str], name: str) -> str | None:
    for f in flags:
        if f.startswith(f"--{name}="):
            return f.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    flags = [a for a in args if a.startswith("--")]
    overrides, _ = split_cli([a for a in args if not a.startswith("--")])
    cfg = load_config(overrides)

    challenger_path, production_path = _flag(flags, "challenger"), _flag(flags, "production")
    challenger_version = None
    if challenger_path:
        challenger, c_label = ModelBundle.load(challenger_path), challenger_path
    else:
        from credit_default.registry import mlflow_registry as reg

        reg.setup_tracking(cfg)
        loaded = reg.load_alias(cfg, cfg.registry.staging_alias)
        if loaded is None:
            log.error("no model under alias @%s", cfg.registry.staging_alias)
            return 2
        challenger, challenger_version = loaded
        c_label = f"models:/{cfg.registry.model_name}@{cfg.registry.staging_alias} (v{challenger_version})"

    if production_path:
        production = ModelBundle.load(production_path) if Path(production_path).exists() else None
        p_label = production_path if production is not None else "none"
    else:
        from credit_default.registry import mlflow_registry as reg

        reg.setup_tracking(cfg)
        loaded = reg.load_alias(cfg, cfg.registry.production_alias)
        production, p_label = (
            (None, "none")
            if loaded is None
            else (
                loaded[0],
                f"models:/{cfg.registry.model_name}@{cfg.registry.production_alias} (v{loaded[1]})",
            )
        )

    decision = run_gate(cfg, challenger, production)
    out_dir = REPO_ROOT / cfg.paths.artifacts_dir / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(decision, out_dir, {"challenger_ref": c_label, "production_ref": p_label})
    for r in decision.reasons:
        log.info("gate: %s", r)

    if not decision.passed:
        log.error("QUALITY GATE REJECTED the challenger (%s)", c_label)
        return 1
    log.info("QUALITY GATE PASSED (%s)", c_label)
    if "--promote" in flags:
        if challenger_version is None:
            log.warning("--promote needs a registry challenger; nothing promoted")
        else:
            from credit_default.registry import mlflow_registry as reg

            reg.set_alias(cfg, challenger_version, cfg.registry.production_alias)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
