"""Drift scenarios and the monitoring policy that turns them into actions.

The UCI table has no timestamps, so incoming batches are simulated from the test split (the
model never trained on it). Each scenario is one documented failure mode:

  control           untouched random sample: expected outcome "no action"
  covariate_shift   population shift (limits x0.7, ages -6 y, PAY_0 worse for 35 %, bills x1.25)
  missing_values    an upstream feed drops 15 % of BILL_AMT1-3 and PAY_AMT1 values
  new_categories    a new EDUCATION / MARRIAGE coding appears on 10 % of rows
  riskier_mix       acquisition mix moves to applicants already behind on payments
                    (feature and prediction drift, labels follow)
  label_shift       default rate rises to ~35 % with the same P(features | label) (prior shift)
  concept_drift     P(default | features) changes for highly utilised customers who pay on time
                    (no feature drift at all; only labelled performance reveals it)

The policy (``configs/monitor/default.yaml`` -> ``policy``) decides per batch:
  block      data-quality failure (missing rate, unseen categories): fix the feed, do not
             retrain on broken data, and do not trust the batch's scores
  retrain    feature drift share, prediction drift or labelled performance beyond limits
  monitor    something moved, but below the action limits
  none       nothing moved
Statistical significance alone never triggers an action.

Usage: python -m credit_default.monitor.scenarios [overrides] [--out=results/monitoring]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import (
    BILL_AMT,
    CATEGORICAL,
    DOMAINS,
    FEATURE_COLUMNS,
    PAY_STATUS,
    TARGET,
)
from credit_default.data.split import make_splits
from credit_default.decision.engine import CostPolicy, choose_threshold
from credit_default.evaluate.metrics import evaluate_probs
from credit_default.features.build import engineer
from credit_default.monitor import stats
from credit_default.monitor.simulate import apply_shift
from credit_default.pipeline.steps import git_commit, now_iso, write_json

log = logging.getLogger(__name__)

DISCRETE = [*CATEGORICAL, *PAY_STATUS]  # chi-square + categorical PSI
CONTINUOUS = [c for c in FEATURE_COLUMNS if c not in DISCRETE]
SCENARIOS = (
    "control",
    "covariate_shift",
    "missing_values",
    "new_categories",
    "riskier_mix",
    "label_shift",
    "concept_drift",
)
DESCRIPTIONS = {
    "control": "Untouched random sample of the test split.",
    "covariate_shift": "Credit limits x0.7, ages -6 years, PAY_0 one month worse for 35% of "
    "rows, bill amounts x1.25.",
    "missing_values": "15% of BILL_AMT1-3 and PAY_AMT1 values missing (upstream feed failure).",
    "new_categories": "10% of rows carry EDUCATION 7/8 or MARRIAGE 4, codes the model never saw.",
    "riskier_mix": "Resampled so 60% of applicants are already 1+ months late on PAY_0 "
    "(about 23% in the reference).",
    "label_shift": "Resampled to a ~35% default rate, features drawn from the same per-class "
    "distributions (prior shift).",
    "concept_drift": "Same applicants, but 30% of non-defaulters with utilisation >= 80% and "
    "PAY_0 <= 0 now default.",
}


# ---------------------------------------------------------------------------- scenarios
def make_scenario(name: str, pool: pd.DataFrame, n: int, cfg, seed: int) -> pd.DataFrame:
    """A labelled batch of ``n`` rows drawn from ``pool`` (the test split) for one scenario."""
    rng = np.random.default_rng(seed)
    base = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
    if name == "control":
        return base
    if name == "covariate_shift":
        shifted = apply_shift(base[FEATURE_COLUMNS], cfg.monitor.drifted_shift, seed)
        return pd.concat([shifted, base[[TARGET]]], axis=1)
    if name == "missing_values":
        out = base.astype({c: "float64" for c in [*BILL_AMT[:3], "PAY_AMT1"]})
        for c in [*BILL_AMT[:3], "PAY_AMT1"]:
            out.loc[rng.random(len(out)) < 0.15, c] = np.nan
        return out
    if name == "new_categories":
        out = base.copy()
        hit = rng.random(len(out)) < 0.10
        edu = rng.random(len(out)) < 0.5
        out.loc[hit & edu, "EDUCATION"] = rng.choice([7, 8], size=int((hit & edu).sum()))
        out.loc[hit & ~edu, "MARRIAGE"] = 4
        return out
    if name == "riskier_mix":
        late = pool[pool["PAY_0"] >= 1]
        ok = pool[pool["PAY_0"] < 1]
        k = int(0.6 * n)
        return (
            pd.concat(
                [
                    late.sample(n=k, replace=len(late) < k, random_state=seed),
                    ok.sample(n=n - k, replace=len(ok) < n - k, random_state=seed),
                ]
            )
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    if name == "label_shift":
        pos, neg = pool[pool[TARGET] == 1], pool[pool[TARGET] == 0]
        k = int(0.35 * n)
        return (
            pd.concat(
                [
                    pos.sample(n=k, replace=len(pos) < k, random_state=seed),
                    neg.sample(n=n - k, replace=len(neg) < n - k, random_state=seed),
                ]
            )
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    if name == "concept_drift":
        out = base.copy()
        f = engineer(out[FEATURE_COLUMNS])
        target = (f["util_1"] >= 0.8) & (out["PAY_0"] <= 0) & (out[TARGET] == 0)
        flip = target & (rng.random(len(out)) < 0.30)
        out.loc[flip, TARGET] = 1
        return out
    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")


# ---------------------------------------------------------------------------- checks
def data_quality(batch: pd.DataFrame) -> dict:
    X = batch[FEATURE_COLUMNS]
    missing = X.isna().mean()
    unseen = {}
    for col in ("EDUCATION", "MARRIAGE", "SEX"):
        vals = X[col].dropna()
        bad = ~vals.isin(DOMAINS[col])
        if bad.any():
            unseen[col] = sorted(int(v) for v in vals[bad].unique())
    pay = X[PAY_STATUS].to_numpy()
    out_of_range = int(np.sum((pay < -2) | (pay > 8)))
    return {
        "max_missing_rate": float(missing.max()),
        "columns_with_missing": {c: float(v) for c, v in missing.items() if v > 0},
        "unseen_categories": unseen,
        "unseen_category_rows": int(sum((~X[c].dropna().isin(DOMAINS[c])).sum() for c in unseen)),
        "pay_status_out_of_range": out_of_range,
    }


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, pol) -> list[dict]:
    rows = []
    for col in FEATURE_COLUMNS:
        if col in DISCRETE:
            psi = stats.psi_categorical(ref[col], cur[col])
            stat, p, effect = stats.chi2(ref[col], cur[col])
            test = "chi-square"
        else:
            psi = stats.psi_numeric(ref[col], cur[col])
            stat, p = stats.ks(ref[col], cur[col])
            effect, test = stat, "KS"
        band = stats.psi_band(psi, pol.psi_moderate, pol.psi_major)
        rows.append(
            {
                "feature": col,
                "test": test,
                "statistic": stat,
                "p_value": p,
                "effect": effect,
                "psi": psi,
                "psi_band": band,
                "significant": p < pol.p_value,
                "drifted": band == "major" or (band == "moderate" and p < pol.p_value),
            }
        )
    return rows


def prediction_drift(p_ref: np.ndarray, p_cur: np.ndarray, threshold: float) -> dict:
    ks_stat, ks_p = stats.ks(p_ref, p_cur)
    return {
        "psi": stats.psi_fixed(p_ref, p_cur),
        "ks": ks_stat,
        "ks_p_value": ks_p,
        "mean_reference": float(np.mean(p_ref)),
        "mean_current": float(np.mean(p_cur)),
        "decline_rate_reference": float(np.mean(p_ref >= threshold)),
        "decline_rate_current": float(np.mean(p_cur >= threshold)),
    }


def decide(
    dq: dict, fdrift: list[dict], pdrift: dict, perf: dict | None, labels: dict, pol
) -> dict:
    """Apply the monitoring policy; returns the action and every reason that fired."""
    reasons, action = [], "none"
    if dq["max_missing_rate"] > pol.max_missing_rate:
        reasons.append(
            f"missing rate {dq['max_missing_rate']:.1%} > {pol.max_missing_rate:.0%} on "
            + ", ".join(dq["columns_with_missing"])
        )
    if dq["unseen_categories"] and not pol.allow_unseen_categories:
        reasons.append(f"unseen category codes {dq['unseen_categories']}")
    if reasons:
        return {
            "action": "block",
            "reasons": reasons + ["fix the upstream feed; scores unreliable"],
        }
    share = float(np.mean([r["drifted"] for r in fdrift]))
    moved = [r["feature"] for r in fdrift if r["psi_band"] != "stable" or r["drifted"]]
    drift = []
    if share > pol.feature_drift_share:
        drift.append(f"feature drift share {share:.0%} > {pol.feature_drift_share:.0%}")
    if pdrift["psi"] >= pol.prediction_psi:
        drift.append(f"prediction PSI {pdrift['psi']:.3f} >= {pol.prediction_psi}")
    labelled = perf is not None and labels["n"] >= pol.min_labelled_rows
    degraded = []
    if labelled:
        # Prevalence-robust checks only: PR-AUC and cost per applicant move with the default
        # rate even when the model is fine, so they are reported but do not trigger.
        if perf["roc_auc_drop"] > pol.max_roc_auc_drop:
            degraded.append(f"ROC-AUC dropped {perf['roc_auc_drop']:.3f} > {pol.max_roc_auc_drop}")
        if abs(perf["calibration_gap"]) > pol.max_calibration_gap:
            degraded.append(
                f"calibration gap {perf['calibration_gap']:+.3f} (mean predicted - observed) "
                f"beyond +/-{pol.max_calibration_gap}"
            )
        if perf["cost_regret"] > pol.max_cost_regret:
            degraded.append(
                f"deployed cutoff costs {perf['cost_regret']:.0%} more than the batch-optimal "
                f"cutoff {perf['batch_optimal_threshold']:.3f} (limit {pol.max_cost_regret:.0%})"
            )
    if degraded:
        action, reasons = "retrain", degraded + drift
    elif drift and not labelled:
        # No outcomes yet: train a challenger on the latest labelled data as a precaution.
        # It still has to pass the promotion gate, so the champion is never replaced blindly.
        action, reasons = "retrain", drift + ["no labels yet: precautionary challenger"]
    elif drift or moved or labels["p_value"] < pol.p_value:
        action = "monitor"
        reasons = list(drift)
        if drift:
            reasons.append("labelled performance within limits: the model handles this shift")
        elif moved:
            reasons.append(f"moved below action limits: {', '.join(moved[:6])}")
        if labels["p_value"] < pol.p_value:
            reasons.append(f"default rate {labels['rate']:.1%} (p = {labels['p_value']:.1e})")
    return {"action": action, "reasons": reasons, "feature_drift_share": share}


def evaluate_batch(name, batch, reference, bundle, baseline: dict, cfg, p_ref=None) -> dict:
    """``reference`` = training rows (feature drift); ``p_ref`` = held-out scores (prediction
    drift: scores on rows the model trained on are over-confident, so they are not a fair
    reference)."""
    pol = cfg.monitor.policy
    ref_X = reference[FEATURE_COLUMNS]
    X = batch[FEATURE_COLUMNS]
    dq = data_quality(batch)
    fdrift = feature_drift(ref_X, X, pol)
    if p_ref is None:
        p_ref = bundle.predict_proba(ref_X)
    try:
        p_cur = bundle.predict_proba(X)
        scored = "as received"
    except ValueError:  # e.g. the linear model cannot score missing values
        p_cur = bundle.predict_proba(X.fillna(ref_X.median()))
        scored = "after median imputation (the model cannot score missing values)"
    pdrift = prediction_drift(p_ref, p_cur, bundle.threshold)
    y = batch[TARGET].to_numpy()
    yr = reference[TARGET].to_numpy()
    z, pz = stats.proportion_z(int(yr.sum()), len(yr), int(y.sum()), len(y))
    labels = {
        "n": int(len(y)),
        "rate": float(y.mean()),
        "reference_rate": float(yr.mean()),
        "z": z,
        "p_value": pz,
    }
    m = evaluate_probs(y, p_cur, bundle.threshold, cfg.cost.fn_cost, cfg.cost.fp_cost)
    policy = CostPolicy(cfg.cost.fn_cost, cfg.cost.fp_cost)
    best_thr, _ = choose_threshold(y, p_cur, policy)
    best_cost = evaluate_probs(y, p_cur, best_thr, policy.fn_cost, policy.fp_cost)["business_cost"]
    perf = {
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "brier": m["brier"],
        "ece": m["ece"],
        "business_cost": m["business_cost"],
        "recall": m["recall"],
        "roc_auc_drop": baseline["roc_auc"] - m["roc_auc"],
        "calibration_gap": float(p_cur.mean() - y.mean()),
        "batch_optimal_threshold": best_thr,
        "cost_regret": m["business_cost"] / best_cost - 1 if best_cost else 0.0,
    }
    decision = decide(dq, fdrift, pdrift, perf, labels, pol)
    return {
        "scenario": name,
        "description": DESCRIPTIONS[name],
        "n": int(len(batch)),
        "data_quality": dq,
        "feature_drift": fdrift,
        "n_drifted_features": int(sum(r["drifted"] for r in fdrift)),
        "n_significant_features": int(sum(r["significant"] for r in fdrift)),
        "prediction_drift": pdrift,
        "scored": scored,
        "labels": labels,
        "performance": perf,
        "decision": decision,
        "label_free_decision": decide(dq, fdrift, pdrift, None, labels, pol),
    }


def run_scenarios(cfg, df: pd.DataFrame, bundle, names=SCENARIOS) -> dict:
    splits = make_splits(df, cfg.train.splits, cfg.seed)
    reference = splits.train
    pool = splits.test.reset_index(drop=True)
    baseline = evaluate_probs(
        pool[TARGET],
        bundle.predict_proba(pool[FEATURE_COLUMNS]),
        bundle.threshold,
        cfg.cost.fn_cost,
        cfg.cost.fp_cost,
    )
    p_ref = bundle.predict_proba(splits.validation[FEATURE_COLUMNS])
    results = []
    for i, name in enumerate(names):
        batch = make_scenario(name, pool, cfg.monitor.slice_rows, cfg, cfg.monitor.slice_seed + i)
        r = evaluate_batch(name, batch, reference, bundle, baseline, cfg, p_ref)
        log.info("%s: %s (%s)", name, r["decision"]["action"], "; ".join(r["decision"]["reasons"]))
        results.append(r)
    return {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "model": {
            "family": bundle.family,
            "threshold": bundle.threshold,
            **{k: bundle.metadata.get(k) for k in ("git_commit", "generated_at")},
        },
        "reference": {
            "split": "train",
            "n": int(len(reference)),
            "prediction_reference": "validation split scores",
            "n_prediction_reference": int(len(p_ref)),
        },
        "baseline_test": {
            k: baseline[k] for k in ("roc_auc", "pr_auc", "brier", "business_cost", "recall")
        },
        "policy": dict(cfg.monitor.policy),
        "scenarios": results,
    }


def to_markdown(r: dict) -> str:
    pol = r["policy"]
    lines = [
        "# Drift scenarios and monitoring decisions",
        "",
        f"Generated {r['generated_at']} at commit `{r['git_commit'][:10]}`. Reference: training "
        f"split (n = {r['reference']['n']:,}); batches of {r['scenarios'][0]['n']:,} rows "
        "simulated from the test split. Model: "
        f"{r['model']['family']} (cutoff {r['model']['threshold']:.3f}).",
        "",
        f"Policy: block if any column misses > {pol['max_missing_rate']:.0%} or unseen codes "
        f"appear; retrain if > {pol['feature_drift_share']:.0%} of features drift "
        f"(PSI >= {pol['psi_major']}, or >= {pol['psi_moderate']} with p < {pol['p_value']}), "
        f"prediction PSI >= {pol['prediction_psi']}, or (with >= {pol['min_labelled_rows']} "
        f"labels) ROC-AUC drops > {pol['max_roc_auc_drop']}, the calibration gap exceeds "
        f"+/-{pol['max_calibration_gap']}, or the deployed cutoff costs > "
        f"{pol['max_cost_regret']:.0%} more than the batch-optimal one.",
        "",
        "| Scenario | Drifted / significant features | Prediction PSI | Decline rate | Default "
        "rate | ROC-AUC | Calibration gap | Cost regret | Action without labels | Action with "
        "labels |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in r["scenarios"]:
        pdr, perf = s["prediction_drift"], s["performance"]
        lines.append(
            f"| {s['scenario']} | {s['n_drifted_features']} / {s['n_significant_features']} | "
            f"{pdr['psi']:.3f} | {pdr['decline_rate_current']:.3f} | {s['labels']['rate']:.3f} | "
            f"{perf['roc_auc']:.3f} | {perf['calibration_gap']:+.3f} | {perf['cost_regret']:.1%} | "
            f"{s['label_free_decision']['action']} | **{s['decision']['action']}** |"
        )
    lines += ["", "## Why", ""]
    for s in r["scenarios"]:
        lines.append(
            f"- **{s['scenario']}** ({s['description']}): "
            + ("; ".join(s["decision"]["reasons"]) or "nothing moved")
        )
    lines += [
        "",
        "Limitations: batches are simulated, not production traffic; PSI depends on the "
        "binning; univariate tests miss joint changes (concept_drift has no feature drift at "
        "all); significance with thousands of rows says little about impact, which is why "
        "actions need effect sizes or labelled performance.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from credit_default.models.bundle import ModelBundle
    from credit_default.validation.schema import validate

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    opts = dict(f.lstrip("-").split("=", 1) for f in flags if "=" in f)
    cfg = load_config(overrides)
    df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    bundle = ModelBundle.load(opts.get("bundle", REPO_ROOT / cfg.paths.bundle_dir))
    out = Path(opts.get("out", REPO_ROOT / cfg.paths.results_dir / "monitoring"))
    if cfg.data.source != "uci" and "out" not in opts:
        out = REPO_ROOT / cfg.paths.artifacts_dir / "monitoring"
    result = run_scenarios(cfg, df, bundle)
    out.mkdir(parents=True, exist_ok=True)
    write_json(result, out / "scenarios.json")
    (out / "scenarios.md").write_text(to_markdown(result))
    log.info("wrote %s", out / "scenarios.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
