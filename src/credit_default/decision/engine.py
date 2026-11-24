"""Cost-sensitive decision engine: probability of default -> approve / decline.

This is an educational decision-support component, not an approved lending engine. Costs are
relative units chosen as assumptions (``configs/cost/default.yaml``), not measured losses, so
the expected costs it reports are model-evaluation quantities, not financial savings.

Policy inputs:
  fn_cost            cost of approving a customer who then defaults (missed default)
  fp_cost            cost of declining a customer who would have repaid (lost margin)
  threshold          optional fixed cutoff; otherwise the cost-minimising cutoff is chosen
  min_approval_rate  optional business constraint (approve at least this share of applicants)

Decision rule: decline when P(default) >= threshold. With calibrated probabilities the
Bayes-optimal cutoff is fp_cost / (fp_cost + fn_cost); the engine still chooses the cutoff
empirically on validation data and reports both.

Usage (evaluate a policy with the committed model on the real test split):
  python -m credit_default.decision.engine --fn-cost=8 --fp-cost=1 [--min-approval=0.7]
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass

import numpy as np

from credit_default.evaluate.metrics import decision_metrics
from credit_default.threshold.cost import max_f1_threshold

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational decision-support analysis on a public 2005 dataset. Costs are stated "
    "assumptions in relative units; expected costs are not financial savings, and this is not "
    "an approved lending or credit-scoring system."
)


@dataclass(frozen=True)
class CostPolicy:
    fn_cost: float = 5.0
    fp_cost: float = 1.0
    threshold: float | None = None
    min_approval_rate: float | None = None

    def __post_init__(self):
        if self.fn_cost <= 0 or self.fp_cost <= 0:
            raise ValueError("fn_cost and fp_cost must be positive")
        if self.threshold is not None and not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.min_approval_rate is not None and not 0.0 <= self.min_approval_rate <= 1.0:
            raise ValueError("min_approval_rate must be in [0, 1]")

    @property
    def ratio(self) -> float:
        return self.fn_cost / self.fp_cost

    @property
    def bayes_threshold(self) -> float:
        return self.fp_cost / (self.fp_cost + self.fn_cost)

    @classmethod
    def from_config(cls, cost_cfg) -> CostPolicy:
        return cls(
            fn_cost=float(cost_cfg.fn_cost),
            fp_cost=float(cost_cfg.fp_cost),
            threshold=cost_cfg.get("threshold"),
            min_approval_rate=cost_cfg.get("min_approval_rate"),
        )


def threshold_grid(n: int = 199) -> np.ndarray:
    return np.linspace(0, 1, n + 2)[1:-1]


def decide(proba, threshold: float) -> np.ndarray:
    return np.where(np.asarray(proba) >= threshold, "decline", "approve")


def evaluate_threshold(y_true, proba, threshold: float, policy: CostPolicy) -> dict:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    m = decision_metrics(y_true, y_pred, policy.fn_cost, policy.fp_cost)
    m["threshold"] = float(threshold)
    m["total_cost"] = m["business_cost"] * len(y_pred)
    return m


def sweep(y_true, proba, policy: CostPolicy, grid: np.ndarray | None = None) -> dict:
    """Vectorised metrics for every threshold on the grid (used for curves and selection)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    t = threshold_grid() if grid is None else np.asarray(grid)
    pred = p[None, :] >= t[:, None]
    pos = y[None, :] == 1
    tp = (pred & pos).sum(axis=1)
    fp = (pred & ~pos).sum(axis=1)
    fn = (~pred & pos).sum(axis=1)
    tn = (~pred & ~pos).sum(axis=1)
    n = len(y)
    return {
        "threshold": t,
        "cost": (fn * policy.fn_cost + fp * policy.fp_cost) / n,
        "approval_rate": (tn + fn) / n,
        "recall": tp / np.maximum(tp + fn, 1),
        "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0),
        "fpr": fp / np.maximum(fp + tn, 1),
    }


def choose_threshold(y_true, proba, policy: CostPolicy, grid=None) -> tuple[float, str]:
    """The policy's cutoff: fixed if given, else the cost minimiser subject to the constraint."""
    if policy.threshold is not None:
        return float(policy.threshold), "fixed by policy"
    s = sweep(y_true, proba, policy, grid)
    cost = s["cost"].copy()
    reason = f"minimum expected cost at FN:FP = {policy.ratio:g}:1"
    if policy.min_approval_rate is not None:
        feasible = s["approval_rate"] >= policy.min_approval_rate
        if not feasible.any():
            raise ValueError(f"no threshold approves at least {policy.min_approval_rate:.0%}")
        cost[~feasible] = np.inf
        reason += f", approving at least {policy.min_approval_rate:.0%}"
    return float(s["threshold"][int(np.argmin(cost))]), reason


def compare_rules(y_val, p_val, y_test, p_test, policy: CostPolicy, deployed: float) -> list[dict]:
    """Threshold rules chosen on validation, evaluated on test."""
    opt, _ = choose_threshold(y_val, p_val, CostPolicy(policy.fn_cost, policy.fp_cost))
    s = sweep(y_val, p_val, policy)
    youden = float(s["threshold"][int(np.argmax(s["recall"] - s["fpr"]))])
    rules = [
        ("deployed (cost-optimal on validation)", deployed),
        ("cost-optimal on validation (re-derived)", opt),
        ("Bayes cutoff fp/(fp+fn) on calibrated scores", policy.bayes_threshold),
        ("0.5 (default classifier cutoff)", 0.5),
        ("max F1 on validation", max_f1_threshold(y_val, p_val)),
        ("Youden J on validation", youden),
    ]
    if policy.min_approval_rate is not None:
        constrained, _ = choose_threshold(y_val, p_val, policy)
        rules.append((f"cost-optimal with approval >= {policy.min_approval_rate:.0%}", constrained))
    return [
        {"rule": name, **evaluate_threshold(y_test, p_test, thr, policy)} for name, thr in rules
    ]


def sensitivity(
    y_val,
    p_val,
    y_test,
    p_test,
    ratios,
    deployed: float,
    fp_cost: float = 1.0,
) -> list[dict]:
    """How the optimal cutoff moves with the FN:FP cost ratio, and what a wrong assumption costs.

    For each ratio: the cutoff chosen on validation, the Bayes cutoff, and the test cost of
    (a) the re-optimised cutoff, (b) the deployed cutoff (tuned for the configured ratio),
    (c) 0.5. ``regret_deployed`` = (b) - (a): the price of keeping the deployed cutoff if the
    true ratio were this one.
    """
    rows = []
    for r in ratios:
        pol = CostPolicy(fn_cost=float(r) * fp_cost, fp_cost=fp_cost)
        thr, _ = choose_threshold(y_val, p_val, pol)
        best = evaluate_threshold(y_test, p_test, thr, pol)
        dep = evaluate_threshold(y_test, p_test, deployed, pol)
        half = evaluate_threshold(y_test, p_test, 0.5, pol)
        rows.append(
            {
                "ratio": float(r),
                "bayes_threshold": pol.bayes_threshold,
                "optimal_threshold": thr,
                "test_cost_optimal": best["business_cost"],
                "test_cost_deployed": dep["business_cost"],
                "test_cost_0_5": half["business_cost"],
                "regret_deployed": dep["business_cost"] - best["business_cost"],
                "recall_optimal": best["recall"],
                "approval_rate_optimal": best["approval_rate"],
            }
        )
    return rows


def analyse(y_val, p_val, y_test, p_test, policy: CostPolicy, deployed: float, ratios) -> dict:
    """Everything the decision-analysis page and report need, from real held-out predictions."""
    s = sweep(y_test, p_test, policy)
    step = max(1, len(s["threshold"]) // 99)
    curve = [{k: round(float(s[k][i]), 5) for k in s} for i in range(0, len(s["threshold"]), step)]
    return {
        "policy": {
            **asdict(policy),
            "ratio": policy.ratio,
            "bayes_threshold": policy.bayes_threshold,
        },
        "deployed_threshold": float(deployed),
        "at_deployed": evaluate_threshold(y_test, p_test, deployed, policy),
        "rules": compare_rules(y_val, p_val, y_test, p_test, policy, deployed),
        "sensitivity": sensitivity(y_val, p_val, y_test, p_test, ratios, deployed, policy.fp_cost),
        "test_curve": curve,
        "disclaimer": DISCLAIMER,
    }


def to_markdown(a: dict) -> str:
    pol = a["policy"]
    lines = [
        "# Decision analysis (cost-sensitive cutoff)",
        "",
        f"> {a['disclaimer']}",
        "",
        f"Policy: FN cost {pol['fn_cost']:g}, FP cost {pol['fp_cost']:g} (ratio "
        f"{pol['ratio']:g}:1), Bayes cutoff {pol['bayes_threshold']:.3f}; deployed cutoff "
        f"{a['deployed_threshold']:.3f}. All rows: test split, cutoffs chosen on validation.",
        "",
        "| Rule | Cutoff | Cost / applicant | Recall | Precision | Approval rate | FPR | FNR | "
        "TP | FP | TN | FN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in a["rules"]:
        lines.append(
            f"| {r['rule']} | {r['threshold']:.3f} | {r['business_cost']:.4f} | {r['recall']:.3f} "
            f"| {r['precision']:.3f} | {r['approval_rate']:.3f} | {r['fpr']:.3f} | {r['fnr']:.3f} "
            f"| {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )
    lines += [
        "",
        "## Sensitivity to the cost ratio",
        "",
        "| FN:FP | Bayes cutoff | Optimal cutoff (validation) | Test cost at optimum | "
        "Test cost at deployed cutoff | Regret of deployed cutoff | Test cost at 0.5 | "
        "Recall at optimum | Approval at optimum |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in a["sensitivity"]:
        lines.append(
            f"| {r['ratio']:g}:1 | {r['bayes_threshold']:.3f} | {r['optimal_threshold']:.3f} | "
            f"{r['test_cost_optimal']:.4f} | {r['test_cost_deployed']:.4f} | "
            f"{r['regret_deployed']:+.4f} | {r['test_cost_0_5']:.4f} | {r['recall_optimal']:.3f} | "
            f"{r['approval_rate_optimal']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Evaluate a policy with the committed bundle on the real validation/test splits."""
    import pandas as pd

    from credit_default.config import REPO_ROOT, load_config, split_cli
    from credit_default.data.split import make_splits
    from credit_default.models.bundle import ModelBundle
    from credit_default.validation.schema import validate

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    opts = dict(f.lstrip("-").split("=", 1) for f in flags if "=" in f)
    cfg = load_config(overrides)
    policy = CostPolicy(
        fn_cost=float(opts.get("fn-cost", cfg.cost.fn_cost)),
        fp_cost=float(opts.get("fp-cost", cfg.cost.fp_cost)),
        threshold=float(opts["threshold"]) if "threshold" in opts else None,
        min_approval_rate=float(opts["min-approval"]) if "min-approval" in opts else None,
    )
    bundle = ModelBundle.load(opts.get("bundle", REPO_ROOT / cfg.paths.bundle_dir))
    df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    splits = make_splits(df, cfg.train.splits, cfg.seed)
    (Xv, yv), (Xt, yt) = splits.xy(splits.validation), splits.xy(splits.test)
    pv, pt = bundle.predict_proba(Xv), bundle.predict_proba(Xt)
    thr, reason = choose_threshold(yv, pv, policy)
    m = evaluate_threshold(yt, pt, thr, policy)
    print(f"policy {policy}: cutoff {thr:.3f} ({reason})")
    print(
        f"test: cost/applicant {m['business_cost']:.4f}, recall {m['recall']:.3f}, precision "
        f"{m['precision']:.3f}, approval {m['approval_rate']:.3f}, FPR {m['fpr']:.3f}, "
        f"FNR {m['fnr']:.3f}"
    )
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
