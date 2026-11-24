"""Promotion policy: the checks a challenger must pass before it can replace the champion.

Four groups, all configurable in ``configs/registry/default.yaml`` -> ``gate``:
  performance   absolute floors (ROC-AUC, PR-AUC) and PR-AUC no worse than the champion
                beyond a tolerance
  calibration   ECE ceiling and Brier no worse than the champion beyond a tolerance
  cost          expected cost per applicant no worse than the champion beyond a tolerance
  operational   API input contract unchanged, probabilities finite and in [0, 1],
                deterministic predictions, threshold in (0, 1), single-row p95 latency and
                bundle size under limits

Every check is evaluated (no short-circuit) so the decision record shows all failures. With no
champion, the comparative checks are skipped and the absolute ones still apply. A challenger
never replaces the champion unless every check passes.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from credit_default.data.schema import FEATURE_COLUMNS


@dataclass
class Check:
    name: str
    group: str
    passed: bool
    value: float | str | None
    limit: str
    skipped: bool = False

    def describe(self) -> str:
        v = f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        state = "skip" if self.skipped else "pass" if self.passed else "FAIL"
        return f"[{state}] {self.group}/{self.name}: {v} (limit {self.limit})"


@dataclass
class PolicyResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.skipped]

    def as_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.checks]


def operational_profile(bundle, X, n_latency: int = 200, bundle_path: Path | None = None) -> dict:
    """Measure what the operational checks need, on real rows."""
    p1 = bundle.predict_proba(X)
    p2 = bundle.predict_proba(X)
    rows = [X.iloc[[i % len(X)]] for i in range(n_latency)]
    bundle.predict_proba(rows[0])  # warm-up
    times = []
    for r in rows:
        t = time.perf_counter()
        bundle.predict_proba(r)
        times.append((time.perf_counter() - t) * 1000)
    size_mb = bundle_path.stat().st_size / 1e6 if bundle_path and bundle_path.exists() else None
    return {
        "finite_in_range": bool(np.all(np.isfinite(p1)) and np.all((p1 >= 0) & (p1 <= 1))),
        "deterministic": bool(np.array_equal(p1, p2)),
        "contract_ok": list(bundle.feature_columns) == list(FEATURE_COLUMNS),
        "threshold": float(bundle.threshold),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "bundle_mb": size_mb,
    }


def evaluate(challenger: dict, champion: dict | None, ops: dict | None, gate) -> PolicyResult:
    """Apply the gate config to test-split metrics (and the operational profile, if given)."""
    checks: list[Check] = []

    def add(name, group, passed, value, limit, skipped=False):
        checks.append(Check(name, group, bool(passed), value, limit, skipped))

    g = gate
    c = challenger
    add(
        "roc_auc_floor",
        "performance",
        c["roc_auc"] >= g.min_roc_auc,
        c["roc_auc"],
        f">= {g.min_roc_auc}",
    )
    add(
        "pr_auc_floor",
        "performance",
        c["pr_auc"] >= g.min_pr_auc,
        c["pr_auc"],
        f">= {g.min_pr_auc}",
    )
    if "ece" in c:
        add("ece_ceiling", "calibration", c["ece"] <= g.max_ece, c["ece"], f"<= {g.max_ece}")
    if champion is None:
        for name, group in (
            ("pr_auc_vs_champion", "performance"),
            ("brier_vs_champion", "calibration"),
            ("cost_vs_champion", "cost"),
        ):
            add(name, group, True, None, "no champion yet", skipped=True)
    else:
        p = champion
        floor = p["pr_auc"] - g.pr_auc_tolerance
        add(
            "pr_auc_vs_champion",
            "performance",
            c["pr_auc"] >= floor,
            c["pr_auc"],
            f">= champion {p['pr_auc']:.4f} - {g.pr_auc_tolerance}",
        )
        ceiling = p["brier"] + g.brier_tolerance
        add(
            "brier_vs_champion",
            "calibration",
            c["brier"] <= ceiling,
            c["brier"],
            f"<= champion {p['brier']:.4f} + {g.brier_tolerance}",
        )
        cost_ceiling = p["business_cost"] * (1 + g.cost_tolerance)
        add(
            "cost_vs_champion",
            "cost",
            c["business_cost"] <= cost_ceiling,
            c["business_cost"],
            f"<= champion {p['business_cost']:.4f} x (1 + {g.cost_tolerance})",
        )
    if ops is not None:
        add(
            "input_contract",
            "operational",
            ops["contract_ok"],
            str(ops["contract_ok"]),
            "API columns unchanged",
        )
        add(
            "probabilities_valid",
            "operational",
            ops["finite_in_range"],
            str(ops["finite_in_range"]),
            "finite, in [0, 1]",
        )
        add(
            "deterministic",
            "operational",
            ops["deterministic"],
            str(ops["deterministic"]),
            "same input -> same output",
        )
        add(
            "threshold_valid",
            "operational",
            0 < ops["threshold"] < 1,
            ops["threshold"],
            "in (0, 1)",
        )
        add(
            "p95_latency_ms",
            "operational",
            ops["p95_latency_ms"] <= g.max_p95_latency_ms,
            ops["p95_latency_ms"],
            f"<= {g.max_p95_latency_ms} ms (single row, in-process)",
        )
        if ops.get("bundle_mb") is not None:
            add(
                "bundle_size_mb",
                "operational",
                ops["bundle_mb"] <= g.max_bundle_mb,
                ops["bundle_mb"],
                f"<= {g.max_bundle_mb} MB",
            )
    passed = all(ch.passed or ch.skipped for ch in checks)
    return PolicyResult(passed, checks)
