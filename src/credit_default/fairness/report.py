"""Group-level fairness report on held-out predictions.

Groups come from attributes recorded in the data (SEX, EDUCATION, MARRIAGE, AGE bands); no
attribute is inferred. For each group: size, observed default rate, decline rate, precision,
recall (TPR), FPR, FNR and calibration (mean predicted vs observed default rate), with
stratified bootstrap 95% CIs. Disparities are summarised per attribute (ratio of lowest to
highest approval rate, FPR and FNR gaps, calibration-gap range).

Limits, stated in every report: metric parity is not legal compliance; the fairness criteria
conflict with each other when base rates differ; small groups have wide intervals; the data is
a 2005 Taiwanese sample and says nothing about other populations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.evaluate.metrics import decision_metrics

LABELS = {
    "SEX": {1: "male", 2: "female"},
    "EDUCATION": {
        1: "graduate school",
        2: "university",
        3: "high school",
        4: "others / unlabelled (0, 4, 5, 6)",
    },
    "MARRIAGE": {1: "married", 2: "single", 3: "others / unlabelled (0, 3)"},
}
AGE_BANDS = [
    (21, 29, "21-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 79, "60+"),
]
MIN_GROUP = 100  # below this the group is reported but flagged as too small to compare
LIMITATIONS = [
    "Groups use attributes recorded in the data; nothing is inferred.",
    "SEX, MARRIAGE and AGE are not model inputs, but other features can act as proxies, so "
    "group outcomes can still differ.",
    "Equal FPR/FNR (equalised odds), equal decline rates (demographic parity) and equal "
    "calibration cannot all hold when default rates differ between groups; the report shows "
    "all three instead of declaring one 'fair'.",
    "Passing any of these metrics is not evidence of legal or regulatory compliance.",
    "Groups under 100 applicants have wide intervals and are flagged.",
]


def group_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Grouping attributes with readable labels (undocumented codes folded as in training)."""
    out = pd.DataFrame(index=X.index)
    edu = X["EDUCATION"].where(X["EDUCATION"].between(1, 4), 4)
    mar = X["MARRIAGE"].where(X["MARRIAGE"].between(1, 3), 3)
    out["SEX"] = X["SEX"].map(LABELS["SEX"])
    out["EDUCATION"] = edu.map(LABELS["EDUCATION"])
    out["MARRIAGE"] = mar.map(LABELS["MARRIAGE"])
    age = X["AGE"].to_numpy()
    band = np.full(len(X), "other", dtype=object)
    for lo, hi, name in AGE_BANDS:
        band[(age >= lo) & (age <= hi)] = name
    out["AGE_BAND"] = band
    return out


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    d = decision_metrics(y, pred, 1.0, 1.0)
    return {
        "decline_rate": 1 - d["approval_rate"],
        "approval_rate": d["approval_rate"],
        "precision": d["precision"],
        "recall": d["recall"],
        "fpr": d["fpr"],
        "fnr": d["fnr"],
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
    }


CI_KEYS = ("decline_rate", "recall", "fpr", "fnr", "calibration_gap")


def _bootstrap(y, p, thr, n_boot, rng) -> dict:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos)) if len(pos) else pos, rng.choice(neg, len(neg))]
        )
        m = _metrics(y[idx], p[idx], thr)
        for k in CI_KEYS:
            draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()
    }


def fairness_report(
    X: pd.DataFrame, y, proba, threshold: float, n_boot: int = 300, seed: int = 0
) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    groups = group_columns(X.reset_index(drop=True))
    rng = np.random.default_rng(seed)
    overall = _metrics(y, p, threshold)
    attributes = {}
    for attr in groups.columns:
        rows = []
        for value in sorted(groups[attr].unique(), key=str):
            mask = (groups[attr] == value).to_numpy()
            yg, pg = y[mask], p[mask]
            row = {"group": str(value), "n": int(mask.sum()), "n_default": int(yg.sum())}
            row.update(_metrics(yg, pg, threshold))
            row["small"] = bool(mask.sum() < MIN_GROUP)
            row["ci95"] = _bootstrap(yg, pg, threshold, n_boot, rng) if yg.sum() > 1 else {}
            rows.append(row)
        comparable = [r for r in rows if not r["small"]] or rows
        approvals = [r["approval_rate"] for r in comparable]
        attributes[attr] = {
            "groups": rows,
            "summary": {
                "approval_rate_ratio_min_max": float(min(approvals) / max(approvals))
                if max(approvals) > 0
                else None,
                "fpr_gap": float(
                    max(r["fpr"] for r in comparable) - min(r["fpr"] for r in comparable)
                ),
                "fnr_gap": float(
                    max(r["fnr"] for r in comparable) - min(r["fnr"] for r in comparable)
                ),
                "calibration_gap_range": [
                    float(min(r["calibration_gap"] for r in comparable)),
                    float(max(r["calibration_gap"] for r in comparable)),
                ],
                "observed_rate_range": [
                    float(min(r["observed_rate"] for r in comparable)),
                    float(max(r["observed_rate"] for r in comparable)),
                ],
            },
        }
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "overall": overall,
        "attributes": attributes,
        "limitations": LIMITATIONS,
    }


def to_markdown(r: dict, title: str = "Fairness report (test split)") -> str:
    lines = [
        f"# {title}",
        "",
        f"Cutoff {r['threshold']:.3f}; n = {r['n']:,}. Positive = predicted default = decline. "
        "Brackets are stratified bootstrap 95% intervals.",
        "",
    ]
    for attr, block in r["attributes"].items():
        s = block["summary"]
        lines += [
            f"## {attr}",
            "",
            f"Approval-rate ratio (lowest / highest group): "
            f"{s['approval_rate_ratio_min_max']:.3f}; FPR gap {s['fpr_gap']:.3f}; FNR gap "
            f"{s['fnr_gap']:.3f}; observed default rate {s['observed_rate_range'][0]:.3f} to "
            f"{s['observed_rate_range'][1]:.3f}.",
            "",
            "| Group | n | Default rate | Decline rate | Precision | Recall | FPR | FNR | "
            "Mean predicted | Calibration gap |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in block["groups"]:
            ci = g["ci95"]

            def cell(key, g=g, ci=ci):
                v = f"{g[key]:.3f}"
                return f"{v} [{ci[key][0]:.3f}, {ci[key][1]:.3f}]" if key in ci else v

            flag = " (small)" if g["small"] else ""
            lines.append(
                f"| {g['group']}{flag} | {g['n']:,} | {g['observed_rate']:.3f} | "
                f"{cell('decline_rate')} | {g['precision']:.3f} | {cell('recall')} | "
                f"{cell('fpr')} | {cell('fnr')} | {g['mean_predicted']:.3f} | "
                f"{cell('calibration_gap')} |"
            )
        lines.append("")
    lines += ["## Limitations", "", *[f"- {t}" for t in r["limitations"]], ""]
    return "\n".join(lines)
