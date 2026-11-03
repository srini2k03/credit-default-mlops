"""Markdown run report built from metrics.json."""

from __future__ import annotations

from pathlib import Path


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_f(r.get(k, "")) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


METRIC_COLS = [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("f1", "F1"),
    ("brier", "Brier"),
    ("business_cost", "Cost / applicant"),
    ("threshold", "Threshold"),
    ("recall", "Recall"),
    ("approval_rate", "Approval rate"),
]


def write_markdown(m: dict, out_dir: Path) -> Path:
    d = m["data"]
    banner = (
        "> **SYNTHETIC FIXTURE RUN - these numbers are meaningless and must not be reported.**\n\n"
        if d.get("synthetic")
        else ""
    )
    fam_rows = [
        {
            "family": f["family"],
            "cv_pr_auc": f["cv_pr_auc"],
            **f["test"],
            "brier_uncal": f["test_brier_uncalibrated"],
            "n_trials": f["n_trials"],
        }
        for f in m["families"]
    ]
    lines = [
        "# Training run report",
        "",
        banner,
        f"- Data: `{d['source']}` ({d['n_rows']} rows, default rate {d['default_rate']:.3f}, "
        f"sha256 `{d['sha256'][:12]}...`)",
        f"- Generated: {m['generated_at']} from commit `{m['git_commit'][:10]}`",
        f"- Splits: train {d['n_train']} / calibration {d['n_calibration']} / validation {d['n_validation']} "
        f"/ test {d['n_test']}",
        f"- Costs: FN = {m['cost']['fn_cost']}, FP = {m['cost']['fp_cost']} "
        f"(Bayes threshold for calibrated scores = {m['cost']['bayes_threshold']:.3f})",
        f"- Imbalance handling: `{m['imbalance']}`; champion: **{m['champion']}**",
        "",
        "## Test holdout, all families (threshold = each model's cost-optimal threshold from validation)",
        "",
        table(
            fam_rows,
            [("family", "Model"), ("cv_pr_auc", "CV PR-AUC (train)")]
            + METRIC_COLS
            + [("brier_uncal", "Brier before calibration"), ("n_trials", "Optuna trials")],
        ),
        "",
        "## Threshold rules for the champion (test holdout)",
        "",
        table(m["thresholds"], [("rule", "Rule")] + METRIC_COLS),
        "",
    ]
    if m.get("ablation"):
        lines += [
            "## Imbalance ablation (CV PR-AUC on the training split, same hyper-parameters)",
            "",
            table(
                m["ablation"],
                [
                    ("family", "Model"),
                    ("strategy", "Strategy"),
                    ("cv_pr_auc", "CV PR-AUC"),
                    ("cv_pr_auc_std", "Std"),
                ],
            ),
            "",
        ]
    lines += [
        "## Figures",
        "",
        "![cost curve](figures/cost_curve.png)",
        "",
        "![calibration](figures/calibration.png)",
        "",
        "![SHAP summary](figures/shap_bar.png)",
        "",
        "![SHAP beeswarm](figures/shap_beeswarm.png)",
        "",
    ]
    path = out_dir / "metrics.md"
    path.write_text("\n".join(lines))
    return path
