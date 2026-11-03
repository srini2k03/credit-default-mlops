"""Rewrite the README results block from results/metrics.json (called by train.yml only).

The block between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced. Runs whose
metrics come from anything but the real UCI data are refused.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def render(m: dict, run_url: str | None = None) -> str:
    d = m["data"]
    if d.get("synthetic") or d.get("source") != "uci":
        raise SystemExit("refusing to write non-UCI metrics into the README")
    src = f"[train workflow run]({run_url})" if run_url else "train workflow"
    lines = [
        f"Real UCI data ({d['n_rows']:,} rows, default rate {d['default_rate']:.1%}), "
        f"test holdout n = {d['n_test']:,}. Generated {m['generated_at']} by the {src} "
        f"at commit `{m['git_commit'][:7]}`. Costs FN:FP = {m['cost']['fn_cost']:g}:{m['cost']['fp_cost']:g}.",
        "",
        "| Model | ROC-AUC | PR-AUC | F1 | Brier | Cost / applicant | Threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in m["families"]:
        t = f["test"]
        name = f"**{f['family']}** (champion)" if f["family"] == m["champion"] else f["family"]
        lines.append(
            f"| {name} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | {t['f1']:.3f} | {t['brier']:.3f} "
            f"| {t['business_cost']:.3f} | {t['threshold']:.2f} |"
        )
    lines += [
        "",
        "Champion at different decision thresholds (test holdout):",
        "",
        "| Threshold rule | Threshold | Cost / applicant | Recall | Approval rate | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in m["thresholds"]:
        lines.append(
            f"| {r['rule']} | {r['threshold']:.2f} | {r['business_cost']:.3f} | {r['recall']:.3f} "
            f"| {r['approval_rate']:.3f} | {r['f1']:.3f} |"
        )
    if m.get("ablation"):
        lines += [
            "",
            "Imbalance ablation (CV PR-AUC on the training split):",
            "",
            "| Model | none | class_weight | smote |",
            "|---|---|---|---|",
        ]
        by = {}
        for r in m["ablation"]:
            by.setdefault(r["family"], {})[r["strategy"]] = r["cv_pr_auc"]
        for fam, s in by.items():
            lines.append(
                f"| {fam} | "
                + " | ".join(
                    f"{s[k]:.3f}" if k in s else "-" for k in ("none", "class_weight", "smote")
                )
                + " |"
            )
    return "\n".join(lines)


def update(readme: Path, block: str) -> None:
    text = readme.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit("README has no results markers")
    readme.write_text(pattern.sub(lambda _: f"{START}\n{block}\n{END}", text))


def main() -> int:
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    run_url = None
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
    update(ROOT / "README.md", render(m, run_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
