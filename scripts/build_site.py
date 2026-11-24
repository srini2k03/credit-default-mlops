"""Build the static GitHub Pages site into site/.

Inputs (each optional): results/*.json, results/*.md and figures (committed by train.yml),
reports/drift_*.html (Evidently, built in the pages workflow), site/screenshots/*.png
(Playwright, pages workflow). Missing inputs are skipped or replaced by a short note.

The page uses the portfolio's design kit (web/sb-kit.css and web/sb-kit.js, copied unchanged
from https://srini2k03.github.io/) plus this project's web/site.css and web/site.js. Every
number on the page is read from the committed results at build time.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
WEB = ROOT / "web"
REPO = "https://github.com/srini2k03/credit-default-mlops"
PORTFOLIO = "https://srini2k03.github.io/"
API_DOCS = "https://credit-default-api-fkqg.onrender.com/docs"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Geist:wght@300..700"
    "&family=Geist+Mono:wght@400..600&family=Instrument+Serif:ital@0;1&display=swap"
)
SHOT_TITLES = {
    "mlflow-runs": ("Experiment runs", "experiments / credit-default"),
    "mlflow-registry": ("Registered model", "models / credit-default"),
}
SLICE_TEXT = {
    "control": "An untouched random sample of the test split. Expected result: no drift.",
    "drifted": (
        "Another test-split sample with deliberate shifts: credit limits ×0.7, ages −6 years, "
        "PAY_0 one month worse for 35% of rows, bill amounts ×1.25. Expected result: drift."
    ),
    "log": "The API's own prediction log, compared with the training split.",
}

esc = html.escape

# ---------------------------------------------------------------------------- icons
ARROW = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M7 17 17 7M8 7h9v9"/></svg>'
)
GITHUB = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37'
    "-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 "
    "1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29"
    "-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 "
    "1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 "
    '2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>'
)
THEME_ICONS = (
    '<svg class="icon-sun" width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
    'M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '<svg class="icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>'
)


# ---------------------------------------------------------------------------- inputs
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def inline_md(text: str) -> str:
    """Markdown for one table cell or bullet (code, bold, links), without the <p> wrapper."""
    out = markdown.markdown(text.strip())
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_tables(text: str) -> list[dict]:
    """Every pipe table in a markdown file, with the nearest heading above it."""
    lines, tables, heading, i = text.splitlines(), [], None, 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip())
        ):
            head, rows, i = split_row(line), [], i + 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            tables.append({"heading": heading, "head": head, "rows": rows})
            continue
        i += 1
    return tables


def md_bullets(text: str) -> list[str]:
    return [inline_md(ln[2:]) for ln in text.splitlines() if ln.startswith("- ")]


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def f4(x) -> str:
    """The 4-decimal format of results/metrics.md and results/gate.md."""
    return f"{x:.4f}" if isinstance(x, float) else ("-" if x is None else str(x))


def pct(x: float) -> str:
    return f"{x:.0%}"


def minus(s: str) -> str:
    return s.replace("-", "−")


# ---------------------------------------------------------------------------- components
NUM = re.compile(r"[−+-]?[\d.,]+\s?(%|ms|s|x)?|-|—")


def is_num(cell_html: str) -> bool:
    return bool(NUM.fullmatch(re.sub(r"<[^>]+>", "", cell_html).strip()))


def table(
    head: list[str], rows: list[list[str]], caption: str, best: int | None = None, cls: str = ""
) -> str:
    """Cells are HTML already. Numeric columns (judged on the body) are right-aligned."""
    numeric = []
    for j in range(len(head)):
        cells = [r[j] for r in rows if j < len(r) and r[j].strip()]
        numeric.append(bool(cells) and all(is_num(c) for c in cells))
    th = "".join(
        f'<th scope="col"{" class=num" if numeric[j] else ""}>{esc(h)}</th>'
        for j, h in enumerate(head)
    )
    body = []
    for i, r in enumerate(rows):
        tds = "".join(
            f"<td{' class=num' if j < len(numeric) and numeric[j] else ''}>{c}</td>"
            for j, c in enumerate(r)
        )
        body.append(f"<tr{' class=is-best' if i == best else ''}>{tds}</tr>")
    return (
        f'<div class="sb-table-wrap" tabindex="0" role="region" aria-label="{esc(caption)}">'
        f'<table class="sb-table {cls}"><caption class="sr-only">{esc(caption)}</caption>'
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def md_table(t: dict, caption: str, best_if=None) -> str:
    rows = [[inline_md(c) for c in r] for r in t["rows"]]
    best = next((i for i, r in enumerate(t["rows"]) if best_if and best_if(r)), None)
    return table(t["head"], rows, caption, best=best)


def section(sid: str, idx: str, label: str, title: str, lead: str, body: str) -> str:
    """A page section: kicker with index and rule, h2 with one serif accent word, lead, body."""
    lead_html = f'<p class="sb-lead">{lead}</p>' if lead else ""
    return (
        f'<section class="sb-section" id="{sid}" aria-labelledby="{sid}-h">'
        f'<div class="sb-container"><div class="section-head sb-reveal"><p class="sb-kicker">'
        f'<span class="sb-kicker__index">{idx}</span><span class="sb-kicker__rule"></span>'
        f'{esc(label)}</p><h2 class="sb-h2" id="{sid}-h">{title}</h2>{lead_html}</div>'
        f"{body}</div></section>"
    )


def card(title: str, sub: str, body: str, foot: str = "", cls: str = "") -> str:
    sub_html = f'<p class="sb-card__sub">{sub}</p>' if sub else ""
    foot_html = f'<p class="sb-card__foot">{foot}</p>' if foot else ""
    return (
        f'<article class="sb-card sb-spotlight sb-reveal {cls}">'
        f'<header class="card-head"><h3 class="sb-card__title">{title}</h3>{sub_html}</header>'
        f"{body}{foot_html}</article>"
    )


def figure(src: str, path: Path, title: str, sub: str, alt: str) -> str:
    size = png_size(path)
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
    return (
        f'<figure class="sb-card fig-card sb-reveal"><figcaption class="card-head">'
        f'<span class="sb-card__title">{title}</span><span class="sb-card__sub">{sub}</span>'
        f'</figcaption><a class="fig-plate" href="{src}" target="_blank" rel="noopener" '
        f'title="Open the full-size image"><img src="{src}" alt="{esc(alt)}" loading="lazy" '
        f'decoding="async"{dims}></a></figure>'
    )


def note(text: str, tone: str = "info") -> str:
    return f'<div class="sb-callout sb-callout--{tone} sb-reveal" role="note"><p>{text}</p></div>'


def bars(groups: list[tuple[str, list[tuple[float, str]], float]], legend: str, cap: str) -> str:
    """Horizontal comparison bars (the portfolio's credit visual)."""
    out = []
    for label, values, scale in groups:
        rows = "".join(
            f'<div class="bar-row"><div class="bar-track"><span class="bar bar--{k}" '
            f'style="--w:{min(v / scale, 1) * 100:.1f}%"></span></div>'
            f'<span class="bar-val">{txt}</span></div>'
            for k, (v, txt) in zip(("muted", "s1"), values, strict=False)
        )
        out.append(f'<div class="bar-group"><p class="bar-label">{label}</p>{rows}</div>')
    return (
        f'<figure class="viz sb-reveal"><div class="viz__body">{legend}{"".join(out)}</div>'
        f"<figcaption>{cap}</figcaption></figure>"
    )


# ---------------------------------------------------------------------------- sections
def hero(m: dict | None, gate: dict | None, load: dict | None) -> str:
    real = bool(m) and m["data"].get("source") == "uci"
    pills = ['<span class="sb-pill sb-pill--live">API live on Render</span>']
    if real:
        pills.append('<span class="sb-pill">Results committed · real UCI data</span>')
    if gate:
        tone = "pill-ok" if gate.get("passed") else "sb-pill--pending"
        word = "passed" if gate.get("passed") else "rejected"
        pills.append(f'<span class="sb-pill {tone}">Quality gate {word}</span>')
    lead = (
        "A credit-risk model on {n} customers whose approve/decline cutoff minimises expected "
        "cost. It sits behind a promotion-gated MLflow registry, a load-tested FastAPI service and "
        "policy-driven monitoring and retraining. These pages are rebuilt by CI from the committed results."
    ).format(n=f"{m['data']['n_rows']:,}" if real else "30,000")
    stats = ""
    if real:
        rules = {r["rule"]: r for r in m["thresholds"]}
        opt, base, t = rules.get("cost-optimal"), rules.get("0.5"), m["test"]
        tiles = [
            (
                f"{t['roc_auc']:.3f}",
                "ROC-AUC (holdout)",
                f"n = {t['n']:,} · PR-AUC {t['pr_auc']:.3f}",
            )
        ]
        if opt and base:
            change = opt["business_cost"] / base["business_cost"] - 1
            tiles += [
                (
                    minus(f"{change:+.0%}"),
                    "Expected cost / applicant",
                    f"{f4(base['business_cost'])} → {f4(opt['business_cost'])} vs the 0.5 cutoff",
                ),
                (
                    pct(opt["recall"]),
                    "Default recall",
                    f"from {pct(base['recall'])} at the 0.5 cutoff",
                ),
            ]
        if load and load.get("all"):
            a = load["all"]
            tiles.append(
                (
                    f"{a['p95']} ms",
                    "p95 latency under load",
                    f"Locust: {int(a['Requests']):,} requests, {a['Failures']} failures",
                )
            )
        stats = (
            '<div class="sb-stats hero-stats sb-reveal" role="list">'
            + "".join(
                f'<div class="sb-stat" role="listitem"><div class="sb-stat__value">{v}</div>'
                f'<div class="sb-stat__label">{lab}</div><div class="sb-stat__ctx">{ctx}</div></div>'
                for v, lab, ctx in tiles
            )
            + "</div>"
        )
    else:
        stats = note(
            "No real-data training run has been published yet. The train workflow fills this page "
            "after it trains on the UCI data and the quality gate passes."
        )
    return f"""
<section class="sb-hero" aria-labelledby="top-h">
  <div class="sb-container">
    <p class="sb-kicker"><span class="sb-kicker__index">MLOps · Credit risk</span>
      <span class="sb-kicker__rule"></span>2026 · Results &amp; monitoring</p>
    <h1 class="sb-h1" id="top-h">Credit default risk, decided on <span class="sb-serif">cost</span>.</h1>
    <p class="sb-lead">{lead}</p>
    <div class="sb-hero__meta">{"".join(pills)}</div>
    <div class="sb-hero__actions">
      <a class="sb-btn sb-btn--primary" href="predict.html">Score an applicant {ARROW}</a>
      <a class="sb-btn" href="{API_DOCS}" target="_blank" rel="noopener">Live API docs</a>
      <a class="sb-btn sb-btn--ghost" href="{REPO}" target="_blank" rel="noopener">{GITHUB} Code</a>
    </div>
    <p class="hero-note">The API runs on Render's free tier: the first request wakes the service in 30–60 s.</p>
    {stats}
  </div>
</section>"""


def decision_section(m: dict, figures: Path) -> str:
    rules = {r["rule"]: r for r in m["thresholds"]}
    opt, base = rules.get("cost-optimal"), rules.get("0.5")
    c = m["cost"]
    text = (
        f"<p>Approving a customer who defaults costs the lender far more than declining one who "
        f"would have repaid. The pipeline prices a miss at <strong>{c['fn_cost']:g}</strong> and a "
        f"false alarm at <strong>{c['fp_cost']:g}</strong>, then picks the cutoff that minimises "
        f"expected cost per applicant on the validation split.</p>"
        f"<p>With calibrated scores the Bayes-optimal cutoff is "
        f"<span class=mono>{c['bayes_threshold']:.3f}</span>; the empirical optimum was "
        f"<span class=mono>{opt['threshold']:.2f}</span>. Both are reported, and the decision is "
        f"scored once on the untouched test holdout.</p>"
        if opt
        else ""
    )
    viz = ""
    if opt and base:
        legend = (
            '<div class="legend"><span><i class="sw sw--muted"></i>0.5 default</span>'
            f'<span><i class="sw sw--s1"></i>cost-optimal (t = {opt["threshold"]:.2f})</span></div>'
        )
        top = max(base["business_cost"], opt["business_cost"]) * 1.08
        viz = bars(
            [
                (
                    "Expected cost / applicant",
                    [
                        (base["business_cost"], f4(base["business_cost"])),
                        (opt["business_cost"], f4(opt["business_cost"])),
                    ],
                    top,
                ),
                (
                    "Default recall",
                    [(base["recall"], pct(base["recall"])), (opt["recall"], pct(opt["recall"]))],
                    1.0,
                ),
            ],
            legend,
            f"Test holdout, n = {opt['n']:,} (results/metrics.json). Lower cost is better; costs "
            "are in units of one false alarm.",
        )
    cols = ["threshold", "business_cost", "recall", "approval_rate", "f1"]
    heads = ["Rule", "Threshold", "Cost / applicant", "Recall", "Approval rate", "F1"]
    rows = [[esc(r["rule"])] + [f4(r[k]) for k in cols] for r in m["thresholds"]]
    best = next((i for i, r in enumerate(m["thresholds"]) if r["rule"] == "cost-optimal"), None)
    t = m["test"]
    rules_card = card(
        "Threshold rules for the champion",
        "Test holdout. Each rule moves the cutoff on the same calibrated scores.",
        table(heads, rows, "Threshold rules for the champion model", best=best),
        f"ROC-AUC {f4(t['roc_auc'])}, PR-AUC {f4(t['pr_auc'])} and Brier {f4(t['brier'])} do not "
        "depend on the cutoff. Source: results/metrics.md",
    )
    fig = ""
    if (figures / "cost_curve.png").exists():
        fig = figure(
            "results/figures/cost_curve.png",
            figures / "cost_curve.png",
            "Cost vs threshold",
            "Champion on the validation split, where the cutoff is chosen.",
            "Expected cost per applicant against the decision threshold, with the cost-optimal, "
            "0.5 and max-F1 cutoffs marked",
        )
    return section(
        "decision",
        "01",
        "Decision",
        'The cutoff is a <span class="sb-serif">cost</span> decision.',
        "",
        f'<div class="split"><div class="prose sb-reveal">{text}</div>{viz}</div>'
        f'<div class="stack">{rules_card}{fig}</div>',
    )


def models_section(m: dict, figures: Path) -> str:
    d, champ = m["data"], m["champion"]
    cols = [
        ("cv_pr_auc", "CV PR-AUC (train)"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1"),
        ("brier", "Brier"),
        ("brier_uncal", "Brier before calibration"),
        ("business_cost", "Cost / applicant"),
        ("threshold", "Threshold"),
        ("recall", "Recall"),
        ("approval_rate", "Approval rate"),
        ("n_trials", "Optuna trials"),
    ]
    rows, best = [], None
    for i, f in enumerate(m["families"]):
        v = {
            "cv_pr_auc": f["cv_pr_auc"],
            **f["test"],
            "brier_uncal": f.get("test_brier_uncalibrated"),
            "n_trials": f.get("n_trials"),
        }
        name = esc(f["family"])
        if f["family"] == champ:
            best = i
            name += ' <span class="sb-tag tag-champ">champion</span>'
        rows.append([name] + [f4(v.get(k)) for k, _ in cols])
    fam_card = card(
        "Test holdout, all model families",
        "Each family at its own cost-optimal threshold from validation. Optuna TPE tuned CV PR-AUC "
        "on the training split; isotonic calibration on a separate split.",
        table(["Model"] + [lab for _, lab in cols], rows, "Model comparison", best=best),
        f"Champion = lowest expected cost on validation. Imbalance handling: {esc(m.get('imbalance', ''))}. "
        "Source: results/metrics.md",
    )
    abl = ""
    if m.get("ablation"):
        strategies = list(dict.fromkeys(a["strategy"] for a in m["ablation"]))
        fams = list(dict.fromkeys(a["family"] for a in m["ablation"]))
        look = {(a["family"], a["strategy"]): a for a in m["ablation"]}
        arows = []
        for fam in fams:
            cells = [esc(fam)]
            vals = [look.get((fam, s)) for s in strategies]
            top = max((a["cv_pr_auc"] for a in vals if a), default=None)
            for a in vals:
                if not a:
                    cells.append("-")
                    continue
                txt = f'{f4(a["cv_pr_auc"])} <span class="sd">± {f4(a["cv_pr_auc_std"])}</span>'
                cells.append(f"<strong>{txt}</strong>" if a["cv_pr_auc"] == top else txt)
            arows.append(cells)
        abl = card(
            "Imbalance ablation",
            "CV PR-AUC ± std on the training split, same hyper-parameters. SMOTE runs inside the "
            "folds, so oversampling never touches validation data. Bold = best per model.",
            table(["Model"] + strategies, arows, "Imbalance ablation"),
            "Source: results/metrics.md",
        )
    figs = [
        (
            "calibration.png",
            "Reliability",
            "Calibrated probabilities on the test holdout.",
            "Reliability diagram: observed default rate against mean predicted probability",
        ),
        (
            "shap_bar.png",
            "SHAP importance",
            "Mean absolute SHAP value per feature.",
            "Bar chart of mean absolute SHAP values per feature",
        ),
        (
            "shap_beeswarm.png",
            "SHAP beeswarm",
            "Direction and spread of each feature's effect.",
            "SHAP beeswarm plot of feature contributions",
        ),
    ]
    fig_html = "".join(
        figure(f"results/figures/{n}", figures / n, t, s, a)
        for n, t, s, a in figs
        if (figures / n).exists()
    )
    meta = (
        f'<dl class="kv sb-reveal">'
        f"<div><dt>Data</dt><dd>UCI Default of Credit Card Clients · {d['n_rows']:,} rows · "
        f"default rate {d['default_rate']:.1%}</dd></div>"
        f"<div><dt>Splits</dt><dd>train {d['n_train']:,} / calibration {d['n_calibration']:,} / "
        f"validation {d['n_validation']:,} / test {d['n_test']:,}</dd></div>"
        f"<div><dt>Run</dt><dd>{esc(m['generated_at'])} · commit "
        f'<a href="{REPO}/commit/{esc(m["git_commit"])}"><code>{esc(m["git_commit"][:7])}</code></a></dd></div>'
        f"</dl>"
    )
    return section(
        "models",
        "02",
        "Models",
        'Three families, one <span class="sb-serif">champion</span>.',
        "Logistic regression, LightGBM and XGBoost, tuned with Optuna and calibrated with "
        "isotonic regression. The champion is picked on validation cost, then scored once on "
        "the holdout.",
        meta
        + f'<div class="stack">{fam_card}{abl}</div>'
        + (f'<div class="fig-grid">{fig_html}</div>' if fig_html else ""),
    )


def gate_section(gate: dict | None, gate_md: Path) -> str:
    if gate:
        ok = gate.get("passed")
        verdict = (
            f'<p class="verdict-word {"is-pass" if ok else "is-fail"}">'
            f"{'PASS' if ok else 'REJECT'}</p>"
        )
        reasons = "".join(f"<li>{esc(r)}</li>" for r in gate.get("reasons", []))
        left = card(
            "Decision",
            "Challenger vs production, re-scored on the same test holdout.",
            verdict
            + f'<ul class="reasons">{reasons}</ul>'
            + '<dl class="kv kv--tight">'
            + f"<div><dt>Challenger</dt><dd><code>{esc(gate.get('challenger_ref', ''))}</code></dd></div>"
            + f"<div><dt>Production</dt><dd><code>{esc(gate.get('production_ref', ''))}</code></dd></div>"
            + "</dl>",
            "Exit 1 if challenger PR-AUC &lt; production − 0.005 or cost &gt; production × 1.01. "
            "Source: results/gate.json",
            cls="gate-card",
        )
        keys = ["pr_auc", "roc_auc", "business_cost", "brier", "f1", "threshold"]
        prod = gate.get("production") or {}
        rows = [[f"<code>{k}</code>", f4(gate["challenger"].get(k)), f4(prod.get(k))] for k in keys]
        right = card(
            "Metrics",
            "Each model at its own threshold.",
            table(["Metric", "Challenger", "Production"], rows, "Quality gate metrics"),
            "A unit test trains a model on permuted labels and asserts that this gate rejects it.",
        )
        body = f'<div class="sb-grid-2">{left}{right}</div>'
    elif gate_md.exists():
        tables = md_tables(gate_md.read_text())
        body = card(
            "Quality gate",
            "",
            "".join(md_table(t, "Quality gate") for t in tables),
            "Source: results/gate.md",
        )
    else:
        return ""
    return section(
        "gate",
        "03",
        "Quality gate",
        'Nothing ships without the <span class="sb-serif">gate</span>.',
        "Every training run registers a challenger as <code>@staging</code>. The gate compares "
        "it with <code>@production</code>; only a pass moves the production alias.",
        body,
    )


def champion_run(t: dict):
    """Row predicate for the champion family's run in the MLflow runs table."""
    if "champion" not in t["head"]:
        return None
    j = t["head"].index("champion")
    champ = next((r[j] for r in t["rows"] if j < len(r) and r[j]), None)
    return (lambda r: r[0] == champ) if champ else None


def registry_section(summary: Path, shots: list[Path]) -> str:
    if not summary.exists() and not shots:
        return ""
    cards = []
    if summary.exists():
        for t in md_tables(summary.read_text()):
            if t["heading"] and "version" in t["heading"].lower():
                cards.append(
                    card(
                        "Registered versions",
                        "Aliases replace the deprecated registry stages: <code>@staging</code> is "
                        "the challenger, <code>@production</code> is what the API serves.",
                        md_table(t, "Registered model versions"),
                        "Source: results/mlflow_summary.md",
                    )
                )
            else:
                cards.append(
                    card(
                        "Experiment runs",
                        "Experiment <code>credit-default</code>: one parent run and one nested run "
                        "per model family, exported after the training run.",
                        md_table(t, "MLflow experiment runs", best_if=champion_run(t)),
                        "Source: results/mlflow_summary.md",
                    )
                )
    if shots:
        frames = []
        for s in shots:
            title, url = SHOT_TITLES.get(s.stem, (s.stem.replace("-", " ").title(), s.stem))
            size = png_size(s)
            dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
            frames.append(
                f'<figure class="sb-card shot-card sb-reveal"><figcaption class="card-head">'
                f'<span class="sb-card__title">MLflow UI · {esc(title)}</span>'
                f'<span class="sb-card__sub">Captured with Playwright in the pages workflow.</span>'
                f'</figcaption><div class="browser"><div class="browser__bar" aria-hidden="true">'
                f'<span class="dots"><i></i><i></i><i></i></span><span class="browser__url">'
                f"mlflow · {esc(url)}</span></div>"
                f'<a href="screenshots/{esc(s.name)}" class="browser__body" target="_blank" rel="noopener">'
                f'<img src="screenshots/{esc(s.name)}" alt="Screenshot of the MLflow UI: {esc(title.lower())}" '
                f'loading="lazy" decoding="async"{dims}></a></div></figure>'
            )
        cards.append(f'<div class="shot-grid">{"".join(frames)}</div>')
    else:
        cards.append(
            note(
                "MLflow UI screenshots are captured in the pages workflow from the MLflow store of "
                "the latest successful train run. This build had no store to screenshot."
            )
        )
    return section(
        "registry",
        "04",
        "Registry",
        'Tracked and versioned in <span class="sb-serif">MLflow</span>.',
        "Params, metrics, figures, config and the deployable model bundle are logged per run. "
        "The API resolves <code>models:/credit-default@production</code>, so promotion and "
        "rollback are alias moves.",
        f'<div class="stack">{"".join(cards)}</div>',
    )


def parse_loadtest(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    tables = md_tables(text)
    if not tables:
        return None
    t = tables[0]
    out = {"table": t, "bullets": md_bullets(text), "all": None}
    for r in t["rows"]:
        if "All" in r[0]:
            out["all"] = dict(zip(t["head"], r, strict=False))
    return out


def load_section(load: dict | None) -> str:
    if not load:
        return ""
    t = load["table"]
    head = t["head"]
    rows = [r for r in t["rows"] if "All" not in r[0]]
    try:
        i50, i95, i99 = head.index("p50"), head.index("p95"), head.index("p99")
        top = max(float(r[i99]) for r in t["rows"]) * 1.05
        lat = "".join(
            f'<div class="lat-row"><span class="lat-name">{inline_md(r[0])}</span>'
            f'<div class="lat-track" aria-hidden="true">'
            f'<span class="lat lat--p99" style="--w:{float(r[i99]) / top * 100:.1f}%"></span>'
            f'<span class="lat lat--p95" style="--w:{float(r[i95]) / top * 100:.1f}%"></span>'
            f'<span class="lat lat--p50" style="--w:{float(r[i50]) / top * 100:.1f}%"></span></div>'
            f'<span class="lat-val">{esc(r[i50])} / <strong>{esc(r[i95])}</strong> / {esc(r[i99])} ms</span></div>'
            for r in rows
        )
        legend = (
            '<div class="legend"><span><i class="sw sw--p50"></i>p50</span>'
            '<span><i class="sw sw--s1"></i>p95</span><span><i class="sw sw--muted"></i>p99</span></div>'
        )
        viz = (
            f'<figure class="viz sb-reveal"><div class="viz__body">{legend}{lat}</div>'
            "<figcaption>Client-side latency per endpoint in milliseconds (results/loadtest.md).</figcaption></figure>"
        )
    except (ValueError, IndexError):
        viz = ""
    meta = "".join(f"<li>{b}</li>" for b in load["bullets"])
    tbl = card(
        "Locust results",
        "Every endpoint, with the aggregate row.",
        md_table(t, "Load test results", best_if=lambda r: "All" in r[0]),
        "Source: results/loadtest.md, written by the train workflow",
    )
    return section(
        "load",
        "05",
        "Serving",
        'FastAPI, measured under <span class="sb-serif">load</span>.',
        "<code>/predict</code>, <code>/predict_batch</code>, <code>/explain</code> (per-request "
        "SHAP) and <code>/health</code>, load-tested with Locust in CI against the Docker image.",
        f'<div class="split"><ul class="meta-list sb-reveal">{meta}</ul>{viz}</div>'
        f'<div class="stack">{tbl}</div>',
    )


def drifted_columns(info: dict) -> list[str]:
    """Columns over their drift threshold, by score. Distance tests drift when score >= threshold,
    p-value tests when p < threshold. Empty unless the count matches Evidently's own count."""
    cols = []
    for c, v in info.get("columns", {}).items():
        method = str(v.get("method") or "").lower()
        if v.get("score") is None or v.get("threshold") is None:
            return []
        pvalue = "p_value" in method or "p-value" in method
        if (v["score"] < v["threshold"]) if pvalue else (v["score"] >= v["threshold"]):
            cols.append(c)
    if len(cols) != info.get("drifted_columns"):
        return []
    return sorted(cols, key=lambda c: -info["columns"][c]["score"])


def drift_section(reports: Path) -> str:
    drift = sorted(
        reports.glob("drift_*.html") if reports.exists() else [],
        key=lambda f: ({"control": 0, "drifted": 1}.get(f.stem[6:], 2), f.name),
    )
    intro = (
        "The human-readable view of the same idea: Evidently's data-drift reports for the "
        "<em>control</em> slice (untouched test rows) and the <em>drifted</em> slice "
        "(covariate shift), against the training split. The share of drifted columns below uses "
        "Evidently's own tests; the retrain decision itself comes from the policy above."
    )
    head = ("drift", "06", "Monitoring", 'Watching for <span class="sb-serif">drift</span>.', intro)
    if not drift:
        return section(*head, note("No drift reports were built in this run."))
    (SITE / "reports").mkdir(exist_ok=True)
    summary = load_json(reports / "drift_summary.json") or {}
    verdicts, tabs = [], []
    for f in drift:
        shutil.copy2(f, SITE / "reports" / f.name)
        name = f.stem[len("drift_") :]
        info = load_json(f.with_suffix(".json")) or {}
        mb = f.stat().st_size / 1e6
        body = f'<p class="slice-text">{esc(SLICE_TEXT.get(name, ""))}</p>'
        pill = ""
        if info:
            drifted = info["drift_detected"]
            pill = (
                '<span class="sb-pill sb-pill--pending">drift detected</span>'
                if drifted
                else '<span class="sb-pill pill-ok">no drift</span>'
            )
            share, thr = info["drift_share"], info["threshold"]
            cols = drifted_columns(info)
            col_tags = (
                '<p class="sb-label">Drifted columns</p><p class="tags">'
                + "".join(f'<span class="sb-tag">{esc(c)}</span>' for c in cols)
                + "</p>"
                if cols
                else ""
            )
            body += (
                f'<p class="big-num"><span>{info["drifted_columns"]}</span>'
                f'<span class="of">/ {info["n_columns"]} columns drifted</span></p>'
                f'<div class="share" role="img" aria-label="Drifted-column share {share:.2f} '
                f'against a threshold of {thr:.2f}"><div class="sb-meter"><span class="{"is-hot" if drifted else ""}" '
                f'style="width:{min(share, 1) * 100:.1f}%"></span></div>'
                f'<span class="share__thr" style="left:{thr * 100:.1f}%"></span></div>'
                f'<p class="share-legend"><span>share {share:.2f}</span><span>threshold {thr:.2f}</span></p>'
                f'<p class="sb-small">{info.get("n_current", 0):,} rows vs {info.get("n_reference", 0):,} '
                f"reference rows</p>{col_tags}"
            )
        body += (
            f'<p class="card-actions"><a class="sb-btn sb-btn--sm" href="reports/{f.name}" '
            f'target="_blank" rel="noopener">Open full report {ARROW}</a>'
            f'<span class="sb-small">{mb:.1f} MB · Evidently HTML</span></p>'
        )
        verdicts.append(
            f'<article class="sb-card sb-spotlight sb-reveal verdict">'
            f'<header class="verdict__top"><h3 class="sb-card__title">{esc(name.title())} slice</h3>'
            f"{pill}</header>{body}</article>"
        )
        tabs.append((name, f.name))
    warn = ""
    if summary.get("data_source") != "uci":
        warn = note(
            "<strong>Built on the synthetic test fixture, not the real data.</strong>", tone="warn"
        )
    first = next((t for t in tabs if t[0] == "drifted"), tabs[0])
    buttons = "".join(
        f'<button type="button" class="sb-tab" role="tab" id="tab-{esc(n)}" '
        f'aria-selected="{"true" if (n, fn) == first else "false"}" aria-controls="drift-panel" '
        f'data-src="reports/{esc(fn)}" data-label="{esc(n)}">{esc(n.title())}</button>'
        for n, fn in tabs
    )
    viewer = f"""
<div class="sb-card viewer sb-reveal">
  <div class="viewer__head">
    <div>
      <h3 class="sb-card__title">Evidently report, embedded</h3>
      <p class="sb-card__sub">Third-party HTML from Evidently, shown as-is in a frame that scrolls on its own. Its header applies Evidently's default dataset-drift share of 0.5; the retrain check and the cards above use this project's threshold (<code>drift_share_threshold</code> in <code>configs/monitor/default.yaml</code>).</p>
    </div>
    <div class="sb-tabs js-only" role="tablist" aria-label="Choose a slice">{buttons}</div>
  </div>
  <div class="browser" id="drift-panel" role="tabpanel" aria-labelledby="tab-{esc(first[0])}">
    <div class="browser__bar"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <a class="browser__url" id="drift-open" href="reports/{esc(first[1])}" target="_blank" rel="noopener">reports/{esc(first[1])} {ARROW}</a></div>
    <iframe id="drift-frame" src="reports/{esc(first[1])}" title="Evidently data-drift report: {esc(first[0])} slice" loading="lazy"></iframe>
  </div>
</div>"""
    return section(
        *head, warn + f'<div class="sb-grid-2 verdicts">{"".join(verdicts)}</div>' + viewer
    )


# ---------------------------------------------------------------------------- dashboard helpers
PAGES = [
    ("index.html", "Overview"),
    ("predict.html", "Predict"),
    ("evaluation.html", "Evaluation"),
    ("monitoring.html", "Monitoring"),
    ("registry.html", "Registry"),
]
API_BASE = API_DOCS.removesuffix("/docs")
ACTION_PILL = {
    "none": "pill-none",
    "monitor": "pill-monitor",
    "retrain": "pill-retrain",
    "block": "pill-block",
    "promote": "pill-promote",
    "reject": "pill-reject",
    "rollback": "pill-rollback",
}


def pill(word: str, cls: str | None = None) -> str:
    return f'<span class="sb-pill {cls or ACTION_PILL.get(word, "")}">{esc(word)}</span>'


def embed(sid: str, obj) -> str:
    """Inline JSON for the page's scripts (escaped so it cannot close the script element)."""
    text = json.dumps(obj, separators=(",", ":"), default=str).replace("</", "<\\/")
    return f'<script type="application/json" id="{sid}">{text}</script>'


def provenance(obj: dict | None, source: str) -> str:
    if not obj:
        return f"Source: {source}"
    commit = str(obj.get("git_commit", ""))[:7]
    when = obj.get("generated_at", "")
    link = (
        f'<a href="{REPO}/commit/{esc(obj.get("git_commit", ""))}"><code>{esc(commit)}</code></a>'
    )
    return f"Source: {source} · generated {esc(when)} at {link}" if commit else f"Source: {source}"


def page_hero(kicker: str, title: str, lead: str, extra: str = "") -> str:
    return f"""
<section class="sb-hero page-hero" aria-labelledby="top-h">
  <div class="sb-container">
    <p class="sb-kicker"><span class="sb-kicker__index">Credit risk · MLOps</span>
      <span class="sb-kicker__rule"></span>{esc(kicker)}</p>
    <h1 class="sb-h1" id="top-h">{title}</h1>
    <p class="sb-lead">{lead}</p>
    {extra}
  </div>
</section>"""


def chart_card(
    title: str,
    sub: str,
    host_id: str,
    table_html: str,
    foot: str,
    controls: str = "",
    legend: str = "",
) -> str:
    return card(
        title,
        sub,
        f'{controls}{legend}<div class="chart-host" id="{host_id}"></div>'
        f'<details class="data-table"><summary>Show the data table</summary>{table_html}</details>',
        foot,
        cls="chart-card",
    )


def legend(items: list[tuple[str, str]]) -> str:
    out = []
    for cls, label in items:
        mark = '<i class="dash"></i>' if cls == "dash" else f'<i class="sw {cls}"></i>'
        out.append(f"<span>{mark}{esc(label)}</span>")
    return f'<div class="chart-legend" aria-hidden="true">{"".join(out)}</div>'


# ---------------------------------------------------------------------------- overview
def overview_status(m, gate, history, scenarios, dq, bench, load) -> str:
    tiles = []
    if m:
        t = m["test"]
        tiles.append(
            (
                "Champion",
                f"{esc(m['champion'])}",
                f"version <code>local:{esc(m['git_commit'][:10])}</code> · "
                f"{esc(m.get('calibration', 'isotonic'))} calibration · cutoff "
                f"{t['threshold']:.3f} · ROC-AUC {t['roc_auc']:.3f} · PR-AUC {t['pr_auc']:.3f}",
                '<a href="evaluation.html">Evaluation →</a>',
            )
        )
    if history:
        last = history[-1]
        when = str(last.get("ts", ""))[:10]
        tiles.append(
            (
                "Latest governance event",
                pill(last["event"]),
                f"{esc(when)} · {esc('; '.join(last.get('reasons', []))[:140])}",
                '<a href="registry.html">Registry →</a>',
            )
        )
    elif gate:
        tiles.append(("Latest gate", pill("promote" if gate["passed"] else "reject"), "", ""))
    if scenarios:
        counts: dict[str, int] = {}
        for s in scenarios["scenarios"]:
            counts[s["decision"]["action"]] = counts.get(s["decision"]["action"], 0) + 1
        control = next((s for s in scenarios["scenarios"] if s["scenario"] == "control"), None)
        ctx = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        tiles.append(
            (
                "Monitoring (simulated batches)",
                f"{len(scenarios['scenarios'])} batches",
                f"Control batch: {esc(control['decision']['action'] if control else '-')}. "
                f"Decisions: {esc(ctx)}.",
                '<a href="monitoring.html">Monitoring →</a>',
            )
        )
    if dq:
        tiles.append(
            (
                "Data contract",
                pill("pass", "pill-none") if dq["schema"]["passed"] else pill("fail", "pill-block"),
                f"{dq['n_rows']:,} rows · {sum(c['n_missing'] for c in dq['columns'])} missing values · "
                f"{dq['duplicates']['duplicate_ids']} duplicate IDs",
                '<a href="monitoring.html#quality">Data quality →</a>',
            )
        )
    if bench:
        fast = (
            bench["in_process"]["single"].get("fast") or bench["in_process"]["single"]["pipeline"]
        )
        http = next((h for h in bench["http"] if h["fast_scoring"]), bench["http"][0])
        c1 = http["predict"][0]
        tiles.append(
            (
                "Serving (benchmark)",
                f"{c1['p95_ms']:.1f} ms",
                f"p95 single /predict over HTTP, 1 client ({esc(bench['environment']['machine'])}); "
                f"in-process scoring p50 {fast['p50_ms']:.2f} ms",
                '<a href="#serving">Serving →</a>',
            )
        )
    elif load and load.get("all"):
        a = load["all"]
        tiles.append(("Serving (load test)", f"{a['p95']} ms", "p95 under Locust load", ""))
    body = "".join(
        f'<article class="sb-card sb-spotlight sb-reveal status"><p class="sb-label">{lab}</p>'
        f'<div class="status__value">{val}</div><p class="status__ctx">{ctx}</p>'
        f'<p class="sb-small">{link}</p></article>'
        for lab, val, ctx, link in tiles
    )
    cards = "".join(
        f'<a class="sb-card sb-spotlight sb-reveal page-card" href="{href}"><h3 class="sb-card__title">{t}</h3>'
        f"<p>{d}</p><span class=go>{href} →</span></a>"
        for href, t, d in (
            (
                "predict.html",
                "Predict",
                "Score an applicant with the live API: probability, decision at your cutoff, SHAP waterfall.",
            ),
            (
                "evaluation.html",
                "Evaluation",
                "Model comparison, calibration, the cost slider on real held-out predictions, fairness and errors.",
            ),
            (
                "monitoring.html",
                "Monitoring",
                "Data quality, seven drift scenarios and the policy decisions, Evidently reports.",
            ),
            (
                "registry.html",
                "Registry",
                "Promotion policy checks, champion/challenger history, the rollback simulation, MLflow.",
            ),
        )
    )
    return section(
        "status",
        "01",
        "Status",
        'What is <span class="sb-serif">live</span> right now.',
        "Every value below is read from files committed by the CI workflows when this page was built.",
        f'<div class="status-grid">{body}</div><div class="page-cards" style="margin-top:1.25rem">{cards}</div>',
    )


def benchmark_section(bench: dict | None) -> str:
    if not bench:
        return ""
    ip = bench["in_process"]
    rows = []
    for path, sgl in sorted(ip["single"].items(), key=lambda kv: kv[0] != "pipeline"):
        rows.append(
            [
                esc(path),
                "1",
                f"{sgl['p50_ms']:.2f}",
                f"{sgl['p95_ms']:.2f}",
                f"{sgl['p99_ms']:.2f}",
                f"{1000 / sgl['p50_ms']:,.0f}",
            ]
        )
    for b in ip["batch"]:
        for path in ("pipeline", "fast"):
            if path in b:
                x = b[path]
                rows.append(
                    [
                        esc(path),
                        f"{b['batch']:,}",
                        f"{x['p50_ms']:.2f}",
                        f"{x['p95_ms']:.2f}",
                        f"{x['p99_ms']:.2f}",
                        f"{x['rows_per_s']:,.0f}",
                    ]
                )
    inproc = card(
        "In-process scoring",
        f"Pipeline vs the NumPy fast path; identical probabilities on all {ip['n_test_rows']:,} "
        f"test rows: <strong>{ip['identical_on_test_split']}</strong>.",
        table(
            ["Path", "Rows", "p50 ms", "p95 ms", "p99 ms", "Rows / s"],
            rows,
            "In-process scoring latency",
        ),
        provenance(bench, "results/benchmarks/benchmark.json"),
    )
    http_rows = []
    for h in bench["http"]:
        for name, key in (
            ("/predict", "predict"),
            ("/predict_batch (100)", "predict_batch_100"),
            ("/explain", "explain"),
        ):
            for x in h[key]:
                http_rows.append(
                    [
                        "on" if h["fast_scoring"] else "off",
                        esc(name),
                        str(x["concurrency"]),
                        f"{x['requests_per_s']:.1f}",
                        f"{x['p50_ms']:.1f}",
                        f"{x['p95_ms']:.1f}",
                        f"{x['p99_ms']:.1f}",
                        str(x["errors"]),
                    ]
                )
    mem = ", ".join(
        f"fast path {'on' if h['fast_scoring'] else 'off'}: {h['server_rss_idle_mb']:.0f} MB idle / "
        f"{h['server_rss_after_load_mb']:.0f} MB after load"
        for h in bench["http"]
    )
    http = card(
        "HTTP, one uvicorn worker",
        f"Real test rows, clients on the same {esc(bench['environment']['machine'])} "
        f"({bench['environment']['cpu_count']} CPUs). Server memory: {esc(mem)}.",
        table(
            ["Fast path", "Endpoint", "Clients", "Req / s", "p50 ms", "p95 ms", "p99 ms", "Errors"],
            http_rows,
            "HTTP benchmark",
        ),
        "Relative comparison on a CI runner, not a capacity claim for the free hosting tier.",
    )
    lrows = [
        [
            esc(x["strategy"]),
            "-" if x["file_mb"] is None else f"{x['file_mb']:.2f}",
            f"{x['load_s_median']:.3f}",
            f"{x['first_predict_s_median']:.3f}",
            f"{x['rss_mb_median']:.0f}",
            str(x["same_predictions"]),
        ]
        for x in bench["loading"]
    ]
    loading = card(
        "Model-loading strategies",
        "Each in a fresh process (median of repeats).",
        table(
            ["Strategy", "File MB", "Load s", "First predict s", "RSS MB", "Same output"],
            lrows,
            "Model loading",
        ),
        "The API loads the compressed joblib bundle once at startup.",
    )
    return f'<div class="stack">{inproc}{http}{loading}</div>'


# ---------------------------------------------------------------------------- evaluation
def decision_tool_section(m: dict, decision: dict | None) -> str:
    t = m["test"]
    c = m["cost"]
    controls = f"""
<div class="tool-controls">
  <div class="range-row">
    <label class="sb-label" for="ev-threshold">Cutoff: decline when P(default) ≥</label>
    <output id="ev-threshold-out" for="ev-threshold">{t["threshold"]:.3f}</output>
    <input type="range" id="ev-threshold" min="0.005" max="0.995" step="0.005" value="{t["threshold"]:.3f}" aria-describedby="ev-help">
  </div>
  <div class="two-col">
    <div class="sb-field"><label class="sb-label" for="ev-fn">Cost of a missed default (FN)</label>
      <input class="sb-input" id="ev-fn" type="number" min="0.01" max="1000" step="any" value="{c["fn_cost"]:g}"></div>
    <div class="sb-field"><label class="sb-label" for="ev-fp">Cost of a false alarm (FP)</label>
      <input class="sb-input" id="ev-fp" type="number" min="0.01" max="1000" step="any" value="{c["fp_cost"]:g}"></div>
  </div>
  <div class="sb-field"><label class="sb-label" for="ev-split">Held-out split</label>
    <select class="sb-select" id="ev-split"><option value="test" selected>test (n = {t["n"]:,})</option><option value="validation">validation (n = {m["data"]["n_validation"]:,})</option></select></div>
  <div class="btn-row">
    <button type="button" class="sb-btn sb-btn--sm" data-ev-set="optimal">Optimal for these costs</button>
    <button type="button" class="sb-btn sb-btn--sm" data-ev-set="deployed">Deployed cutoff</button>
    <button type="button" class="sb-btn sb-btn--sm" data-ev-set="bayes">Bayes cutoff</button>
    <button type="button" class="sb-btn sb-btn--sm" data-ev-set="half">0.5</button>
  </div>
  <p class="sb-small" id="ev-help">Optimal = minimum expected cost on the <em>validation</em> split
    (<span data-ev="optimal">-</span>); Bayes cutoff fp/(fp+fn) = <span data-ev="bayes">-</span>
    for a <span data-ev="ratio">-</span>:1 ratio. Costs are relative units, not money.</p>
  <p class="tool-status" id="ev-status" role="status" aria-live="polite">Loading the held-out predictions…</p>
</div>"""
    outputs = f"""
<div class="tool-out">
  <div class="cm" role="group" aria-label="Confusion matrix at the selected cutoff">
    <div class="cm-h"></div><div class="cm-h">Declined (predicted default)</div><div class="cm-h">Approved</div>
    <div class="cm-h">Defaulted</div>
    <div class="is-good"><span class="cm-v" data-ev="tp">-</span>caught (TP)</div>
    <div class="is-bad"><span class="cm-v" data-ev="fn">-</span>missed (FN)</div>
    <div class="cm-h">Repaid</div>
    <div class="is-bad"><span class="cm-v" data-ev="fp">-</span>false alarm (FP)</div>
    <div class="is-good"><span class="cm-v" data-ev="tn">-</span>approved (TN)</div>
  </div>
  <div class="mini-stats" aria-live="polite">
    <div><b data-ev="cost">-</b><span>cost / applicant · <span data-ev="vs05"></span></span></div>
    <div><b data-ev="recall">-</b><span>recall (defaults caught)</span></div>
    <div><b data-ev="precision">-</b><span>precision</span></div>
    <div><b data-ev="approval">-</b><span>approval rate</span></div>
    <div><b data-ev="fpr">-</b><span>false-positive rate</span></div>
    <div><b data-ev="fnr">-</b><span>false-negative rate</span></div>
    <div><b data-ev="f1">-</b><span>F1</span></div>
    <div><b data-ev="total">-</b><span>total cost (<span data-ev="n"></span> applicants)</span></div>
  </div>
  {legend([("s-1", "expected cost / applicant"), ("s-2", "selected cutoff")])}
  <div class="chart-host" id="ev-chart"></div>
</div>"""
    tool = (
        f'<div class="sb-card sb-reveal"><div class="tool" id="ev-tool" data-src="data/predictions.json">'
        f"{controls}{outputs}</div></div>"
    )
    rule_rows = []
    if decision:
        for r in decision["rules"]:
            rule_rows.append(
                [
                    esc(r["rule"]),
                    f"{r['threshold']:.3f}",
                    f4(r["business_cost"]),
                    pct(r["recall"]),
                    pct(r["precision"]),
                    pct(r["approval_rate"]),
                    pct(r["fpr"]),
                    pct(r["fnr"]),
                    f"{r['tp']:,}",
                    f"{r['fp']:,}",
                    f"{r['tn']:,}",
                    f"{r['fn']:,}",
                ]
            )
        rules = card(
            "Threshold rules (test split, cutoffs chosen on validation)",
            esc(decision["disclaimer"]),
            table(
                [
                    "Rule",
                    "Cutoff",
                    "Cost / applicant",
                    "Recall",
                    "Precision",
                    "Approval",
                    "FPR",
                    "FNR",
                    "TP",
                    "FP",
                    "TN",
                    "FN",
                ],
                rule_rows,
                "Threshold rules",
                best=0,
            ),
            provenance(m, "results/decision.json"),
        )
    else:
        rules = ""
    return section(
        "decision",
        "01",
        "Decision",
        'Move the cutoff, see the <span class="sb-serif">cost</span>.',
        "The slider recomputes the confusion matrix and every decision metric from the committed "
        "held-out predictions of the served model, at any cutoff and any cost ratio. This is an "
        "educational decision-support tool, not an approved lending engine.",
        tool + f'<div class="stack">{rules}</div>',
    )


def sensitivity_section(decision: dict | None) -> str:
    if not decision:
        return ""
    rows = [
        [
            f"{r['ratio']:g}:1",
            f"{r['bayes_threshold']:.3f}",
            f"{r['optimal_threshold']:.3f}",
            f4(r["test_cost_optimal"]),
            f4(r["test_cost_deployed"]),
            minus(f"{r['regret_deployed']:+.4f}"),
            f4(r["test_cost_0_5"]),
            pct(r["recall_optimal"]),
            pct(r["approval_rate_optimal"]),
        ]
        for r in decision["sensitivity"]
    ]
    tbl = table(
        [
            "FN:FP",
            "Bayes cutoff",
            "Optimal (val.)",
            "Cost at optimum",
            "Cost at deployed",
            "Regret of deployed",
            "Cost at 0.5",
            "Recall",
            "Approval",
        ],
        rows,
        "Cost-ratio sensitivity",
    )
    chart = chart_card(
        "How the optimal cutoff moves with the cost ratio",
        "Empirical optimum on validation vs the theoretical Bayes cutoff for calibrated scores.",
        "chart-sensitivity",
        tbl,
        "Regret = extra test cost of keeping the deployed cutoff if the true ratio were different. Source: results/decision.json",
        legend=legend(
            [("s-1", "optimal cutoff (validation)"), ("dash", "Bayes cutoff fp/(fp+fn)")]
        ),
    )
    p = decision["policy"]
    return section(
        "sensitivity",
        "02",
        "Sensitivity",
        'If the cost ratio is <span class="sb-serif">wrong</span>.',
        f"The configured ratio is {p['ratio']:g}:1, an assumption about card economics. The table shows "
        "what each other ratio would imply and what keeping today's cutoff would cost.",
        embed(
            "decision-data", {"sensitivity": decision["sensitivity"], "policy": decision["policy"]}
        )
        + chart,
    )


def models_compare_section(m: dict, ev: dict | None, figures: Path) -> str:
    champ = m["champion"]
    rows, best = [], None
    for i, f in enumerate(m["families"]):
        t, ci = f["test"], f.get("test_ci95", {})

        def withci(k, t=t, ci=ci):
            v = f4(t[k])
            return f'{v} <span class="sd">[{ci[k][0]:.3f}, {ci[k][1]:.3f}]</span>' if k in ci else v

        name = esc(f["family"])
        if f["family"] == champ:
            best = i
            name += ' <span class="sb-tag tag-champ">champion</span>'
        if not f.get("deployable", True):
            name += ' <span class="sb-tag">research only</span>'
        rows.append(
            [
                name,
                f"{f4(f['cv_pr_auc'])}"
                + (
                    f' <span class="sd">± {f["cv_pr_auc_std"]:.3f}</span>'
                    if f.get("cv_pr_auc_std")
                    else ""
                ),
                withci("roc_auc"),
                withci("pr_auc"),
                withci("brier"),
                f4(t.get("ece")),
                withci("business_cost"),
                f4(t["threshold"]),
                pct(t["recall"]),
                pct(t.get("precision", 0)),
                pct(t.get("specificity", 0)),
                f4(t.get("balanced_accuracy")),
                f4(t["f1"]),
                esc(f.get("calibration", "isotonic")),
            ]
        )
    skipped = "".join(
        f"; {esc(s['family'])} skipped ({esc(s['reason'])})" for s in m.get("skipped_families", [])
    )
    fam = card(
        "Every family under one protocol (test split, n = {:,})".format(m["test"]["n"]),
        "Optuna on stratified CV PR-AUC (train split) → fit on train → calibration chosen on "
        "validation Brier → cost-optimal cutoff on validation → scored once on test. Brackets: "
        "stratified bootstrap 95% intervals.",
        table(
            [
                "Model",
                "CV PR-AUC",
                "ROC-AUC",
                "PR-AUC",
                "Brier",
                "ECE",
                "Cost / applicant",
                "Cutoff",
                "Recall",
                "Precision",
                "Specificity",
                "Bal. acc.",
                "F1",
                "Calibration",
            ],
            rows,
            "Model comparison",
            best=best,
        ),
        f"Selection rule: {esc(m.get('selection_rule', 'lowest validation expected cost'))}. "
        f"Excluded model inputs: {esc(', '.join(m.get('excluded_features', [])) or 'none')}{skipped}. "
        + provenance(m, "results/metrics.json"),
    )
    curves = ""
    if ev and ev.get("curves"):
        fams = list(ev["curves"])
        opts = "".join(
            f'<option value="{esc(f)}"{" selected" if i == 0 and f != champ else ""}>{esc(f)}</option>'
            for i, f in enumerate([x for x in fams if x != champ])
        )
        controls = f'<div class="chart-controls"><label for="curve-family">Compare the champion with <select class="sb-select" id="curve-family">{opts}</select></label></div>'
        auc_rows = [
            [esc(f["family"]), f4(f["test"]["roc_auc"]), f4(f["test"]["pr_auc"])]
            for f in m["families"]
        ]
        paired = ev.get("paired_vs_champion", {})
        prow = [
            [
                esc(k),
                minus(f"{v['pr_auc']['mean']:+.4f}"),
                f"[{v['pr_auc']['ci95'][0]:+.4f}, {v['pr_auc']['ci95'][1]:+.4f}]",
                minus(f"{v['business_cost']['mean']:+.4f}"),
                f"[{v['business_cost']['ci95'][0]:+.4f}, {v['business_cost']['ci95'][1]:+.4f}]",
            ]
            for k, v in paired.items()
        ]
        curves = (
            '<div class="sb-grid-2">'
            + chart_card(
                "ROC curves",
                "Test split.",
                "chart-roc",
                table(["Model", "ROC-AUC", "PR-AUC"], auc_rows, "Areas"),
                "",
                controls,
                legend([("s-1", f"{champ} (champion)"), ("s-2", "selected family")]),
            )
            + chart_card(
                "Precision-recall curves",
                "Test split; base rate {:.1%}.".format(m["data"]["default_rate"]),
                "chart-pr",
                table(["Model", "ROC-AUC", "PR-AUC"], auc_rows, "Areas"),
                "",
                "",
                legend([("s-1", f"{champ} (champion)"), ("s-2", "selected family")]),
            )
            + "</div>"
        )
        if prow:
            curves += card(
                "Champion minus each other family (paired bootstrap on the test split)",
                "Positive PR-AUC difference = champion ranks better; negative cost difference = champion is cheaper.",
                table(
                    ["Family", "Δ PR-AUC", "95% CI", "Δ cost / applicant", "95% CI"],
                    prow,
                    "Paired differences",
                ),
                "Overlapping intervals mean the families are not distinguishable on this test set.",
            )
    abl = ""
    if m.get("ablation"):
        strategies = list(dict.fromkeys(a["strategy"] for a in m["ablation"]))
        look = {(a["family"], a["strategy"]): a for a in m["ablation"]}
        arows = []
        for fam_name in dict.fromkeys(a["family"] for a in m["ablation"]):
            vals = [look.get((fam_name, s_)) for s_ in strategies]
            top = max((a["cv_pr_auc"] for a in vals if a), default=None)
            cells = [esc(fam_name)]
            for a in vals:
                txt = (
                    "-"
                    if not a
                    else f'{f4(a["cv_pr_auc"])} <span class="sd">± {f4(a["cv_pr_auc_std"])}</span>'
                )
                cells.append(f"<strong>{txt}</strong>" if a and a["cv_pr_auc"] == top else txt)
            arows.append(cells)
        abl = card(
            "Imbalance ablation",
            "CV PR-AUC ± std on the training split with each family's tuned hyper-parameters; SMOTE runs inside the folds.",
            table(["Model"] + strategies, arows, "Imbalance ablation"),
            "Source: results/metrics.json",
        )
    return section(
        "models",
        "03",
        "Models",
        f'{len(m["families"])} families, one <span class="sb-serif">protocol</span>.',
        "ROC-AUC is reported but never used to choose: the champion has the lowest expected cost "
        "on validation, and differences are shown with their uncertainty.",
        embed(
            "evaluation-data",
            {
                "champion": champ,
                "curves": (ev or {}).get("curves", {}),
                "reliability": (ev or {}).get("reliability", {}),
            },
        )
        + f'<div class="stack">{fam}{curves}{abl}</div>',
    )


def calibration_section(m: dict, figures: Path) -> str:
    comp = m.get("calibration_comparison")
    if not comp:
        return ""
    rows = [
        [
            esc(c["method"])
            + (' <span class="sb-tag tag-champ">chosen</span>' if c["chosen"] else ""),
            f4(c["validation"]["brier"]),
            f4(c["validation"]["log_loss"]),
            f4(c["validation"]["ece"]),
            f4(c["test"]["brier"]),
            f4(c["test"]["log_loss"]),
            f4(c["test"]["ece"]),
            f4(c["test"]["roc_auc"]),
            f4(c["test"]["pr_auc"]),
            f4(c["test"]["business_cost"]),
            f"{c['test']['threshold']:.3f}",
        ]
        for c in comp
    ]
    best = next((i for i, c in enumerate(comp) if c["chosen"]), None)
    tbl = table(
        [
            "Method",
            "Val Brier",
            "Val log loss",
            "Val ECE",
            "Test Brier",
            "Test log loss",
            "Test ECE",
            "Test ROC-AUC",
            "Test PR-AUC",
            "Test cost",
            "Cutoff",
        ],
        rows,
        "Calibration comparison",
        best=best,
    )
    chart = chart_card(
        "Reliability on the test split",
        "Deciles of predicted probability: mean predicted vs observed default rate. The diagonal is perfect calibration.",
        "chart-reliability",
        tbl,
        "All three are fitted on the calibration split only; the chosen one has the lowest validation Brier. Source: results/metrics.json",
        legend=legend(
            [
                ("s-1", "sigmoid (Platt)"),
                ("s-2", "isotonic"),
                ("s-muted", "none"),
                ("dash", "perfect calibration"),
            ]
        ),
    )
    return section(
        "calibration",
        "04",
        "Calibration",
        'Probabilities you can <span class="sb-serif">price</span>.',
        f"The cost rule needs real probabilities. Champion ({esc(m['champion'])}): uncalibrated, "
        "Platt (sigmoid) and isotonic calibrators compared on data none of them was fitted on.",
        '<div class="stack">'
        + card(
            "Calibration comparison",
            "",
            tbl,
            "Lower Brier / log loss / ECE is better. Isotonic's step function creates ties, which can lower PR-AUC.",
        )
        + chart
        + "</div>",
    )


def explain_section(ev: dict | None, figures: Path) -> str:
    if not ev or not ev.get("importance"):
        return ""
    imp = ev["importance"]
    perm = {r["feature"]: r for r in imp["permutation_raw_inputs"]}
    rows = [
        [
            esc(r["feature"]),
            str(r["shap_rank"]),
            f"{r['mean_abs_shap']:.4f}",
            str(r["native_rank"]),
            f"{100 * r['native_share']:.1f}%",
            f"{perm[r['feature']]['pr_auc_drop']:.4f}" if r["feature"] in perm else "derived",
        ]
        for r in imp["model_inputs"][:15]
    ]
    rho = imp.get("spearman_shap_vs_native")
    tbl = card(
        "Three importance methods",
        f"Mean |SHAP| (test sample) vs the model's own {esc(imp['native_kind'])} vs permutation importance "
        "(drop in test PR-AUC when one raw input is shuffled). "
        + (f"Spearman rank correlation SHAP vs native: {rho:.2f}." if rho is not None else ""),
        table(
            [
                "Feature",
                "SHAP rank",
                "Mean |SHAP|",
                "Native rank",
                "Native share",
                "Permutation Δ PR-AUC",
            ],
            rows,
            "Feature importance",
        ),
        esc(imp["note"]),
    )
    figs = [
        (
            "shap_bar.png",
            "SHAP importance",
            "Mean |SHAP| per model input.",
            "Bar chart of mean absolute SHAP values",
        ),
        (
            "shap_beeswarm.png",
            "SHAP beeswarm",
            "Direction and spread per feature.",
            "SHAP beeswarm plot",
        ),
    ]
    figs += [
        (
            f.name,
            f"Dependence: {f.stem.removeprefix('shap_dependence_')}",
            "SHAP value against the feature value.",
            f"SHAP dependence plot for {f.stem}",
        )
        for f in sorted(figures.glob("shap_dependence_*.png"))
    ]
    figs += [
        (
            f.name,
            f"Waterfall: {f.stem.removeprefix('shap_waterfall_')} test applicant",
            "How one prediction was reached.",
            f"SHAP waterfall for a {f.stem} applicant",
        )
        for f in sorted(figures.glob("shap_waterfall_*.png"))
    ]
    fig_html = "".join(
        figure(f"results/figures/{n}", figures / n, t, s_, a)
        for n, t, s_, a in figs
        if (figures / n).exists()
    )
    return section(
        "explain",
        "05",
        "Explainability",
        'Why a score is <span class="sb-serif">high</span>.',
        "Associations inside the model, not causes. SHAP values are in the base model's units "
        "before calibration; the calibrator is monotone, so directions carry over.",
        tbl + (f'<div class="fig-grid">{fig_html}</div>' if fig_html else ""),
    )


def fairness_section(fair: dict | None) -> str:
    if not fair:
        return ""
    blocks = []
    for attr, b in fair["attributes"].items():
        rows = []
        for g in b["groups"]:
            ci = g.get("ci95", {})

            def cell(k, g=g, ci=ci):
                v = f"{g[k]:.3f}"
                return (
                    f'{v} <span class="sd">[{ci[k][0]:.3f}, {ci[k][1]:.3f}]</span>'
                    if k in ci
                    else v
                )

            rows.append(
                [
                    esc(g["group"]) + (' <span class="sb-tag">small</span>' if g["small"] else ""),
                    f"{g['n']:,}",
                    f"{g['observed_rate']:.3f}",
                    cell("decline_rate"),
                    cell("recall"),
                    cell("fpr"),
                    cell("fnr"),
                    f"{g['mean_predicted']:.3f}",
                    cell("calibration_gap"),
                ]
            )
        s_ = b["summary"]
        blocks.append(
            card(
                attr.replace("_", " ").title(),
                f"Approval-rate ratio lowest/highest {s_['approval_rate_ratio_min_max']:.2f} · FPR gap {s_['fpr_gap']:.3f} · FNR gap {s_['fnr_gap']:.3f} · observed default rate {s_['observed_rate_range'][0]:.3f}–{s_['observed_rate_range'][1]:.3f}",
                table(
                    [
                        "Group",
                        "n",
                        "Default rate",
                        "Decline rate",
                        "Recall",
                        "FPR",
                        "FNR",
                        "Mean predicted",
                        "Calibration gap",
                    ],
                    rows,
                    f"Fairness by {attr}",
                ),
                "",
            )
        )
    aware = fair.get("with_excluded_features")
    cmp_ = ""
    if aware:
        aw = aware["test"]
        rows = []
        for attr in ("SEX", "MARRIAGE", "AGE_BAND"):
            a, b = fair["attributes"].get(attr), aware["report"]["attributes"].get(attr)
            if a and b:
                rows.append(
                    [
                        esc(attr),
                        f"{a['summary']['fpr_gap']:.3f}",
                        f"{b['summary']['fpr_gap']:.3f}",
                        f"{a['summary']['fnr_gap']:.3f}",
                        f"{b['summary']['fnr_gap']:.3f}",
                        f"{a['summary']['approval_rate_ratio_min_max']:.2f}",
                        f"{b['summary']['approval_rate_ratio_min_max']:.2f}",
                    ]
                )
        cmp_ = card(
            "Served model vs the same family trained with SEX, MARRIAGE and AGE",
            f"With the attributes as inputs: test PR-AUC {aw['pr_auc']:.4f}, cost {aw['business_cost']:.4f}, Brier {aw['brier']:.4f}.",
            table(
                [
                    "Attribute",
                    "FPR gap (served)",
                    "FPR gap (with attrs)",
                    "FNR gap (served)",
                    "FNR gap (with attrs)",
                    "Approval ratio (served)",
                    "Approval ratio (with attrs)",
                ],
                rows,
                "Excluded-attribute comparison",
            ),
            "Excluding attributes is not fairness by itself: other features can act as proxies.",
        )
    lim = "".join(f"<li>{esc(x)}</li>" for x in fair["limitations"])
    return section(
        "fairness",
        "06",
        "Fairness",
        'Outcomes by <span class="sb-serif">group</span>.',
        f"Test split at the deployed cutoff ({fair['threshold']:.3f}). Groups use attributes recorded "
        "in the data; nothing is inferred. Brackets are bootstrap 95% intervals.",
        f'<div class="stack">{cmp_}{"".join(blocks)}'
        + card(
            "Limitations", "", f'<ul class="meta-list">{lim}</ul>', "Source: results/fairness.json"
        )
        + "</div>",
    )


def errors_section(ev: dict | None) -> str:
    if not ev or not ev.get("error_analysis"):
        return ""
    ea = ev["error_analysis"]
    cards = []
    for name, rows in ea["segments"].items():
        cards.append(
            card(
                f"By {esc(name)}",
                "",
                table(
                    [
                        "Segment",
                        "n",
                        "Default rate",
                        "Decline rate",
                        "FNR",
                        "FPR",
                        "Cost / applicant",
                        "Share of cost",
                    ],
                    [
                        [
                            esc(r["segment"]),
                            f"{r['n']:,}",
                            f"{r['default_rate']:.3f}",
                            f"{r['decline_rate']:.3f}",
                            f"{r['fnr']:.3f}",
                            f"{r['fpr']:.3f}",
                            f"{r['cost_per_applicant']:.3f}",
                            pct(r["share_of_total_cost"]),
                        ]
                        for r in rows
                    ],
                    f"Errors by {name}",
                ),
                "",
            )
        )
    prof = ea["profiles"]
    keys = [k for k in prof["FN"] if k != "n"]
    prow = [
        [esc(k)]
        + [("-" if prof[g][k] is None else f"{prof[g][k]:,.2f}") for g in ("FN", "TP", "FP", "TN")]
        for k in keys
    ]
    prow.append(["rows"] + [f"{prof[g]['n']:,}" for g in ("FN", "TP", "FP", "TN")])
    share = ea.get("fn_share_pay0_not_late")
    lead = (
        f"At the deployed cutoff, {share:.0%} of missed defaults had no delay on their latest bill "
        f"(PAY_0 ≤ 0) and {ea['confident_misses']} were scored below half the cutoff."
        if share is not None
        else ""
    )
    cards.insert(
        0,
        card(
            "Average profile of each outcome",
            lead,
            table(
                ["Feature", "FN (missed)", "TP", "FP (false alarm)", "TN"], prow, "Outcome profiles"
            ),
            esc(ea["note"]),
        ),
    )
    return section(
        "errors",
        "07",
        "Error analysis",
        'Where the model is <span class="sb-serif">wrong</span>.',
        "Test split, deployed cutoff. Descriptive segments to guide features and monitoring; not causal.",
        f'<div class="stack">{"".join(cards)}</div>',
    )


# ---------------------------------------------------------------------------- monitoring
def scenarios_section(sc: dict | None) -> str:
    if not sc:
        return ""
    rows = []
    for s in sc["scenarios"]:
        perf = s["performance"]
        rows.append(
            [
                esc(s["scenario"]),
                f'<span class="cell-wrap">{esc(s["description"])}</span>',
                f"{s['n_drifted_features']} / {s['n_significant_features']}",
                f"{s['prediction_drift']['psi']:.3f}",
                f"{s['prediction_drift']['decline_rate_current']:.3f}",
                f"{s['labels']['rate']:.3f}",
                f"{perf['roc_auc']:.3f}",
                minus(f"{perf['calibration_gap']:+.3f}"),
                f"{100 * perf['cost_regret']:.1f}%",
                pill(s["label_free_decision"]["action"]),
                pill(s["decision"]["action"]),
            ]
        )
    pol = sc["policy"]
    summary = card(
        "Seven simulated batches, one policy",
        f"Batches of {sc['scenarios'][0]['n']:,} rows drawn from the test split; feature reference = "
        f"training split ({sc['reference']['n']:,} rows); prediction reference = validation-split scores. "
        f"Model: {esc(sc['model']['family'])}, cutoff {sc['model']['threshold']:.3f}. Baseline test ROC-AUC "
        f"{sc['baseline_test']['roc_auc']:.3f}.",
        table(
            [
                "Batch",
                "What changed",
                "Drifted / significant features",
                "Prediction PSI",
                "Decline rate",
                "Default rate",
                "ROC-AUC",
                "Calibration gap",
                "Cost regret",
                "Action without labels",
                "Action with labels",
            ],
            rows,
            "Drift scenarios",
        ),
        provenance(sc, "results/monitoring/scenarios.json"),
    )
    opts = "".join(
        f'<option value="{esc(s["scenario"])}"{" selected" if s["scenario"] == "covariate_shift" else ""}>{esc(s["scenario"])}</option>'
        for s in sc["scenarios"]
    )
    first = next(
        (s for s in sc["scenarios"] if s["scenario"] == "covariate_shift"), sc["scenarios"][0]
    )
    psi_rows = [
        [
            esc(r["feature"]),
            esc(r["test"]),
            f"{r['psi']:.3f}",
            esc(r["psi_band"]),
            f"{r['p_value']:.2g}",
            "yes" if r["drifted"] else "no",
        ]
        for r in sorted(first["feature_drift"], key=lambda r: -r["psi"])
    ]
    explorer = chart_card(
        "Per-feature drift",
        "PSI per input feature (top 12); bars past 0.2 are major, 0.1–0.2 moderate. Hover a bar for the test and p-value.",
        "chart-psi",
        table(
            ["Feature", "Test", "PSI", "Band", "p-value", "Drift"],
            psi_rows,
            "Feature drift (covariate_shift)",
        ),
        "The data table shows the covariate_shift batch. A feature counts as drifted when its PSI is major, or moderate with p < "
        f"{pol['p_value']}.",
        controls=f'<div class="chart-controls"><label for="sc-select">Batch <select class="sb-select" id="sc-select">{opts}</select></label></div>',
        legend=legend(
            [("s-2", "major (≥ 0.2)"), ("s-1", "moderate (0.1–0.2)"), ("s-muted", "stable")]
        ),
    )
    detail = (
        f'<dl class="kv" id="sc-detail"><div><dt>Batch</dt><dd data-sc="desc">{esc(first["description"])}</dd></div>'
        f'<div><dt>Without labels</dt><dd data-sc="free">{esc(first["label_free_decision"]["action"])}</dd></div>'
        f'<div><dt>With labels</dt><dd data-sc="labelled">{esc(first["decision"]["action"])}</dd></div>'
        f'<div><dt>Why</dt><dd data-sc="why">{esc("; ".join(first["decision"]["reasons"]) or "nothing moved")}</dd></div></dl>'
    )
    policy = (
        f"<p>Block when any input misses more than {pol['max_missing_rate']:.0%} or unseen codes "
        f"appear. Retrain (a gated challenger) when more than {pol['feature_drift_share']:.0%} of "
        f"features drift or prediction PSI ≥ {pol['prediction_psi']} while no labels exist yet, or "
        f"once ≥ {pol['min_labelled_rows']:,} labels arrive, when ROC-AUC drops by more than "
        f"{pol['max_roc_auc_drop']}, the calibration gap exceeds ±{pol['max_calibration_gap']}, or the "
        f"deployed cutoff costs more than {pol['max_cost_regret']:.0%} above the batch optimum. "
        "With labels, drift that the model demonstrably handles is only monitored. Significance "
        "alone never triggers an action: with thousands of rows, KS flags shifts too small to matter.</p>"
    )
    slim = {
        "scenarios": [
            {
                k: s[k]
                for k in (
                    "scenario",
                    "description",
                    "feature_drift",
                    "decision",
                    "label_free_decision",
                )
            }
            for s in sc["scenarios"]
        ]
    }
    return section(
        "scenarios",
        "01",
        "Drift scenarios",
        'Drift is not one <span class="sb-serif">thing</span>.',
        "Covariate shift, a broken feed, new category codes, a riskier applicant mix, prior shift "
        "and concept drift each need a different response. The UCI table has no timestamps, so "
        "the batches are simulated and documented in <code>credit_default.monitor.scenarios</code>.",
        embed("scenarios-data", slim)
        + f'<div class="stack">{summary}<div class="split"><div class="prose sb-reveal">{policy}{detail}</div>{explorer}</div></div>',
    )


def quality_section(dq: dict | None) -> str:
    if not dq:
        return ""
    d, lk = dq["duplicates"], dq["leakage"]
    checks = "".join(
        f'<li><span class="{"check-pass" if c["passed"] else "check-fail"}">{"PASS" if c["passed"] else "FAIL"}</span> {esc(c["check"])}</li>'
        for c in lk["checks"]
    )
    facts = (
        f'<dl class="kv"><div><dt>Rows</dt><dd>{dq["n_rows"]:,} · default rate {dq["default_rate"]:.1%} · {sum(c["n_missing"] for c in dq["columns"])} missing values</dd></div>'
        f"<div><dt>Contract</dt><dd>{'PASS' if dq['schema']['passed'] else 'FAIL'} (Pandera: types, domains, ranges, unique IDs)</dd></div>"
        f"<div><dt>Duplicates</dt><dd>{d['duplicate_ids']} duplicate IDs · {d['duplicate_rows_excluding_id']} duplicate rows · {d['duplicate_feature_vectors']} duplicate feature vectors ({d['groups_with_conflicting_labels']} with conflicting labels)</dd></div>"
        f"<div><dt>Overlap</dt><dd>{lk['test_rows_with_identical_feature_vector_in_train']} test rows ({lk['test_rows_with_identical_feature_vector_in_train_share']:.2%}) share a feature vector with a training row</dd></div>"
        f"<div><dt>Split</dt><dd>{esc(lk['temporal_note'])}</dd></div></dl>"
    )
    code_rows = [
        [
            esc(c["column"]),
            str(c["code"]),
            f"{c['n']:,}",
            "-" if c["default_rate"] is None else f"{c['default_rate']:.1%}",
            esc(c["handling"]),
        ]
        for c in dq["undocumented_codes"]
    ]
    flag_rows = [
        [
            esc(f["check"]),
            f"{f['n']:,}",
            f"{f['share']:.1%}",
            "-" if f["default_rate"] is None else f"{f['default_rate']:.1%}",
        ]
        for f in dq["domain_flags"]
    ]
    uni = ", ".join(f"{esc(u['feature'])} {u['auc']:.3f}" for u in lk["univariate_auc_top"][:5])
    return section(
        "quality",
        "02",
        "Data quality",
        'Validated before it is <span class="sb-serif">trusted</span>.',
        "The Pandera contract stops the pipeline on a violation; this report describes what passed.",
        f'<div class="split"><div class="sb-reveal">{facts}</div>'
        + card(
            "Leakage and split checks",
            f"Strongest single features (univariate ROC-AUC): {uni}.",
            f'<ul class="meta-list">{checks}</ul>',
            provenance(dq, "results/data_quality.json"),
        )
        + "</div>"
        + '<div class="sb-grid-2" style="margin-top:1.25rem">'
        + card(
            "Undocumented codes",
            "",
            table(
                ["Column", "Code", "Rows", "Default rate", "Handling"],
                code_rows,
                "Undocumented codes",
            ),
            "",
        )
        + card(
            "Domain flags (kept)",
            "",
            table(["Check", "Rows", "Share", "Default rate"], flag_rows, "Domain flags"),
            "",
        )
        + "</div>",
    )


def retrain_history_section(history: list | None) -> str:
    if not history:
        return ""
    items = []
    for e in reversed(history):
        trig = e.get("trigger") or e.get("source", "")
        items.append(
            f'<li><div class="t-head">{pill(e["event"])}<span class="t-title">#{e["id"]} · {esc(str(e.get("ts", ""))[:16].replace("T", " "))}</span></div>'
            f'<p class="t-meta">trigger: {esc(trig)}'
            + (f' · <a href="{esc(e["run_url"])}">workflow run</a>' if e.get("run_url") else "")
            + f"</p><p>{esc('; '.join(e.get('reasons', [])))}</p></li>"
        )
    return section(
        "retraining",
        "04",
        "Retraining history",
        'Every retrain is <span class="sb-serif">recorded</span>.',
        "Scheduled drift checks call the training workflow only when the policy says retrain; each "
        "run's gate decision is appended here, rejections included.",
        card(
            "Training and promotion events",
            "",
            f'<ol class="timeline">{"".join(items)}</ol>',
            "Source: results/governance/history.json",
        ),
    )


# ---------------------------------------------------------------------------- registry
def checks_table(checks: list[dict], caption: str) -> str:
    rows = []
    for c in checks:
        state = (
            '<span class="check-skip">skipped</span>'
            if c["skipped"]
            else (
                '<span class="check-pass">pass</span>'
                if c["passed"]
                else '<span class="check-fail">FAIL</span>'
            )
        )
        v = c["value"]
        v = f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else esc(str(v)))
        rows.append([esc(c["name"]), esc(c["group"]), state, v, esc(c["limit"])])
    return table(["Check", "Group", "Result", "Value", "Limit"], rows, caption)


def gate_checks_section(gate: dict | None, challenger: dict | None) -> str:
    if not gate:
        return ""
    ok = gate.get("passed")
    verdict = (
        f'<p class="verdict-word {"is-pass" if ok else "is-fail"}">{"PASS" if ok else "REJECT"}</p>'
    )
    reasons = "".join(f"<li>{esc(r)}</li>" for r in gate.get("reasons", []))
    left = card(
        "Latest promotion decision",
        "Challenger vs production, re-scored on the same test split.",
        verdict + f'<ul class="reasons">{reasons}</ul><dl class="kv kv--tight">'
        f"<div><dt>Challenger</dt><dd><code>{esc(gate.get('challenger_ref', ''))}</code></dd></div>"
        f"<div><dt>Production</dt><dd><code>{esc(gate.get('production_ref', ''))}</code></dd></div></dl>",
        "Source: results/gate.json"
        + (" (the challenger was rejected: see below)" if challenger else ""),
        cls="gate-card",
    )
    keys = ["pr_auc", "roc_auc", "business_cost", "brier", "ece", "recall", "threshold"]
    prod = gate.get("production") or {}
    rows = [
        [f"<code>{k}</code>", f4(gate["challenger"].get(k)), f4(prod.get(k))]
        for k in keys
        if k in gate["challenger"]
    ]
    right = card(
        "Metrics",
        "Each model at its own cutoff.",
        table(["Metric", "Challenger", "Production"], rows, "Gate metrics"),
        "",
    )
    body = f'<div class="sb-grid-2">{left}{right}</div>'
    if gate.get("checks"):
        body += card(
            "Promotion checks",
            "Every check must pass; all are evaluated and recorded.",
            checks_table(gate["checks"], "Promotion checks"),
            "Configured in configs/registry/default.yaml → gate.",
        )
    else:
        body += note(
            "This decision predates the promotion policy: it checked PR-AUC and expected cost only. The next training run records every check."
        )
    return section(
        "gate",
        "01",
        "Promotion gate",
        'Nothing ships without the <span class="sb-serif">gate</span>.',
        "Performance floors and PR-AUC vs production, calibration (ECE, Brier), expected cost, and "
        "operational checks (API contract, valid and deterministic probabilities, latency, size).",
        body,
    )


def governance_section(history: list | None, demo: dict | None) -> str:
    parts = []
    if demo:
        cands = [e for e in demo["timeline"] if e["event"] != "rollback"]
        names = []
        for e in cands:
            for c in e["checks"]:
                if c["name"] not in names:
                    names.append(c["name"])
        rows = []
        for n in names:
            row = [esc(n)]
            for e in cands:
                c = next((c for c in e["checks"] if c["name"] == n), None)
                row.append(
                    "-"
                    if c is None
                    else (
                        '<span class="check-skip">skip</span>'
                        if c["skipped"]
                        else (
                            '<span class="check-pass">pass</span>'
                            if c["passed"]
                            else '<span class="check-fail">FAIL</span>'
                        )
                    )
                )
            rows.append(row)
        matrix = table(["Check"] + [esc(e["candidate"]) for e in cands], rows, "Check matrix")
        items = []
        for e in demo["timeline"]:
            if e["event"] == "rollback":
                items.append(
                    f'<li><div class="t-head">{pill("rollback")}<span class="t-title">@production v{esc(e["from_version"])} → v{esc(e["to_version"])}</span></div><p>{esc(e["reasons"][0])}</p></li>'
                )
            else:
                c = e["challenger"]
                items.append(
                    f'<li><div class="t-head">{pill(e["event"])}<span class="t-title">{esc(e["candidate"])} (v{esc(e["version"])})</span></div>'
                    f'<p class="t-meta">PR-AUC {c["pr_auc"]:.4f} · Brier {c["brier"]:.4f} · ECE {c["ece"]:.4f} · cost {c["business_cost"]:.4f} · production {esc(_ver(e["production_before"]))} → {esc(_ver(e["production_after"]))}</p>'
                    f"<p>{esc(e['description'])}</p></li>"
                )
        parts.append(
            '<div class="split">'
            + card(
                "Simulated lifecycle",
                esc(demo["note"]),
                f'<ol class="timeline">{"".join(items)}</ol>',
                provenance(demo, "results/governance/demo.json"),
            )
            + card(
                "Which checks stopped which candidate",
                "Same test split for every candidate.",
                matrix,
                f"Final aliases: {esc(', '.join(f'@{k} → v{v}' for k, v in demo['final_aliases'].items()))}.",
            )
            + "</div>"
        )
    if history:
        rows = []
        for e in reversed(history):
            ch = e.get("challenger") or {}
            rows.append(
                [
                    str(e["id"]),
                    esc(str(e.get("ts", ""))[:10]),
                    pill(e["event"]),
                    esc(ch.get("family") or ""),
                    "-" if "pr_auc" not in ch else f4(ch["pr_auc"]),
                    "-" if "business_cost" not in ch else f4(ch["business_cost"]),
                    esc(", ".join(e.get("failed_checks", [])) or "-"),
                    f'<a href="{esc(e["run_url"])}">run</a>' if e.get("run_url") else "-",
                ]
            )
        parts.append(
            card(
                "Real promotion history",
                "Append-only; written by the train workflow after every gate.",
                table(
                    ["#", "Date", "Event", "Family", "PR-AUC", "Cost", "Failed checks", "Run"],
                    rows,
                    "Promotion history",
                ),
                "Source: results/governance/history.json",
            )
        )
    if not parts:
        return ""
    rollback = (
        "<p><strong>Rollback.</strong> Registry mode: <code>python -m credit_default.registry.governance "
        "rollback --mode=registry</code> moves <code>@production</code> back to "
        "<code>@previous-production</code>, which every promotion sets. Served bundle (what Render "
        "runs): <code>rollback --mode=git</code> restores <code>models/bundled/</code> from the "
        "previous bundle commit and records the event; pushing it redeploys.</p>"
    )
    return section(
        "governance",
        "02",
        "Champion / challenger",
        'A failed challenger keeps the <span class="sb-serif">champion</span>.',
        "A passing and two failing challengers and a rollback, run on the real splits in a "
        "throw-away MLflow registry, next to the real promotion history.",
        f'<div class="stack">{"".join(parts)}</div><div class="prose sb-reveal" style="margin-top:1.25rem">{rollback}</div>',
    )


def _ver(v) -> str:
    return "none" if v is None else f"v{v}"


def challenger_section(ch: dict | None) -> str:
    if not ch:
        return ""
    t = ch["test"]
    return section(
        "challenger",
        "03",
        "Rejected challenger",
        'The latest challenger was <span class="sb-serif">rejected</span>.',
        "Its report is kept in <code>results/challenger/</code>; production was not changed.",
        card(
            f"{esc(ch['champion'])} challenger",
            "",
            f'<dl class="kv"><div><dt>Test</dt><dd>ROC-AUC {t["roc_auc"]:.4f} · PR-AUC {t["pr_auc"]:.4f} · Brier {t["brier"]:.4f} · cost {t["business_cost"]:.4f}</dd></div></dl>',
            provenance(ch, "results/challenger/metrics.json"),
        ),
    )


# ---------------------------------------------------------------------------- predict
FIELD_GROUPS = [
    (
        "Profile",
        [
            ("LIMIT_BAL", "Credit limit (NT$)", 1000, 5000000),
            ("SEX", "Sex", None, None),
            ("EDUCATION", "Education", None, None),
            ("MARRIAGE", "Marital status", None, None),
            ("AGE", "Age", 18, 100),
        ],
    ),
    (
        "Repayment status (months late; Sep → Apr)",
        [(f"PAY_{i}", f"PAY_{i}", None, None) for i in (0, 2, 3, 4, 5, 6)],
    ),
    (
        "Bill amounts (NT$; Sep → Apr)",
        [(f"BILL_AMT{i}", f"BILL_AMT{i}", -1000000, 2000000) for i in range(1, 7)],
    ),
    (
        "Payments (NT$; Sep → Apr)",
        [(f"PAY_AMT{i}", f"PAY_AMT{i}", 0, 2000000) for i in range(1, 7)],
    ),
]
CHOICES = {
    "SEX": [(1, "1 male"), (2, "2 female")],
    "EDUCATION": [
        (1, "1 graduate school"),
        (2, "2 university"),
        (3, "3 high school"),
        (4, "4 others"),
        (0, "0 unlabelled"),
        (5, "5 unlabelled"),
        (6, "6 unlabelled"),
    ],
    "MARRIAGE": [(1, "1 married"), (2, "2 single"), (3, "3 others"), (0, "0 unlabelled")],
    "PAY": [(-2, "−2 no use"), (-1, "−1 paid in full"), (0, "0 revolving")]
    + [(k, f"{k} month{'s' if k > 1 else ''} late") for k in range(1, 9)],
}


def example_results(m: dict | None) -> dict:
    """Swagger examples and a few real test applicants, scored and explained at build time by
    the committed bundle (labelled as precomputed on the page)."""
    try:
        import pandas as pd

        from credit_default.config import load_config
        from credit_default.data.schema import FEATURE_COLUMNS
        from credit_default.data.split import make_splits
        from credit_default.models.bundle import ModelBundle
        from credit_default.serve.schemas import EXAMPLE, LOW_RISK_EXAMPLE
    except Exception as exc:  # the site still builds without the package
        print("examples skipped:", exc)
        return {"examples": []}
    bundle_path = ROOT / "models" / "bundled" / "model.joblib"
    if not bundle_path.exists():
        return {"examples": []}
    bundle = ModelBundle.load(bundle_path)
    rows = [
        ("Late payer (Swagger example)", EXAMPLE),
        ("Pays in full (Swagger example)", LOW_RISK_EXAMPLE),
    ]
    try:
        cfg = load_config()
        data = ROOT / cfg.data.raw_path
        if data.exists():
            test = make_splits(pd.read_csv(data), cfg.train.splits, cfg.seed).test
            X = test[FEATURE_COLUMNS].reset_index(drop=True)
            p = bundle.predict_proba(X)
            order = p.argsort()
            picks = [
                ("Test applicant, high score", int(order[-3])),
                ("Test applicant, near the cutoff", int(abs(p - bundle.threshold).argmin())),
                ("Test applicant, low score", int(order[len(order) // 4])),
            ]
            rows += [(name, {k: int(v) for k, v in X.iloc[i].items()}) for name, i in picks]
    except Exception as exc:
        print("test applicants skipped:", exc)
    out = []
    version = f"local:{str(bundle.metadata.get('git_commit', 'unknown'))[:10]}"
    for name, feats in rows:
        r = bundle.explain(pd.DataFrame([feats]), top_k=8)[0]
        out.append(
            {
                "name": name,
                "features": feats,
                "result": {
                    "probability_of_default": r["probability"],
                    "base_value": r["base_value"],
                    "raw_score": r.get("raw_score"),
                    "uncalibrated_probability": r.get("uncalibrated_probability"),
                    "other_features_contribution": r.get("other_features_contribution", 0.0),
                    "contributions": r["contributions"],
                    "model_version": version,
                    "calibration": getattr(bundle, "calibration", "isotonic"),
                    "explanation_space": bundle.explanation_space,
                },
            }
        )
    return {"examples": out, "model_version": version}


def predict_section(m: dict | None, examples: dict) -> str:
    policy = (m or {}).get("test", {}).get("threshold", 0.5)
    groups = []
    ex0 = examples["examples"][0]["features"] if examples.get("examples") else {}
    for title, fields in FIELD_GROUPS:
        cells = []
        for name, label, lo, hi in fields:
            val = ex0.get(name, 0)
            key = "PAY" if name.startswith("PAY_") and not name.startswith("PAY_AMT") else name
            if key in CHOICES:
                opts = "".join(
                    f'<option value="{v}"{" selected" if v == val else ""}>{esc(t)}</option>'
                    for v, t in CHOICES[key]
                )
                ctrl = f'<select class="sb-select" id="f-{name}" name="{name}" data-feature required>{opts}</select>'
            else:
                ctrl = f'<input class="sb-input" id="f-{name}" name="{name}" data-feature type="number" inputmode="numeric" step="1" min="{lo}" max="{hi}" value="{val}" required>'
            cells.append(
                f'<div class="sb-field"><label for="f-{name}">{esc(label)}</label>{ctrl}</div>'
            )
        groups.append(
            f'<fieldset class="pf-group"><legend class="sb-label">{esc(title)}</legend><div class="pf-fields">{"".join(cells)}</div></fieldset>'
        )
    ex_buttons = "".join(
        f'<button type="button" class="sb-btn sb-btn--sm" data-example="{i}">{esc(e["name"])}</button>'
        for i, e in enumerate(examples.get("examples", []))
    )
    form = f"""
<form class="sb-card sb-reveal" id="pf" data-api="{API_BASE}" data-policy="{policy:.6f}" novalidate>
  <div class="stack" style="margin-top:0">
    <div><p class="sb-label">Examples</p><div class="btn-row">{ex_buttons}</div></div>
    {"".join(groups)}
    <fieldset class="pf-group"><legend class="sb-label">Decision cutoff</legend>
      <div class="range-row"><label for="pf-threshold" class="sb-small">Decline when P(default) ≥ <output id="pf-threshold-out" for="pf-threshold">{policy:.3f}</output></label>
        <input type="range" id="pf-threshold" min="0.005" max="0.995" step="0.005" value="{policy:.3f}"></div>
      <div class="two-col">
        <div class="sb-field"><label class="sb-small" for="pf-fn">Missed-default cost</label><input class="sb-input" id="pf-fn" type="number" min="0.01" max="1000" step="any" value="5"></div>
        <div class="sb-field"><label class="sb-small" for="pf-fp">False-alarm cost</label><input class="sb-input" id="pf-fp" type="number" min="0.01" max="1000" step="any" value="1"></div>
      </div>
      <div class="btn-row"><button type="button" class="sb-btn sb-btn--sm" id="pf-use-costs">Cutoff from these costs</button>
        <button type="button" class="sb-btn sb-btn--sm" id="pf-use-policy">Model's policy cutoff ({policy:.3f})</button></div>
    </fieldset>
    <div class="pf-actions">
      <button type="submit" class="sb-btn sb-btn--primary" id="pf-submit">Score with the live API</button>
      <button type="button" class="sb-btn" id="pf-precomputed" hidden>Show the precomputed result</button>
    </div>
    <p class="pf-status is-idle" id="pf-status" role="status" aria-live="polite">The API runs on Render's free tier and sleeps when idle: the first request wakes it (30–60 s), with progress shown here.</p>
  </div>
</form>"""
    result = f"""
<div class="sb-card sb-reveal pf-result" id="pf-result" hidden aria-live="polite">
  <p class="sb-label">Probability of default next month</p>
  <p class="pf-prob" id="pf-prob">-</p>
  <div class="pf-meter" aria-hidden="true"><span id="pf-meter"></span><i id="pf-cut-mark"></i></div>
  <dl class="kv kv--tight">
    <div><dt>Decision</dt><dd><span id="pf-decision" class="sb-pill">-</span></dd></div>
    <div><dt>Cutoff</dt><dd id="pf-cutoff">-</dd></div>
    <div><dt>Model</dt><dd id="pf-version">-</dd></div>
    <div><dt>Source</dt><dd id="pf-source">-</dd></div>
  </dl>
  <p class="sb-label" style="margin-top:1.25rem">How the score was reached (SHAP, top 8)</p>
  {legend([("s-up", "increases risk"), ("s-down", "decreases risk"), ("s-muted", "base value")])}
  <div class="chart-host" id="pf-waterfall"></div>
  <p class="pf-story" id="pf-story"></p>
</div>"""
    notes = (
        "<p>The form posts to <code>/explain</code> on the live FastAPI service with the cutoff "
        "as <code>?threshold=</code>. The API validates every field (out-of-domain values return "
        "422), so the page never guesses. SEX, MARRIAGE and AGE are accepted but not used by the "
        "model; they only feed the fairness reports.</p>"
        "<p>Examples load real test-split applicants (public UCI data) and the Swagger examples. "
        "Their precomputed results were produced at build time by the committed bundle that the "
        "API serves, and are labelled as such.</p>"
    )
    return section(
        "predict",
        "01",
        "Predict",
        'Live <span class="sb-serif">prediction</span>.',
        "Educational decision support on a public 2005 dataset; not a lending decision.",
        embed("examples-data", examples)
        + f'<div class="pf-grid">{form}<div class="stack" style="margin-top:0">{result}<div class="prose sb-reveal">{notes}</div></div></div>',
    )


# ---------------------------------------------------------------------------- page shell
def page(
    title: str, desc: str, current: str, main: str, built: str, scripts: tuple[str, ...] = ()
) -> str:
    links = "".join(
        f'<a class="sb-nav__link nav-page" href="{href}"{" aria-current=page" if href == current else ""}>{esc(label)}</a>'
        for href, label in PAGES
    )
    sub = "".join(
        f'<a href="{href}"{" aria-current=page" if href == current else ""}>{esc(label)}</a>'
        for href, label in PAGES
    )
    js = "".join(f'<script src="{s}" defer></script>' for s in scripts)
    return f"""<!doctype html>
<html lang="en" class="no-js" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>(function(){{try{{var t=localStorage.getItem('theme');document.documentElement.dataset.theme=(t==='light'||t==='dark')?t:'dark';}}catch(e){{document.documentElement.dataset.theme='dark';}}document.documentElement.classList.remove('no-js');}})();</script>
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta name="color-scheme" content="dark light">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet">
<link rel="stylesheet" href="sb-kit.css">
<link rel="stylesheet" href="site.css">
<script src="sb-kit.js" defer></script>
<script src="charts.js" defer></script>
<script src="site.js" defer></script>
{js}
</head>
<body class="sb-grain">
<div class="sb-backdrop" aria-hidden="true"></div>
<a class="skip-link" href="#main">Skip to content</a>
<header class="sb-nav">
  <div class="sb-container sb-nav__inner">
    <a class="sb-brand" href="{PORTFOLIO}"><span class="sb-brand__mark">SB</span>
      <span><span class="brand-name">Srinivasan B. </span><span class="sb-brand__sub"><span class="brand-slash">/ </span>Credit-risk MLOps</span></span></a>
    <nav class="sb-nav__links" aria-label="Pages">
      {links}
      <a class="sb-nav__link" href="{REPO}">Code</a>
      <button class="sb-icon-btn theme-btn" type="button" data-theme-toggle aria-label="Switch to light theme" aria-pressed="true">{THEME_ICONS}</button>
    </nav>
  </div>
</header>
<nav class="subnav" aria-label="Dashboard pages"><div class="sb-container subnav__inner">{sub}</div></nav>
<main id="main">
{main}
</main>
<footer class="sb-footer">
  <div class="sb-container sb-footer__inner">
    <div>
      <p class="foot-name">© 2026 Srinivasan Balasubramanian</p>
      <p>Static pages built by <code>scripts/build_site.py</code> in the <code>pages</code> workflow from the committed results, then published to GitHub Pages. Built {built} UTC. Educational decision-support demo, not a lending system.</p>
    </div>
    <p class="foot-links"><a href="{PORTFOLIO}">Portfolio</a><a href="{REPO}">Repository</a><a href="{API_DOCS}">API docs</a></p>
  </div>
</footer>
</body>
</html>
"""


def number_sections(html_: str) -> str:
    counter = iter(range(1, 100))
    return re.sub(
        r'<span class="sb-kicker__index">\d\d</span>',
        lambda _: f'<span class="sb-kicker__index">{next(counter):02d}</span>',
        html_,
    )


ASSETS = (
    "sb-kit.css",
    "sb-kit.js",
    "site.css",
    "site.js",
    "charts.js",
    "evaluation.js",
    "predict.js",
    "monitoring.js",
)


def main() -> int:
    SITE.mkdir(exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    results, reports = ROOT / "results", ROOT / "reports"
    for name in ASSETS:
        shutil.copy2(WEB / name, SITE / name)

    metrics = load_json(results / "metrics.json")
    real = bool(metrics) and metrics["data"].get("source") == "uci"
    m = metrics if real else None
    gate = load_json(results / "gate.json")
    load = parse_loadtest(results / "loadtest.md")
    ev = load_json(results / "evaluation.json")
    decision = load_json(results / "decision.json")
    fair = load_json(results / "fairness.json")
    sc = load_json(results / "monitoring" / "scenarios.json")
    dq = load_json(results / "data_quality.json")
    history = load_json(results / "governance" / "history.json")
    demo = load_json(results / "governance" / "demo.json")
    bench = load_json(results / "benchmarks" / "benchmark.json")
    challenger = load_json(results / "challenger" / "metrics.json")
    figures = results / "figures"
    if figures.exists():
        shutil.copytree(figures, SITE / "results" / "figures", dirs_exist_ok=True)
    if (results / "predictions.json").exists():
        shutil.copy2(results / "predictions.json", SITE / "data" / "predictions.json")
    shots_dir = SITE / "screenshots"
    order = list(SHOT_TITLES)
    shots = sorted(
        shots_dir.glob("*.png") if shots_dir.exists() else [],
        key=lambda f: (order.index(f.stem) if f.stem in order else len(order), f.name),
    )
    built = f"{datetime.now(UTC):%Y-%m-%d %H:%M}"

    # Overview
    serving = load_section(load)
    bench_html = benchmark_section(bench)
    if bench_html:
        serving = section(
            "serving",
            "02",
            "Serving",
            'FastAPI, measured under <span class="sb-serif">load</span>.',
            "Benchmarks of the served bundle and the API (results/benchmarks), plus the Locust load "
            "test recorded by the train workflow (results/loadtest.md).",
            bench_html
            + (
                f'<div class="stack">{card("Locust load test", "", md_table(load["table"], "Load test", best_if=lambda r: "All" in r[0]), "Source: results/loadtest.md")}</div>'
                if load
                else ""
            ),
        )
    overview = [
        hero(m, gate, load),
        overview_status(m, gate, history, sc, dq, bench, load),
        serving,
    ]
    (SITE / "index.html").write_text(
        page(
            "Credit-risk MLOps · overview",
            "Credit default prediction with cost-based decisions: champion model, monitoring status, governance and serving benchmarks, rebuilt by CI.",
            "index.html",
            number_sections("\n".join(p for p in overview if p)),
            built,
        )
    )
    # Predict
    examples = example_results(m)
    (SITE / "predict.html").write_text(
        page(
            "Credit-risk MLOps · predict",
            "Score a credit-card applicant with the live API: calibrated probability, decision at a chosen cutoff and SHAP explanation.",
            "predict.html",
            page_hero(
                "Live API",
                'Score an <span class="sb-serif">applicant</span>.',
                "The form calls the deployed FastAPI service on Render. Nothing is computed or faked in the browser.",
            )
            + number_sections(predict_section(m, examples)),
            built,
            ("predict.js",),
        )
    )
    # Evaluation
    if m:
        evaluation = [
            decision_tool_section(m, decision),
            sensitivity_section(decision),
            models_compare_section(m, ev, figures),
            calibration_section(m, figures),
            explain_section(ev, figures),
            fairness_section(fair),
            errors_section(ev),
        ]
        body = number_sections("\n".join(p for p in evaluation if p))
    else:
        body = note("No real-data training run has been published yet.")
    (SITE / "evaluation.html").write_text(
        page(
            "Credit-risk MLOps · evaluation",
            "Model comparison, calibration, an interactive cost and cutoff slider on real held-out predictions, fairness and error analysis.",
            "evaluation.html",
            page_hero(
                "Model evaluation",
                'Evaluated on <span class="sb-serif">held-out</span> data.',
                "Every chart and table reads the committed results of the training run that produced the served model.",
            )
            + body,
            built,
            ("evaluation.js",),
        )
    )
    # Monitoring
    monitoring = [
        scenarios_section(sc),
        quality_section(dq),
        drift_section(reports),
        retrain_history_section(history),
    ]
    (SITE / "monitoring.html").write_text(
        page(
            "Credit-risk MLOps · monitoring",
            "Data quality, seven simulated drift scenarios with the monitoring policy's decisions, Evidently reports and retraining history.",
            "monitoring.html",
            page_hero(
                "Monitoring",
                'Watching for <span class="sb-serif">drift</span>.',
                "Feature drift, prediction drift, data quality and labelled performance, each with an explicit action.",
            )
            + number_sections("\n".join(p for p in monitoring if p)),
            built,
            ("monitoring.js",),
        )
    )
    # Registry
    registry = [
        gate_checks_section(gate, challenger),
        governance_section(history, demo),
        challenger_section(challenger),
        registry_section(results / "mlflow_summary.md", shots),
    ]
    (SITE / "registry.html").write_text(
        page(
            "Credit-risk MLOps · registry",
            "Promotion policy checks, champion/challenger history, the rollback simulation and the MLflow registry.",
            "registry.html",
            page_hero(
                "Model registry",
                'Promoted by <span class="sb-serif">policy</span>.',
                "MLflow aliases, a multi-check promotion gate, an append-only decision history and rollback.",
            )
            + number_sections("\n".join(p for p in registry if p)),
            built,
        )
    )
    (SITE / ".nojekyll").write_text("")
    print("site written to", SITE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
