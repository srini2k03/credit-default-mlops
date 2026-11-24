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
ASSETS = ("sb-kit.css", "sb-kit.js", "site.css", "site.js")
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
        "drift-triggered retraining. This page is rebuilt by CI from the committed results."
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
      <a class="sb-btn sb-btn--primary" href="{API_DOCS}" target="_blank" rel="noopener">Try the live API {ARROW}</a>
      <a class="sb-btn" href="{REPO}" target="_blank" rel="noopener">{GITHUB} Code</a>
      <a class="sb-btn sb-btn--ghost" href="#drift">Drift reports ↓</a>
    </div>
    <p class="hero-note">Swagger UI on Render's free tier: the first request wakes the service in 30–60 s.</p>
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
        "The UCI table has no timestamps, so production traffic is simulated from the test split: "
        "a <em>control</em> slice left untouched and a <em>drifted</em> slice with documented "
        "shifts. Reference = the training split. The weekly <code>retrain</code> workflow checks "
        "the control slice; a drifted-column share above the threshold triggers retraining."
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


# ---------------------------------------------------------------------------- page
def page(main: str, nav: list[tuple[str, str]], built: str) -> str:
    links = "".join(
        f'<a class="sb-nav__link sb-nav__link--hide-sm" href="#{sid}">{esc(label)}</a>'
        for sid, label in nav
    )
    return f"""<!doctype html>
<html lang="en" class="no-js" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>(function(){{try{{var t=localStorage.getItem('theme');document.documentElement.dataset.theme=(t==='light'||t==='dark')?t:'dark';}}catch(e){{document.documentElement.dataset.theme='dark';}}document.documentElement.classList.remove('no-js');}})();</script>
<title>Credit-risk MLOps · results and monitoring</title>
<meta name="description" content="Credit default prediction with cost-based decisions: model results, quality gate, load test, MLflow registry and Evidently drift reports, rebuilt by CI.">
<meta name="color-scheme" content="dark light">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet">
<link rel="stylesheet" href="sb-kit.css">
<link rel="stylesheet" href="site.css">
<script src="sb-kit.js" defer></script>
<script src="site.js" defer></script>
</head>
<body class="sb-grain">
<div class="sb-backdrop" aria-hidden="true"></div>
<a class="skip-link" href="#main">Skip to content</a>
<header class="sb-nav">
  <div class="sb-container sb-nav__inner">
    <a class="sb-brand" href="{PORTFOLIO}"><span class="sb-brand__mark">SB</span>
      <span><span class="brand-name">Srinivasan B. </span><span class="sb-brand__sub"><span class="brand-slash">/ </span>Credit-risk MLOps</span></span></a>
    <nav class="sb-nav__links" aria-label="Sections">
      {links}
      <a class="sb-nav__link" href="{REPO}">Code</a>
      <button class="sb-icon-btn theme-btn" type="button" data-theme-toggle aria-label="Switch to light theme" aria-pressed="true">{THEME_ICONS}</button>
    </nav>
  </div>
</header>
<main id="main">
{main}
</main>
<footer class="sb-footer">
  <div class="sb-container sb-footer__inner">
    <div>
      <p class="foot-name">© 2026 Srinivasan Balasubramanian</p>
      <p>Static page built by <code>scripts/build_site.py</code> in the <code>pages</code> workflow from the committed results, then published to GitHub Pages. Built {built} UTC.</p>
    </div>
    <p class="foot-links"><a href="{PORTFOLIO}">Portfolio</a><a href="{REPO}">Repository</a><a href="{API_DOCS}">API docs</a></p>
  </div>
</footer>
</body>
</html>
"""


def main() -> int:
    SITE.mkdir(exist_ok=True)
    results, reports = ROOT / "results", ROOT / "reports"
    for name in ASSETS:
        shutil.copy2(WEB / name, SITE / name)

    metrics = load_json(results / "metrics.json")
    real = bool(metrics) and metrics["data"].get("source") == "uci"
    gate = load_json(results / "gate.json")
    load = parse_loadtest(results / "loadtest.md")
    shots_dir = SITE / "screenshots"
    order = list(SHOT_TITLES)
    shots = sorted(
        shots_dir.glob("*.png") if shots_dir.exists() else [],
        key=lambda f: (order.index(f.stem) if f.stem in order else len(order), f.name),
    )

    parts, nav = [hero(metrics if real else None, gate, load)], []
    if real:
        figures = results / "figures"
        if figures.exists():
            shutil.copytree(figures, SITE / "results" / "figures", dirs_exist_ok=True)
        parts += [decision_section(metrics, figures), models_section(metrics, figures)]
        nav += [("decision", "Decision"), ("models", "Models")]
    for sid, label, html_ in (
        ("gate", "Gate", gate_section(gate, results / "gate.md")),
        ("registry", "MLflow", registry_section(results / "mlflow_summary.md", shots)),
        ("load", "Load test", load_section(load)),
        ("drift", "Drift", drift_section(reports)),
    ):
        if html_:
            parts.append(html_)
            nav.append((sid, label))

    # Number the sections in the order they appear (some are skipped when inputs are missing).
    counter = iter(range(1, 100))
    main_html = re.sub(
        r'<span class="sb-kicker__index">\d\d</span>',
        lambda _: f'<span class="sb-kicker__index">{next(counter):02d}</span>',
        "\n".join(parts),
    )
    built = f"{datetime.now(UTC):%Y-%m-%d %H:%M}"
    (SITE / "index.html").write_text(page(main_html, nav, built))
    (SITE / ".nojekyll").write_text("")
    print("site written to", SITE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
