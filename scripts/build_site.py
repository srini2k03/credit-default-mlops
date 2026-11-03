"""Build the static GitHub Pages site into site/.

Inputs (each optional): results/*.md + figures (committed by train.yml), reports/drift_*.html
(Evidently, built in the pages workflow), site/screenshots/*.png (Playwright, pages workflow).
"""

from __future__ import annotations

import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPO = "https://github.com/srini2k03/credit-default-mlops"

CSS = """
:root { --bg:#ffffff; --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --card:#f9fafb; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f172a; --fg:#e5e7eb; --muted:#9ca3af; --line:#334155;
  --accent:#60a5fa; --card:#111827; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
a { color: var(--accent); }
h1 { margin: 0 0 4px; } .sub { color: var(--muted); margin: 0 0 24px; }
section { border-top: 1px solid var(--line); padding-top: 16px; margin-top: 24px; }
table { border-collapse: collapse; display: block; overflow-x: auto; max-width: 100%; font-size: 14px; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: left; white-space: nowrap; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
.note { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
ul.links li { margin: 4px 0; }
"""


def md_file(path: Path, base: str = "") -> str:
    text = path.read_text()
    if base:
        text = text.replace("](figures/", f"]({base}figures/")
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"
    )


def main() -> int:
    SITE.mkdir(exist_ok=True)
    results, reports = ROOT / "results", ROOT / "reports"
    parts = [
        "<h1>credit-default-mlops</h1>",
        f"<p class='sub'>Monitoring and results site, built {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. "
        f"<a href='{REPO}'>Source on GitHub</a>.</p>",
    ]

    real = (results / "metrics.json").exists() and json.loads(
        (results / "metrics.json").read_text()
    )["data"].get("source") == "uci"
    parts.append("<section><h2>Model results (real UCI data)</h2>")
    if real:
        (SITE / "results").mkdir(exist_ok=True)
        if (results / "figures").exists():
            shutil.copytree(results / "figures", SITE / "results" / "figures", dirs_exist_ok=True)
        parts.append(md_file(results / "metrics.md", base="results/"))
    else:
        parts.append(
            "<p class='note'>No real-data training run has been published yet. The train workflow "
            "fills this section after it trains on the UCI data and the quality gate passes.</p>"
        )
    parts.append("</section>")

    for name, title in (
        ("gate.md", "Quality gate"),
        ("loadtest.md", "Load test"),
        ("mlflow_summary.md", "MLflow run summary"),
    ):
        if (results / name).exists():
            parts.append(f"<section><h2>{title}</h2>{md_file(results / name)}</section>")

    shots = sorted((SITE / "screenshots").glob("*.png")) if (SITE / "screenshots").exists() else []
    if shots:
        parts.append("<section><h2>MLflow UI screenshots</h2>")
        parts += [f"<p><img src='screenshots/{s.name}' alt='{s.stem}'></p>" for s in shots]
        parts.append("</section>")

    drift = sorted(reports.glob("drift_*.html")) if reports.exists() else []
    parts.append("<section><h2>Drift monitoring (Evidently)</h2>")
    if drift:
        (SITE / "reports").mkdir(exist_ok=True)
        items = []
        for f in drift:
            shutil.copy2(f, SITE / "reports" / f.name)
            js = f.with_suffix(".json")
            info = json.loads(js.read_text()) if js.exists() else {}
            verdict = ""
            if info:
                verdict = (
                    f" - {info['drifted_columns']}/{info['n_columns']} columns drifted "
                    f"(share {info['drift_share']:.2f}, threshold {info['threshold']:.2f}): "
                    f"<b>{'drift' if info['drift_detected'] else 'no drift'}</b>"
                )
            items.append(
                f"<li><a href='reports/{f.name}'>{f.stem.replace('drift_', '')} slice</a>{verdict}</li>"
            )
        parts.append(
            "<p>The data has no timestamps, so production traffic is simulated from the test split: "
            "a <i>control</i> slice left untouched and a <i>drifted</i> slice with documented shifts "
            "(credit limits x0.7, ages -6 years, PAY_0 one month worse for 35% of rows, balances "
            "x1.25). Reference = training split.</p>"
        )
        summary = reports / "drift_summary.json"
        src = json.loads(summary.read_text()).get("data_source") if summary.exists() else None
        if src != "uci":
            parts.append(
                "<p class='note'><b>Built on the synthetic test fixture, not the real data.</b></p>"
            )
        parts.append("<ul class='links'>" + "".join(items) + "</ul>")
    else:
        parts.append("<p class='note'>No drift reports were built in this run.</p>")
    parts.append("</section>")

    (SITE / "index.html").write_text(page("credit-default-mlops", "\n".join(parts)))
    (SITE / ".nojekyll").write_text("")
    print("site written to", SITE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
