"""Screenshot the MLflow UI (experiment runs and the registered model) with Playwright.

Usage: python scripts/mlflow_screenshots.py http://localhost:5000 site/screenshots

MLflow 3.x opens ``#/experiments/<id>`` on a GenAI "Overview" page, so the runs table is
requested explicitly (``/runs?workflowType=machine_learning``), the "Model training" view is
selected on the registry page, and the assistant side panel is closed if it is open. Pages are
loaded with ``wait_until="load"`` plus a wait for the table: the UI polls in the background, so
"networkidle" may never be reached (pages run 45365907708 timed out that way).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright


def api(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/2.0/mlflow/{path}", timeout=10) as r:
        return json.loads(r.read())


def _tidy(page) -> None:
    """Model-training view, no assistant panel."""
    for name in ("Model training",):
        toggle = page.get_by_text(name, exact=True)
        if toggle.count():
            try:
                toggle.first.click(timeout=3000)
                page.wait_for_timeout(1500)
            except PlaywrightTimeout:
                pass
    close = page.get_by_role("button", name="Close", exact=True)
    if close.count():
        try:
            close.first.click(timeout=2000)
            page.wait_for_timeout(800)
        except PlaywrightTimeout:
            pass


def main() -> int:
    base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    exp = api(base, "experiments/get-by-name?experiment_name=credit-default")["experiment"]
    pages = {
        "mlflow-runs.png": (
            f"{base}/#/experiments/{exp['experiment_id']}/runs?workflowType=machine_learning",
            "Run Name",
        ),
        "mlflow-registry.png": (f"{base}/#/models/credit-default", "Version"),
    }
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (url, marker) in pages.items():
            # A fresh page per URL: the URLs differ only in the #fragment.
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            try:
                page.goto(url, wait_until="load", timeout=60_000)
                page.get_by_text(marker, exact=True).first.wait_for(timeout=30_000)
                page.wait_for_timeout(2000)
                _tidy(page)
                if "Overview" in page.url or "genai" in page.url:
                    raise RuntimeError(f"landed on {page.url}, not the requested page")
                page.screenshot(path=str(out / name), full_page=False)
                print("saved", out / name, "from", page.url)
            except Exception as exc:  # one failed screenshot must not hide the other
                failures += 1
                print("FAILED", name, exc)
            finally:
                page.close()
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
