"""Screenshot the MLflow UI (experiment runs and the registered model) with Playwright.

Usage: python scripts/mlflow_screenshots.py http://localhost:5000 site/screenshots
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def api(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/2.0/mlflow/{path}", timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    exp = api(base, "experiments/get-by-name?experiment_name=credit-default")["experiment"]
    pages = {
        "mlflow-runs.png": f"{base}/#/experiments/{exp['experiment_id']}",
        "mlflow-registry.png": f"{base}/#/models/credit-default",
    }
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, url in pages.items():
            # A fresh page per URL: the URLs differ only in the #fragment, and a same-document
            # goto on a reused page never reaches "networkidle" (Page.goto timed out in CI).
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(out / name), full_page=False)
            page.close()
            print("saved", out / name)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
