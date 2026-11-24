"""Browser checks for the built site (run after scripts/build_site.py; used by pages.yml).

For every page at 1280 px and 390 px, in the dark and light theme (and once with reduced
motion): no console or page errors, no failed same-origin requests, no horizontal overflow, and
the page's main interaction works:
  evaluation  the slider loads the held-out predictions and, at the deployed cutoff and the
              configured costs, shows the test cost from results/metrics.json
  predict     the form posts to the API and renders the result (the API is replaced by a test
              double here: CI must not depend on the free-tier service being awake)
  monitoring  the drift explorer draws its chart
Usage: python scripts/check_site.py [--shots=DIR]
"""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGES = ["index.html", "predict.html", "evaluation.html", "monitoring.html", "registry.html"]
VIEWPORTS = {"desktop": (1280, 900), "phone": (390, 844)}


def serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def scroll_through(page) -> None:
    page.evaluate(
        """async () => { for (let y = 0; y < document.body.scrollHeight; y += 500) {
             window.scrollTo(0, y); await new Promise(r => setTimeout(r, 160)); }
           window.scrollTo(0, 0); }"""
    )
    page.wait_for_timeout(2800)  # sb-kit.js reveals anything still hidden after 2.5 s


def fake_api(page, examples: list) -> None:
    """Test double for the Render API (health + explain), built from a precomputed example."""
    result = examples[0]["result"] if examples else None

    def handle(route):
        url = route.request.url
        if url.endswith("/health"):
            return route.fulfill(status=200, json={"status": "ok", "model_loaded": True})
        if "/explain" in url and result:
            thr = float(url.split("threshold=")[1].split("&")[0]) if "threshold=" in url else 0.5
            body = dict(result, request_id="check-site", threshold=thr)
            return route.fulfill(status=200, json=body)
        return route.fulfill(status=404, json={"detail": "not mocked"})

    page.route("https://credit-default-api-fkqg.onrender.com/**", handle)


def check_page(
    browser,
    base: str,
    name: str,
    vp: str,
    theme: str,
    reduced: bool,
    shots: Path | None,
    metrics: dict | None,
) -> list[str]:
    w, h = VIEWPORTS[vp]
    ctx = browser.new_context(
        viewport={"width": w, "height": h}, reduced_motion="reduce" if reduced else "no-preference"
    )
    ctx.add_init_script(f"try{{localStorage.setItem('theme','{theme}')}}catch(e){{}}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on(
        "console",
        lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None,
    )
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on(
        "requestfailed",
        lambda r: (
            errors.append(f"failed request {r.url}")
            if r.url.startswith(base) and "net::ERR_ABORTED" not in (r.failure or "")
            else None
        ),
    )
    if name == "predict.html":
        html = (SITE / name).read_text()
        raw = html.split('id="examples-data">')[1].split("</script>")[0]
        fake_api(page, json.loads(raw).get("examples", []))
    page.goto(f"{base}/{name}", wait_until="load")
    page.wait_for_timeout(400)
    tag = f"{name} {vp} {theme}{' reduced' if reduced else ''}"
    problems = []
    if page.evaluate("document.documentElement.dataset.theme") != theme:
        problems.append(f"{tag}: theme not applied")
    overflow = page.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
    if overflow > 1:
        problems.append(f"{tag}: horizontal overflow {overflow}px")
    if shots is not None and not reduced:
        scroll_through(page)
        page.screenshot(
            path=str(shots / f"{name.removesuffix('.html')}-{vp}-{theme}.png"), full_page=True
        )
    if name == "evaluation.html" and metrics:
        page.wait_for_selector("#ev-tool.is-ready", timeout=15_000)
        shown = page.inner_text('[data-ev="cost"]').strip()
        expected = f"{metrics['test']['business_cost']:.4f}"
        if shown != expected:
            problems.append(f"{tag}: slider cost {shown} != results/metrics.json {expected}")
        if page.locator("#chart-roc svg").count() != 1:
            problems.append(f"{tag}: ROC chart missing")
        page.fill("#ev-fn", "10")
        page.wait_for_timeout(100)
        if page.inner_text('[data-ev="ratio"]').strip() != "10":
            problems.append(f"{tag}: cost input does not update the analysis")
    if name == "monitoring.html" and page.locator("#chart-psi").count():
        page.wait_for_selector("#chart-psi svg", timeout=10_000)
        page.select_option("#sc-select", "missing_values")
        if "block" not in page.inner_text("#sc-detail"):
            problems.append(f"{tag}: scenario explorer did not update")
    if name == "predict.html" and vp == "desktop" and theme == "dark" and not reduced:
        page.click("#pf-submit")
        page.wait_for_selector("#pf-result:not([hidden])", timeout=20_000)
        if "%" not in page.inner_text("#pf-prob") or page.locator("#pf-waterfall svg").count() != 1:
            problems.append(f"{tag}: prediction result not rendered")
    problems += [f"{tag}: {e}" for e in errors]
    ctx.close()
    return problems


def main() -> int:
    shots = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--shots=")), None)
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    srv, base = serve()
    problems: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name in PAGES:
                if not (SITE / name).exists():
                    problems.append(f"{name} missing")
                    continue
                for vp in VIEWPORTS:
                    for theme in ("dark", "light"):
                        problems += check_page(
                            browser, base, name, vp, theme, False, shots, metrics
                        )
                problems += check_page(browser, base, name, "phone", "dark", True, None, metrics)
            browser.close()
    finally:
        srv.shutdown()
    for pr in problems:
        print("PROBLEM", pr)
    print(f"checked {len(PAGES)} pages: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
