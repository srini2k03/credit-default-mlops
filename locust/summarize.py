"""Turn Locust's ``*_stats.csv`` into the markdown table committed as results/loadtest.md.

Usage: python locust/summarize.py STATS_CSV [--users N] [--duration 60s] [--where "GitHub Actions ..."]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats_csv")
    ap.add_argument("--users", default="20")
    ap.add_argument("--duration", default="60s")
    ap.add_argument(
        "--where", default="GitHub Actions ubuntu-latest runner, API in Docker on the same host"
    )
    args = ap.parse_args()

    with open(args.stats_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("no rows in stats file", file=sys.stderr)
        return 1

    def ms(v):
        return "-" if v in ("", "N/A", None) else f"{float(v):.0f}"

    commit = os.environ.get("GITHUB_SHA", "local")[:10]
    run_url = ""
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
    out = [
        "# Load test",
        "",
        f"- Generated: {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC, commit `{commit}`"
        + (f", [workflow run]({run_url})" if run_url else ""),
        f"- Setup: Locust headless, {args.users} users, {args.duration}; {args.where}",
        "- Latencies in milliseconds, measured client-side",
        "",
        "| Endpoint | Requests | Failures | RPS | p50 | p95 | p99 | Max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = "**All**" if r["Name"] == "Aggregated" else f"`{r['Type']} {r['Name']}`"
        out.append(
            f"| {name} | {r['Request Count']} | {r['Failure Count']} | {float(r['Requests/s']):.1f} "
            f"| {ms(r['50%'])} | {ms(r['95%'])} | {ms(r['99%'])} | {ms(r['Max Response Time'])} |"
        )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
