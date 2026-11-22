"""Summarise the kind/HPA load test: Locust CSVs + replica samples -> results/k8s/.

Usage:
  python scripts/k8s/loadtest_report.py --locust-prefix /tmp/k8s/locust \
      --replicas /tmp/k8s/replicas.csv --out results/k8s \
      --steps 10,25,50,100,150 --step-seconds 120 [--context "GitHub Actions, kind"]

Writes loadtest_stats.csv (per endpoint, whole run), loadtest_history.csv (aggregated time
series), replicas.csv, loadtest.png and summary.md. Per-step figures use the step's samples after
a 20 s settling period: mean requests/s, the median of Locust's rolling-window p50 and p95, the
failure share, and the highest ready-replica count seen during the step.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SETTLE_S = 20


def load_history(prefix: str) -> pd.DataFrame:
    h = pd.read_csv(f"{prefix}_stats_history.csv")
    h = h[h["Name"] == "Aggregated"].copy()
    for col in ["50%", "95%", "99%"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h = h[h["User Count"] > 0]
    h["elapsed_s"] = h["Timestamp"] - h["Timestamp"].min()
    out = h.rename(
        columns={
            "Timestamp": "timestamp",
            "User Count": "users",
            "Requests/s": "rps",
            "Failures/s": "failures_per_s",
            "50%": "p50_ms",
            "95%": "p95_ms",
            "99%": "p99_ms",
            "Total Request Count": "total_requests",
            "Total Failure Count": "total_failures",
        }
    )
    cols = [
        "timestamp",
        "elapsed_s",
        "users",
        "rps",
        "failures_per_s",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "total_requests",
        "total_failures",
    ]
    return out[cols].reset_index(drop=True)


def load_replicas(path: str, t0: float) -> pd.DataFrame:
    r = pd.read_csv(path)
    for col in r.columns:
        r[col] = pd.to_numeric(r[col], errors="coerce")
    r["elapsed_s"] = r["timestamp"] - t0
    return r


def step_table(hist: pd.DataFrame, reps: pd.DataFrame, steps: list[int], step_s: int):
    rows = []
    for i, users in enumerate(steps):
        lo, hi = i * step_s, (i + 1) * step_s
        settle = min(SETTLE_S, step_s / 4)
        h = hist[(hist["elapsed_s"] >= lo + settle) & (hist["elapsed_s"] < hi)]
        r = reps[(reps["elapsed_s"] >= lo) & (reps["elapsed_s"] < hi)]
        if h.empty:
            continue
        reqs = h["rps"].sum()
        rows.append(
            {
                "users": users,
                "rps": h["rps"].mean(),
                "p50_ms": h["p50_ms"].median(),
                "p95_ms": h["p95_ms"].median(),
                "failure_pct": 100 * h["failures_per_s"].sum() / reqs if reqs else 0.0,
                "max_ready_replicas": int(r["ready_pods"].max()) if not r.empty else None,
                "max_cpu_pct": r["cpu_utilization_percent"].max() if not r.empty else None,
            }
        )
    return pd.DataFrame(rows)


def plot(hist: pd.DataFrame, reps: pd.DataFrame, steps, step_s, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    m = hist["elapsed_s"] / 60
    axes[0].plot(m, hist["p50_ms"], label="p50", color="#2a6fdb", linewidth=1.5)
    axes[0].plot(m, hist["p95_ms"], label="p95", color="#d9480f", linewidth=1.5)
    axes[0].set_ylabel("latency (ms)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].plot(m, hist["rps"], color="#2b8a3e", linewidth=1.5, label="requests/s")
    if hist["failures_per_s"].max() > 0:
        axes[1].plot(m, hist["failures_per_s"], color="#c92a2a", linewidth=1, label="failures/s")
    axes[1].set_ylabel("requests/s")
    axes[1].legend(loc="upper left", frameon=False)
    rm = reps["elapsed_s"] / 60
    axes[2].step(rm, reps["ready_pods"], where="post", color="#5f3dc4", label="ready pods")
    axes[2].step(
        rm,
        reps["hpa_desired_replicas"],
        where="post",
        color="#862e9c",
        linestyle="--",
        label="HPA desired",
    )
    axes[2].set_ylabel("replicas")
    axes[2].set_xlabel("minutes since the load started")
    users = axes[2].twinx()
    users.plot(m, hist["users"], color="#868e96", linewidth=1, label="Locust users")
    users.set_ylabel("users", color="#868e96")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = users.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(alpha=0.3)
        for i in range(1, len(steps) + 1):
            ax.axvline(i * step_s / 60, color="#adb5bd", linewidth=0.6, linestyle=":")
    fig.suptitle("kind + HPA load test: latency, throughput and replicas")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(v, spec=".0f"):
    return "n/a" if v is None or pd.isna(v) else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locust-prefix", required=True)
    ap.add_argument("--replicas", required=True)
    ap.add_argument("--out", default="results/k8s")
    ap.add_argument("--steps", default="10,25,50,100,150")
    ap.add_argument("--step-seconds", type=int, default=120)
    ap.add_argument("--context", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in a.steps.split(",") if s]

    hist = load_history(a.locust_prefix)
    t0 = float(hist["timestamp"].min())
    reps = load_replicas(a.replicas, t0)
    table = step_table(hist, reps, steps, a.step_seconds)

    shutil.copy(f"{a.locust_prefix}_stats.csv", out / "loadtest_stats.csv")
    hist.to_csv(out / "loadtest_history.csv", index=False)
    reps.to_csv(out / "replicas.csv", index=False)
    plot(hist, reps, steps, a.step_seconds, out / "loadtest.png")

    stats = pd.read_csv(f"{a.locust_prefix}_stats.csv")
    load_end = len(steps) * a.step_seconds
    during = reps[reps["elapsed_s"] < load_end]
    after = reps[reps["elapsed_s"] >= load_end]
    peak = int(during["ready_pods"].max()) if not during.empty else 0
    start = int(reps["ready_pods"].iloc[0]) if not reps.empty else None
    final = int(after["ready_pods"].iloc[-1]) if not after.empty else None
    back_down = None
    if not after.empty and final is not None:
        low = after[after["ready_pods"] <= final]
        back_down = float(low["elapsed_s"].iloc[0] - load_end) if not low.empty else None

    lines = [
        "# kind + HPA load test",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. {a.context}".rstrip(),
        "",
        f"Stepped Locust load ({', '.join(map(str, steps))} users, {a.step_seconds} s each, the "
        "user mix from `locust/locustfile.py`: /predict, /predict_batch of 20, /explain, /health) "
        "against the Helm chart on a single-node kind cluster. HPA 1-5 replicas at 50% CPU of a "
        "250m request (1 CPU limit per pod), metrics-server at 15 s resolution, scale-down "
        "stabilisation 60 s. Load generator and cluster share one machine, so absolute numbers "
        "are a lower bound on what the service does on dedicated nodes.",
        "",
        "## Per step",
        "",
        "| Users | Requests/s | p50 (ms) | p95 (ms) | Failures | Peak ready pods | Peak CPU (% of request) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {int(r.users)} | {fmt(r.rps, '.1f')} | {fmt(r.p50_ms)} | {fmt(r.p95_ms)} | "
            f"{fmt(r.failure_pct, '.2f')}% | {fmt(r.max_ready_replicas)} | {fmt(r.max_cpu_pct)} |"
        )
    lines += [
        "",
        f"Per-step values use the samples after the first {SETTLE_S} s of each step: mean "
        "requests/s and the median of Locust's rolling-window percentiles.",
        "",
        "## Scaling",
        "",
        f"- Ready pods: {fmt(start)} at the start, peak {peak} under load"
        + (
            f", {final} after the load stopped"
            + (
                f" (reached {fmt(back_down)} s after the last step)"
                if back_down is not None
                else ""
            )
            if final is not None
            else ""
        )
        + ".",
        "- Time series: `replicas.csv` (every 10 s) and `loadtest_history.csv`; chart: `loadtest.png`.",
        "",
        "## Whole run, per endpoint",
        "",
        "| Endpoint | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, s in stats.iterrows():
        name = s["Name"] if s["Name"] != "Aggregated" else "**all**"
        lines.append(
            f"| {name} | {int(s['Request Count'])} | {int(s['Failure Count'])} | "
            f"{fmt(s['50%'])} | {fmt(s['95%'])} | {fmt(s['99%'])} |"
        )
    lines += ["", "![load test](loadtest.png)", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
