"""Latency, throughput and memory benchmarks for the served model and the API.

Measures, on real test-split rows and the committed production bundle:
  1. model-loading strategies: compressed joblib (what ships), uncompressed joblib, MLflow
     pyfunc from a local registry, each in a fresh process (load time and resident memory)
  2. in-process scoring: sklearn pipeline vs the NumPy fast path, single row and batches of
     10 / 100 / 1,000 (p50 / p95 / p99, rows per second), after asserting both paths return
     identical probabilities on the whole test split
  3. SHAP explanation latency (single row)
  4. HTTP: a uvicorn worker under 1 / 4 / 16 / 32 concurrent clients sending real rows
     (single /predict and 100-row /predict_batch), with p50 / p95 / p99, throughput, errors and
     server memory, with the fast path on and off

Numbers depend on the machine; the report records CPU count, Python and platform, and the
recorded run is a GitHub-hosted runner (4 vCPU), not the Render free tier.

Usage: python -m credit_default.perf.benchmark [--out=results/benchmarks] [--quick]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import FEATURE_COLUMNS
from credit_default.data.split import make_splits
from credit_default.models.bundle import ModelBundle
from credit_default.pipeline.steps import git_commit, now_iso, write_json
from credit_default.serve.fast import FastScorer, build_fast_scorer

log = logging.getLogger(__name__)


def _pcts(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "n": int(len(a)),
    }


def _time(fn, n: int, warmup: int = 20) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


# ---------------------------------------------------------------------------- loading
_LOAD_SNIPPET = """
import json, os, sys, time, resource
import psutil
proc = psutil.Process()
base = proc.memory_info().rss
strategy, path = sys.argv[1], sys.argv[2]
t = time.perf_counter()
if strategy == "pyfunc":
    import mlflow.pyfunc
    mlflow.set_tracking_uri(sys.argv[3])
    model = mlflow.pyfunc.load_model(path)
    predict = lambda X: model.predict(X)["probability"].to_numpy()
else:
    from credit_default.models.bundle import ModelBundle
    model = ModelBundle.load(path)
    predict = model.predict_proba
load_s = time.perf_counter() - t
import pandas as pd
X = pd.read_json(sys.argv[4], orient="records")
t = time.perf_counter()
p = predict(X)
first_s = time.perf_counter() - t
print(json.dumps({"load_s": load_s, "first_predict_s": first_s,
                  "rss_mb": proc.memory_info().rss / 1e6, "rss_delta_mb": (proc.memory_info().rss - base) / 1e6,
                  "checksum": float(sum(p))}))
"""


def loading_strategies(bundle_path: Path, rows: pd.DataFrame, repeats: int) -> list[dict]:
    import mlflow

    from credit_default.registry.mlflow_registry import log_bundle

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = ModelBundle.load(bundle_path)
        uncompressed = tmp / "uncompressed.joblib"
        import joblib

        joblib.dump(raw, uncompressed, compress=0)
        uri = f"sqlite:///{tmp}/mlflow.db"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("bench")
        with mlflow.start_run():
            info = log_bundle(raw)
        rows_path = tmp / "rows.json"
        rows.to_json(rows_path, orient="records")
        strategies = [
            ("joblib, compressed (shipped)", "bundle", str(bundle_path)),
            ("joblib, uncompressed", "bundle", str(uncompressed)),
            ("MLflow pyfunc (local registry)", "pyfunc", info.model_uri),
        ]
        sizes = {
            "joblib, compressed (shipped)": bundle_path.stat().st_size / 1e6,
            "joblib, uncompressed": uncompressed.stat().st_size / 1e6,
        }
        env = {**os.environ, "MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "1"}
        for label, kind, path in strategies:
            runs = []
            for _ in range(repeats):
                out = subprocess.run(
                    [sys.executable, "-c", _LOAD_SNIPPET, kind, path, uri, str(rows_path)],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=True,
                )
                runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
            results.append(
                {
                    "strategy": label,
                    "file_mb": sizes.get(label),
                    "load_s_median": float(np.median([r["load_s"] for r in runs])),
                    "first_predict_s_median": float(
                        np.median([r["first_predict_s"] for r in runs])
                    ),
                    "rss_mb_median": float(np.median([r["rss_mb"] for r in runs])),
                    "checksum": runs[0]["checksum"],
                    "repeats": repeats,
                }
            )
    sums = {r["checksum"] for r in results}
    for r in results:
        r["same_predictions"] = len(sums) == 1
    return results


# ---------------------------------------------------------------------------- in-process
def in_process(bundle: ModelBundle, X: pd.DataFrame, n_single: int) -> dict:
    fast = build_fast_scorer(bundle, X)
    A = X.to_numpy(dtype=np.float64)
    identical = fast is not None and bool(
        np.array_equal(fast.predict_proba(A), bundle.predict_proba(X))
    )
    rows = X.to_dict("records")
    out = {
        "identical_on_test_split": identical,
        "n_test_rows": int(len(X)),
        "single": {},
        "batch": [],
    }
    i = iter(range(10**9))

    def one_pipeline():
        bundle.predict_proba(pd.DataFrame([rows[next(i) % len(rows)]]))

    def one_fast():
        fast.predict_proba(FastScorer.rows_to_array([rows[next(i) % len(rows)]]))

    out["single"]["pipeline"] = _pcts(_time(one_pipeline, n_single))
    if fast is not None:
        out["single"]["fast"] = _pcts(_time(one_fast, n_single))
    for size in (10, 100, 1000):
        chunk = X.iloc[:size]
        arr = A[:size]
        p = _pcts(_time(lambda c=chunk: bundle.predict_proba(c), max(20, 2000 // size), 3))
        row = {"batch": size, "pipeline": {**p, "rows_per_s": size / (p["p50_ms"] / 1000)}}
        if fast is not None:
            f = _pcts(_time(lambda a=arr: fast.predict_proba(a), max(20, 2000 // size), 3))
            row["fast"] = {**f, "rows_per_s": size / (f["p50_ms"] / 1000)}
        out["batch"].append(row)
    j = iter(range(10**9))
    out["explain_single"] = _pcts(
        _time(
            lambda: bundle.explain(X.iloc[[next(j) % len(X)]], top_k=5), max(50, n_single // 10), 3
        )
    )
    return out


# ---------------------------------------------------------------------------- HTTP
async def _load(url: str, payloads: list, concurrency: int, duration_s: float) -> dict:
    import httpx

    lat, errors = [], 0
    stop = time.perf_counter() + duration_s
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=concurrency)) as c:

        async def worker(k: int):
            nonlocal errors
            n = k
            while time.perf_counter() < stop:
                body = payloads[n % len(payloads)]
                n += concurrency
                t = time.perf_counter()
                try:
                    r = await c.post(url, json=body)
                    ok = r.status_code == 200
                except Exception:
                    ok = False
                lat.append((time.perf_counter() - t) * 1000)
                errors += not ok

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(k) for k in range(concurrency)))
        elapsed = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        **_pcts(lat),
        "requests_per_s": len(lat) / elapsed,
        "errors": int(errors),
    }


def http_benchmark(bundle_path: Path, X: pd.DataFrame, duration_s: float, fast: bool) -> dict:
    import httpx
    import psutil

    port = 8123 if fast else 8124
    env = {
        **os.environ,
        "MODEL_SOURCE": "local",
        "MODEL_BUNDLE_PATH": str(bundle_path),
        "RATE_LIMIT_PER_MINUTE": "0",
        "FAST_SCORING": "true" if fast else "false",
        "LOG_LEVEL": "WARNING",
        "PREDICTION_LOG_PATH": str(Path(tempfile.mkdtemp()) / "pred.sqlite"),
    }
    env.pop("MLFLOW_TRACKING_URI", None)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "credit_default.serve.app:app",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.perf_counter()
        for _ in range(300):
            try:
                if httpx.get(f"{base}/ready", timeout=1).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        startup_s = time.perf_counter() - t0
        proc = psutil.Process(server.pid)
        rss_idle = proc.memory_info().rss / 1e6
        rows = X.to_dict("records")
        singles = [{k: int(v) for k, v in r.items()} for r in rows[:500]]
        batches = [{"applications": singles[i : i + 100]} for i in range(0, 500, 100)]
        single = [
            asyncio.run(_load(f"{base}/predict", singles, c, duration_s)) for c in (1, 4, 16, 32)
        ]
        batch = [asyncio.run(_load(f"{base}/predict_batch", batches, 4, duration_s))]
        explain = [asyncio.run(_load(f"{base}/explain?top_k=5", singles, 4, duration_s))]
        rss_after = proc.memory_info().rss / 1e6
    finally:
        server.terminate()
        server.wait(timeout=10)
    return {
        "fast_scoring": fast,
        "startup_to_ready_s": startup_s,
        "server_rss_idle_mb": rss_idle,
        "server_rss_after_load_mb": rss_after,
        "predict": single,
        "predict_batch_100": batch,
        "explain": explain,
    }


def to_markdown(r: dict) -> str:
    env = r["environment"]
    ip = r["in_process"]
    lines = [
        "# Serving benchmark",
        "",
        f"Generated {r['generated_at']} at commit `{r['git_commit'][:10]}` on {env['machine']} "
        f"({env['cpu_count']} CPUs, Python {env['python']}). Model: {r['model']['family']} "
        f"({r['model']['calibration']} calibration), {r['model']['n_rows']:,} real test rows.",
        "",
        f"Fast path returns identical probabilities on all {ip['n_test_rows']:,} test rows: "
        f"**{ip['identical_on_test_split']}**.",
        "",
        "## In-process scoring (ms)",
        "",
        "| Path | Rows | p50 | p95 | p99 | Rows / s (p50) |",
        "|---|---|---|---|---|---|",
    ]
    for path, s in ip["single"].items():
        lines.append(
            f"| {path} | 1 | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | "
            f"{1000 / s['p50_ms']:,.0f} |"
        )
    for b in ip["batch"]:
        for path in ("pipeline", "fast"):
            if path in b:
                s = b[path]
                lines.append(
                    f"| {path} | {b['batch']:,} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | "
                    f"{s['p99_ms']:.2f} | {s['rows_per_s']:,.0f} |"
                )
    e = ip["explain_single"]
    lines += [
        "",
        f"SHAP explanation, one row: p50 {e['p50_ms']:.1f} ms, p95 {e['p95_ms']:.1f} ms.",
        "",
        "## Model-loading strategies (fresh process each, median)",
        "",
        "| Strategy | File MB | Load s | First predict s | Process RSS MB | Same predictions |",
        "|---|---|---|---|---|---|",
    ]
    for s in r["loading"]:
        mb = "-" if s["file_mb"] is None else f"{s['file_mb']:.2f}"
        lines.append(
            f"| {s['strategy']} | {mb} | {s['load_s_median']:.3f} | {s['first_predict_s_median']:.3f} "
            f"| {s['rss_mb_median']:.0f} | {s['same_predictions']} |"
        )
    for h in r["http"]:
        lines += [
            "",
            f"## HTTP, one uvicorn worker, fast path {'on' if h['fast_scoring'] else 'off'}",
            "",
            f"Startup to ready {h['startup_to_ready_s']:.1f} s; server RSS {h['server_rss_idle_mb']:.0f} MB "
            f"idle, {h['server_rss_after_load_mb']:.0f} MB after the load.",
            "",
            "| Endpoint | Concurrency | Requests | Errors | Req / s | p50 ms | p95 ms | p99 ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, rows in (
            ("/predict", h["predict"]),
            ("/predict_batch (100 rows)", h["predict_batch_100"]),
            ("/explain", h["explain"]),
        ):
            for x in rows:
                lines.append(
                    f"| {name} | {x['concurrency']} | {x['n']:,} | {x['errors']} | "
                    f"{x['requests_per_s']:.1f} | {x['p50_ms']:.1f} | {x['p95_ms']:.1f} | {x['p99_ms']:.1f} |"
                )
    lines += [
        "",
        "Clients run on the same machine as the server, so they compete for CPU; numbers are "
        "relative comparisons for this environment, not a capacity claim for the free hosting tier.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from credit_default.validation.schema import validate

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    opts = dict(f.lstrip("-").split("=", 1) for f in flags if "=" in f)
    quick = "--quick" in flags
    cfg = load_config(overrides)
    bundle_path = Path(opts.get("bundle", REPO_ROOT / cfg.paths.bundle_dir / "model.joblib"))
    bundle = ModelBundle.load(bundle_path)
    df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    X = make_splits(df, cfg.train.splits, cfg.seed).test[FEATURE_COLUMNS].reset_index(drop=True)
    result = {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "environment": {
            "machine": os.environ.get("BENCH_MACHINE", platform.node() or "unknown"),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "family": bundle.family,
            "calibration": getattr(bundle, "calibration", "isotonic"),
            "bundle_git_commit": bundle.metadata.get("git_commit"),
            "n_rows": int(len(X)),
        },
    }
    log.info("in-process scoring")
    result["in_process"] = in_process(bundle, X, 300 if quick else 2000)
    log.info("loading strategies")
    result["loading"] = loading_strategies(bundle_path, X.head(100), 1 if quick else 5)
    log.info("HTTP")
    duration = 3.0 if quick else 15.0
    result["http"] = [http_benchmark(bundle_path, X, duration, fast) for fast in (True, False)]
    out = Path(opts.get("out", REPO_ROOT / cfg.paths.results_dir / "benchmarks"))
    out.mkdir(parents=True, exist_ok=True)
    write_json(result, out / "benchmark.json")
    (out / "benchmark.md").write_text(to_markdown(result))
    log.info("wrote %s", out / "benchmark.md")
    return 0 if result["in_process"]["identical_on_test_split"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
