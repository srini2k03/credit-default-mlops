# Performance

Source: `results/benchmarks/benchmark.md` (benchmark run
[45375530277](https://github.com/srini2k03/credit-default-mlops/actions/runs/45375530277),
GitHub-hosted runner, 4 vCPU, Python 3.12) and `results/loadtest.md` (train run 45371941736).
CI-runner numbers are relative comparisons, not a capacity claim for Render's free tier.

## Optimisation: an exact fast path

Profiling the single-row path showed most of the ~6.7 ms in pandas bookkeeping inside feature
engineering, not in the model. `serve/fast.py` engineers the same features on a NumPy array with
the same float64 operations in the same order, calls the same fitted LightGBM model, and applies
the bundle's own fitted calibrator (on the model's decision function, as sklearn does). It is
enabled only if it reproduces `ModelBundle.predict_proba` **exactly** (`np.array_equal`) at
startup; the benchmark and a test assert the same on the full test split / fixture.

| Rows | Pipeline p50 | Fast p50 | Fast rows / s |
|---|---|---|---|
| 1 | 6.69 ms | 0.93 ms | 1,081 |
| 10 | 6.31 ms | 0.98 ms | 10,249 |
| 100 | 7.20 ms | 1.56 ms | 63,994 |
| 1,000 | 12.89 ms | 7.20 ms | 138,809 |

Identical probabilities on all 6,000 test rows: true.

## HTTP (one uvicorn worker, real test rows, clients on the same runner)

| Clients | Fast path off: req/s, p50 / p95 / p99 ms | Fast path on: req/s, p50 / p95 / p99 ms |
|---|---|---|
| 1 | 97.7, 9.8 / 12.0 / 13.7 | 247.4, 3.8 / 5.5 / 6.7 |
| 4 | 102.6, 38.1 / 51.2 / 62.1 | 330.3, 11.8 / 14.8 / 18.2 |
| 16 | 102.9, 152.5 / 178.3 / 242.3 | 327.3, 48.1 / 53.0 / 64.0 |
| 32 | 99.1, 313.3 / 398.7 / 456.5 | 296.1, 97.7 / 189.7 / 394.9 |

One worker saturates at about 100 req/s on the pipeline and 330 req/s on the fast path; beyond
that, extra clients only queue (latency grows linearly, throughput flat), which is the expected
behaviour of a CPU-bound single worker. `/predict_batch` with 100 rows: 69.6 → 142.6 req/s
(about 14,000 rows/s). `/explain` (SHAP per request) stays at about 46 req/s with 4 clients; it is
not on the fast path. No request failed in any run.

Memory: the API process holds 358 MB after startup (model, pandas, scikit-learn, LightGBM,
XGBoost and SHAP imported) and 385-390 MB after the load, inside Render's 512 MB.

## Model-loading strategies (fresh process each, median of 5)

| Strategy | File | Load | Process RSS |
|---|---|---|---|
| joblib, compressed (shipped) | 0.30 MB | 1.56 s | 287 MB |
| joblib, uncompressed | 0.71 MB | 1.56 s | 286 MB |
| MLflow pyfunc from a local registry | - | 4.34 s | 411 MB |

Load time is dominated by importing the libraries, not by reading the 0.3 MB file, so
compression costs nothing; loading through MLflow pyfunc adds ~2.8 s and ~125 MB, which is why
the API loads the bundle directly (it still resolves the registry alias when a registry is
configured). All three give the same predictions.

## Load test (Locust, recorded by `train.yml`)

20 users for 60 s against the Docker image on the runner, mixed traffic (predict, batch of 20,
explain, health): 4,842 requests, 0 failures, aggregate p50 120 ms, p95 210 ms, p99 240 ms.
Locust and the API share the runner's CPUs, and the API runs one worker, so these latencies are
queueing under saturation; the HTTP benchmark above separates single-client latency from
throughput. The previous release's load test reported p95 170 ms on different code and a
different runner image; the two are not directly comparable.
