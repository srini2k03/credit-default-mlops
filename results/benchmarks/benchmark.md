# Serving benchmark

Generated 2026-11-24T10:32:52Z at commit `fbaff457e7` on GitHub-hosted ubuntu runner (4 CPUs, Python 3.12.16). Model: lightgbm (sigmoid calibration), 6,000 real test rows.

Fast path returns identical probabilities on all 6,000 test rows: **True**.

## In-process scoring (ms)

| Path | Rows | p50 | p95 | p99 | Rows / s (p50) |
|---|---|---|---|---|---|
| pipeline | 1 | 6.69 | 7.29 | 9.57 | 149 |
| fast | 1 | 0.93 | 0.97 | 1.09 | 1,081 |
| pipeline | 10 | 6.31 | 6.45 | 6.82 | 1,585 |
| fast | 10 | 0.98 | 1.01 | 1.08 | 10,249 |
| pipeline | 100 | 7.20 | 7.44 | 7.48 | 13,888 |
| fast | 100 | 1.56 | 1.60 | 1.61 | 63,994 |
| pipeline | 1,000 | 12.89 | 13.11 | 13.16 | 77,596 |
| fast | 1,000 | 7.20 | 9.32 | 9.32 | 138,809 |

SHAP explanation, one row: p50 20.4 ms, p95 21.3 ms.

## Model-loading strategies (fresh process each, median)

| Strategy | File MB | Load s | First predict s | Process RSS MB | Same predictions |
|---|---|---|---|---|---|
| joblib, compressed (shipped) | 0.30 | 1.559 | 0.029 | 287 | True |
| joblib, uncompressed | 0.71 | 1.559 | 0.030 | 286 | True |
| MLflow pyfunc (local registry) | - | 4.340 | 0.023 | 411 | True |

## HTTP, one uvicorn worker, fast path on

Startup to ready 2.5 s; server RSS 358 MB idle, 390 MB after the load.

| Endpoint | Concurrency | Requests | Errors | Req / s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|
| /predict | 1 | 3,711 | 0 | 247.4 | 3.8 | 5.5 | 6.7 |
| /predict | 4 | 4,954 | 0 | 330.3 | 11.8 | 14.8 | 18.2 |
| /predict | 16 | 4,914 | 0 | 327.3 | 48.1 | 53.0 | 64.0 |
| /predict | 32 | 4,468 | 0 | 296.1 | 97.7 | 189.7 | 394.9 |
| /predict_batch (100 rows) | 4 | 2,139 | 0 | 142.6 | 26.8 | 34.1 | 101.4 |
| /explain | 4 | 700 | 0 | 46.6 | 83.9 | 120.9 | 149.7 |

## HTTP, one uvicorn worker, fast path off

Startup to ready 2.6 s; server RSS 358 MB idle, 385 MB after the load.

| Endpoint | Concurrency | Requests | Errors | Req / s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|
| /predict | 1 | 1,466 | 0 | 97.7 | 9.8 | 12.0 | 13.7 |
| /predict | 4 | 1,539 | 0 | 102.6 | 38.1 | 51.2 | 62.1 |
| /predict | 16 | 1,546 | 0 | 102.9 | 152.5 | 178.3 | 242.3 |
| /predict | 32 | 1,491 | 0 | 99.1 | 313.3 | 398.7 | 456.5 |
| /predict_batch (100 rows) | 4 | 1,045 | 0 | 69.6 | 55.7 | 70.4 | 141.5 |
| /explain | 4 | 678 | 0 | 45.1 | 87.2 | 124.6 | 150.4 |

Clients run on the same machine as the server, so they compete for CPU; numbers are relative comparisons for this environment, not a capacity claim for the free hosting tier.
