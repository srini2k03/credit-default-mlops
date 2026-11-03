# Load test

- Generated: 2026-11-03 13:42 UTC, commit `d2d766943e`, [workflow run](https://github.com/srini2k03/credit-default-mlops/actions/runs/41645854230)
- Setup: Locust headless, 20 users, 60s; GitHub Actions ubuntu-latest runner, API in Docker on the same host
- Latencies in milliseconds, measured client-side

| Endpoint | Requests | Failures | RPS | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|
| `POST /explain` | 727 | 0 | 12.3 | 70 | 130 | 160 | 217 |
| `GET /health` | 358 | 0 | 6.1 | 22 | 87 | 110 | 136 |
| `POST /predict` | 3757 | 0 | 63.6 | 88 | 170 | 190 | 290 |
| `POST /predict_batch (20)` | 701 | 0 | 11.9 | 100 | 180 | 190 | 285 |
| **All** | 5543 | 0 | 93.9 | 80 | 170 | 190 | 290 |
