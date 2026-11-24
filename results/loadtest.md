# Load test

- Generated: 2026-11-24 11:21 UTC, commit `1faa0a8f8f`, [workflow run](https://github.com/srini2k03/credit-default-mlops/actions/runs/45379624037)
- Setup: Locust headless, 20 users, 60s; GitHub Actions ubuntu-latest runner, API in Docker on the same host
- Latencies in milliseconds, measured client-side

| Endpoint | Requests | Failures | RPS | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|
| `POST /explain` | 919 | 0 | 15.6 | 53 | 130 | 170 | 234 |
| `GET /health` | 461 | 0 | 7.8 | 8 | 70 | 130 | 176 |
| `POST /predict` | 4712 | 0 | 79.8 | 20 | 110 | 150 | 231 |
| `POST /predict_batch (20)` | 931 | 0 | 15.8 | 23 | 130 | 170 | 246 |
| **All** | 7023 | 0 | 118.9 | 26 | 110 | 160 | 246 |
