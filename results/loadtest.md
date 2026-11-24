# Load test

- Generated: 2026-11-24 10:30 UTC, commit `48a6a54a17`, [workflow run](https://github.com/srini2k03/credit-default-mlops/actions/runs/45371941736)
- Setup: Locust headless, 20 users, 60s; GitHub Actions ubuntu-latest runner, API in Docker on the same host
- Latencies in milliseconds, measured client-side

| Endpoint | Requests | Failures | RPS | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|
| `POST /explain` | 659 | 0 | 11.2 | 110 | 190 | 210 | 226 |
| `GET /health` | 353 | 0 | 6.0 | 32 | 130 | 180 | 207 |
| `POST /predict` | 3169 | 0 | 53.7 | 120 | 210 | 240 | 289 |
| `POST /predict_batch (20)` | 661 | 0 | 11.2 | 140 | 220 | 250 | 303 |
| **All** | 4842 | 0 | 82.1 | 120 | 210 | 240 | 303 |
