"""Load test for the API.

Headless (as in CI):
  locust -f locust/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000 \
         --csv artifacts/locust/run --only-summary
  python locust/summarize.py artifacts/locust/run_stats.csv > results/loadtest.md

Payloads are 200 schema-valid synthetic applications (locust/payloads.json): the load test
measures serving latency, not model quality, so it does not need real customer rows.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOADS = json.loads((Path(__file__).parent / "payloads.json").read_text())


class ApiUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random.choice(PAYLOADS), name="/predict")

    @task(2)
    def predict_batch(self):
        self.client.post(
            "/predict_batch",
            json={"applications": random.sample(PAYLOADS, 20)},
            name="/predict_batch (20)",
        )

    @task(2)
    def explain(self):
        self.client.post("/explain?top_k=5", json=random.choice(PAYLOADS), name="/explain")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
