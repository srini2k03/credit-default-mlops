"""Stepped load for the kind/HPA test: each step holds a user count for STEP_SECONDS.

  locust -f locust/step_load.py --headless --host http://localhost:8080 \
         --csv results/k8s/raw --csv-full-history

Environment: LOCUST_STEPS (default "10,25,50,100,150"), LOCUST_STEP_SECONDS (default 120),
LOCUST_SPAWN_RATE (users per second, default 10). The user behaviour is ApiUser from
locustfile.py, unchanged.
"""

from __future__ import annotations

import os

from locustfile import ApiUser  # noqa: F401  (Locust picks up the user class from this module)

from locust import LoadTestShape

STEPS = [int(s) for s in os.environ.get("LOCUST_STEPS", "10,25,50,100,150").split(",") if s]
STEP_SECONDS = int(os.environ.get("LOCUST_STEP_SECONDS", "120"))
SPAWN_RATE = float(os.environ.get("LOCUST_SPAWN_RATE", "10"))


class StepLoad(LoadTestShape):
    def tick(self):
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEPS):
            return None
        return STEPS[step], SPAWN_RATE
