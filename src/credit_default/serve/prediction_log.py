"""Append-only prediction log in SQLite (inputs, score, decision, model version).

It feeds monitoring: ``read_log`` returns the logged inputs as a frame that drift checks can
compare with the training reference. On Render's free tier the disk is ephemeral, so the log
lives only as long as the instance; docker-compose mounts it on a volume.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_source TEXT,
    model_version TEXT,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    threshold REAL NOT NULL,
    features TEXT NOT NULL
)
"""


class PredictionLogger:
    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute(_SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def log(self, rows: list[dict]) -> None:
        if not self.enabled or not rows:
            return
        now = time.time()
        values = [
            (
                now,
                r["request_id"],
                r["endpoint"],
                r["model_source"],
                r["model_version"],
                r["probability"],
                int(r["prediction"]),
                r["threshold"],
                json.dumps(r["features"]),
            )
            for r in rows
        ]
        with self._lock, self._connect() as con:
            con.executemany(
                "INSERT INTO predictions (ts, request_id, endpoint, model_source, model_version, probability,"
                " prediction, threshold, features) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def count(self) -> int:
        if not self.enabled or not self.path.exists():
            return 0
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def read_log(path: str | Path) -> pd.DataFrame:
    """Logged requests as one row per prediction: feature columns + probability + prediction."""
    with sqlite3.connect(path) as con:
        raw = pd.read_sql_query("SELECT * FROM predictions ORDER BY id", con)
    if raw.empty:
        return raw
    feats = pd.DataFrame([json.loads(f) for f in raw["features"]])
    return pd.concat([raw.drop(columns=["features"]), feats], axis=1)
