"""Deterministic synthetic fixture with the exact UCI schema.

The generator draws a latent risk score per customer and derives payment status, balances,
payments and the default label from it, so models have real signal to learn. The default rate is
calibrated to about 22% (the UCI rate). Fixture metrics are meaningless for the real problem and
are never published as results.

Usage:  python -m credit_default.data.synthetic [data.n_rows=6000] [data.seed=7]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import (
    BILL_AMT,
    ID_COLUMN,
    PAY_AMT,
    PAY_STATUS,
    RAW_COLUMNS,
    TARGET,
)

log = logging.getLogger(__name__)
TARGET_DEFAULT_RATE = 0.22


def generate(n_rows: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows
    risk = rng.normal(0.0, 1.0, n)  # latent credit risk, higher = riskier

    limit = np.exp(rng.normal(11.6 - 0.35 * risk, 0.7, n))
    limit = np.clip(np.round(limit / 10_000) * 10_000, 10_000, 1_000_000)
    sex = rng.choice([1, 2], size=n, p=[0.40, 0.60])
    education = rng.choice(
        [0, 1, 2, 3, 4, 5, 6], size=n, p=[0.001, 0.35, 0.468, 0.164, 0.004, 0.011, 0.002]
    )
    marriage = rng.choice([0, 1, 2, 3], size=n, p=[0.002, 0.455, 0.532, 0.011])
    age = np.clip(np.round(rng.gamma(9.0, 3.9, n) + 1), 21, 79)

    # Payment status: persistent behaviour driven by risk, months are correlated.
    pay = np.zeros((n, 6), dtype=int)
    base = risk + rng.normal(0, 0.6, n)
    for m in range(6):
        noise = rng.normal(0, 0.7, n)
        score = base + noise - 0.05 * m
        status = np.select(
            [score < -1.6, score < -0.6, score < 0.9, score < 1.5, score < 2.1],
            [-2, -1, 0, 1, 2],
            default=3,
        )
        bump = (score > 2.6) & (rng.random(n) < 0.5)
        status = np.where(bump, status + rng.integers(1, 6, n), status)
        pay[:, m] = np.clip(status, -2, 8)

    # Balances: utilisation rises with risk; payments shrink with risk.
    util = np.clip(rng.beta(2, 3, n) + 0.12 * risk, -0.05, 1.2)
    bills = np.zeros((n, 6))
    pays = np.zeros((n, 6))
    for m in range(6):
        drift = 1 + rng.normal(0, 0.08, n) - 0.02 * m
        bills[:, m] = limit * util * drift
        neg = rng.random(n) < 0.02
        bills[neg, m] = -rng.uniform(0, 5_000, neg.sum())
        zero = pay[:, m] == -2
        bills[zero, m] = 0
        frac = np.clip(rng.beta(1.2, 6, n) - 0.04 * risk, 0, 1)
        pays[:, m] = np.where(pay[:, m] >= 1, frac * 0.3, frac) * np.maximum(bills[:, m], 0)
        pays[rng.random(n) < 0.18, m] = 0

    logit = (
        1.1 * risk
        + 0.55 * np.clip(pay[:, 0], -2, 4)
        + 0.25 * (pay[:, 1:3] > 0).sum(axis=1)
        - 0.25 * np.log(limit / 100_000)
        + 0.004 * (age - 35)
        + rng.normal(0, 0.5, n)
    )
    # Pick the intercept so the default rate matches the UCI rate exactly in expectation.
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if rate < TARGET_DEFAULT_RATE else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + lo)))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: np.arange(1, n + 1),
            "LIMIT_BAL": limit,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **{c: pay[:, i] for i, c in enumerate(PAY_STATUS)},
            **{c: np.round(bills[:, i]) for i, c in enumerate(BILL_AMT)},
            **{c: np.round(pays[:, i]) for i, c in enumerate(PAY_AMT)},
            TARGET: y,
        }
    )
    return df[RAW_COLUMNS].astype("int64")


def write_fixture(cfg) -> Path:
    path = REPO_ROOT / cfg.data.raw_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(cfg.data.n_rows, cfg.data.seed)
    df.to_csv(path, index=False, lineterminator="\n")
    log.info(
        "wrote synthetic fixture %s (%d rows, default rate %.3f)", path, len(df), df[TARGET].mean()
    )
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, _ = split_cli(sys.argv[1:] if argv is None else argv)
    write_fixture(load_config(["data=synthetic", *overrides]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
