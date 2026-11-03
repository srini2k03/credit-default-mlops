"""Build the monitoring slices. The UCI table has no timestamps, so "production" is simulated.

- reference: the training split (what the model learned from)
- control:   an untouched random sample of the test split (should show no drift)
- drifted:   another test-split sample with deliberate, documented population shifts
             (configs/monitor/default.yaml -> drifted_shift):
               LIMIT_BAL x 0.70, AGE - 6 years, PAY_0 one month worse for 35% of rows,
               BILL_AMT1..6 x 1.25
Labels are dropped from control/drifted: production traffic arrives without outcomes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import BILL_AMT, FEATURE_COLUMNS
from credit_default.data.split import make_splits
from credit_default.validation.schema import validate

log = logging.getLogger(__name__)


def apply_shift(df: pd.DataFrame, shift, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["LIMIT_BAL"] = np.clip(
        (out["LIMIT_BAL"] * shift.limit_bal_factor / 10_000).round() * 10_000, 10_000, 1_000_000
    ).astype("int64")
    out["AGE"] = np.clip(out["AGE"] + int(shift.age_shift_years), 21, 79).astype("int64")
    worse = rng.random(len(out)) < shift.pay0_worsen_fraction
    out.loc[worse, "PAY_0"] = np.clip(out.loc[worse, "PAY_0"].clip(lower=0) + 1, -2, 8)
    for c in BILL_AMT:
        out[c] = (out[c] * shift.bill_amt_factor).round().astype("int64")
    return out


def build_slices(cfg, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if df is None:
        df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
    splits = make_splits(df, cfg.train.splits, cfg.seed)
    test = splits.test.sample(frac=1.0, random_state=cfg.monitor.slice_seed).reset_index(drop=True)
    n = min(cfg.monitor.slice_rows, len(test) // 2)
    control = test.iloc[:n][FEATURE_COLUMNS].reset_index(drop=True)
    drifted = apply_shift(
        test.iloc[n : 2 * n][FEATURE_COLUMNS].reset_index(drop=True),
        cfg.monitor.drifted_shift,
        cfg.monitor.slice_seed,
    )
    return {"reference": splits.train[FEATURE_COLUMNS], "control": control, "drifted": drifted}


def write_slices(cfg, slices: dict[str, pd.DataFrame]) -> Path:
    out = REPO_ROOT / cfg.paths.slices_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in slices.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    log.info("wrote slices %s to %s", list(slices), out)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, _ = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    write_slices(cfg, build_slices(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
