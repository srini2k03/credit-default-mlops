"""Deterministic stratified train / calibration / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.data.schema import FEATURE_COLUMNS, TARGET


@dataclass
class Splits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @staticmethod
    def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[FEATURE_COLUMNS], df[TARGET]


def make_splits(df: pd.DataFrame, fractions, seed: int) -> Splits:
    f = dict(fractions)
    total = sum(f.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    rest, test = train_test_split(df, test_size=f["test"], stratify=df[TARGET], random_state=seed)
    rest_frac = 1 - f["test"]
    train, holdout = train_test_split(
        rest,
        test_size=(f["calibration"] + f["validation"]) / rest_frac,
        stratify=rest[TARGET],
        random_state=seed,
    )
    calibration, validation = train_test_split(
        holdout,
        test_size=f["validation"] / (f["calibration"] + f["validation"]),
        stratify=holdout[TARGET],
        random_state=seed,
    )
    return Splits(*(d.reset_index(drop=True) for d in (train, calibration, validation, test)))
