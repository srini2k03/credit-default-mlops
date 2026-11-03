"""Feature engineering shared by training and serving (one code path, no train/serve skew)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.data.schema import BILL_AMT, CATEGORICAL, FEATURE_COLUMNS, PAY_AMT, PAY_STATUS

ENGINEERED_COLUMNS = [
    "util_1",  # latest balance / limit
    "util_mean",  # mean 6-month utilisation
    "util_trend",  # (balance month 1 - month 6) / limit
    "pay_ratio_1",  # latest payment / previous balance
    "pay_ratio_mean",  # total payments / total positive balances
    "max_delay",  # worst payment status over 6 months
    "n_delayed",  # months with any delay
    "n_paid_duly",  # months paid in full (-1)
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean undocumented category codes and add ratio/behaviour features."""
    out = df[FEATURE_COLUMNS].copy().astype("float64")
    # Undocumented codes are folded into "others" (EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3).
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].between(1, 4), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].between(1, 3), 3)

    limit = out["LIMIT_BAL"].clip(lower=1.0)
    bills = out[BILL_AMT].to_numpy()
    pays = out[PAY_AMT].to_numpy()
    status = out[PAY_STATUS].to_numpy()

    out["util_1"] = bills[:, 0] / limit
    out["util_mean"] = bills.mean(axis=1) / limit
    out["util_trend"] = (bills[:, 0] - bills[:, 5]) / limit
    # PAY_AMT1 pays (part of) BILL_AMT2, the previous statement.
    out["pay_ratio_1"] = np.clip(pays[:, 0] / np.maximum(bills[:, 1], 1.0), 0, 10)
    out["pay_ratio_mean"] = np.clip(
        pays.sum(axis=1) / np.maximum(np.clip(bills, 0, None).sum(axis=1), 1.0), 0, 10
    )
    out["max_delay"] = status.max(axis=1)
    out["n_delayed"] = (status > 0).sum(axis=1)
    out["n_paid_duly"] = (status == -1).sum(axis=1)
    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless sklearn wrapper around :func:`engineer` so it lives inside the model pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer(pd.DataFrame(X))

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS + ENGINEERED_COLUMNS, dtype=object)


def build_preprocessor(family: str):
    """Model-family-specific preprocessing on top of the engineered frame.

    Trees take the engineered frame as-is (category codes are ordinal-safe for trees);
    logistic regression gets one-hot categoricals and standardised numerics.
    """
    if family in {"lightgbm", "xgboost"}:
        return "passthrough"
    numeric = [c for c in FEATURE_COLUMNS + ENGINEERED_COLUMNS if c not in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")
