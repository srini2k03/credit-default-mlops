"""Drift statistics: PSI, two-sample KS, chi-square (with Cramer's V), proportion z-test.

Why several tests, and why the decision does not rest on p-values alone: with a few thousand
rows, KS and chi-square flag shifts far too small to matter (statistical significance is not a
business problem), while PSI is a magnitude with conventional bands (< 0.1 stable, 0.1-0.2
moderate, >= 0.2 major) but no sampling distribution. The monitoring policy therefore flags a
feature when its PSI is major, or when it is moderate *and* the test is significant.

Limitations: PSI depends on the binning (deciles of the reference here); KS ignores
missing values and is conservative on discrete data such as the PAY_* status codes, which is
why those are treated as categorical; univariate tests miss changes in joint structure (the
concept-drift scenario shows a case no feature test can see).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_EPS = 1e-4


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return np.clip(counts / total if total else counts, PSI_EPS, None)


def psi_numeric(ref, cur, bins: int = 10) -> float:
    ref = pd.Series(ref).dropna().to_numpy(dtype=float)
    cur = pd.Series(cur).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if len(np.unique(ref)) <= 2 * bins:  # discrete values (e.g. isotonic scores): exact shares
        return psi_categorical(ref, cur)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]
    r = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1)
    c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1)
    pr, pc = _shares(r.astype(float)), _shares(c.astype(float))
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def psi_fixed(ref, cur, edges=None) -> float:
    """PSI on fixed bins (default: ten equal-width probability bins), the usual choice for
    model scores because the bins do not move with the reference sample."""
    edges = np.linspace(0, 1, 11) if edges is None else np.asarray(edges)
    r = np.histogram(np.asarray(ref, dtype=float), edges)[0].astype(float)
    c = np.histogram(np.asarray(cur, dtype=float), edges)[0].astype(float)
    pr, pc = _shares(r), _shares(c)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def psi_categorical(ref, cur) -> float:
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    r = ref.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    c = cur.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    pr, pc = _shares(r), _shares(c)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def ks(ref, cur) -> tuple[float, float]:
    ref = pd.Series(ref).dropna().to_numpy(dtype=float)
    cur = pd.Series(cur).dropna().to_numpy(dtype=float)
    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def chi2(ref, cur) -> tuple[float, float, float]:
    """Chi-square test of homogeneity; returns (statistic, p-value, Cramer's V)."""
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    table = np.vstack(
        [
            ref.value_counts().reindex(cats, fill_value=0).to_numpy(),
            cur.value_counts().reindex(cats, fill_value=0).to_numpy(),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0, 0.0
    stat, p, _, _ = stats.chi2_contingency(table)
    v = np.sqrt(stat / (table.sum() * (min(table.shape) - 1)))
    return float(stat), float(p), float(v)


def proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (pooled); returns (z, two-sided p)."""
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def psi_band(value: float, moderate: float, major: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "major" if value >= major else "moderate" if value >= moderate else "stable"
