"""Data-quality report and leakage checks for the canonical table.

The Pandera contract (``credit_default.validation``) is the hard gate: training stops on a
violation. This module is the descriptive layer on top of it: what is in the data, what looks
odd, and whether the split can leak information from test into training.

Outputs ``data_quality.json`` and ``data_quality.md`` (``results/`` for real data, the run's
artifacts directory otherwise).

Usage: python -m credit_default.data.quality [overrides] [--out=results]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import (
    BILL_AMT,
    CATEGORICAL,
    FEATURE_COLUMNS,
    ID_COLUMN,
    PAY_AMT,
    PAY_STATUS,
    TARGET,
)
from credit_default.data.split import make_splits
from credit_default.pipeline.steps import git_commit, now_iso, write_json

log = logging.getLogger(__name__)

# Codes that occur in the published data but are not in the UCI documentation.
UNDOCUMENTED = {"EDUCATION": (0, 5, 6), "MARRIAGE": (0,)}
# Univariate ROC-AUC above this is treated as a leakage warning (no single honest credit
# feature separates defaulters this well on this data; the best one, PAY_0, is ~0.7).
LEAKAGE_AUC_WARN = 0.90
OUTLIER_IQR_K = 3.0  # Tukey "far out" fences: money columns are heavily right-skewed
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL and c not in PAY_STATUS]


def _schema_check(df: pd.DataFrame, cfg) -> dict:
    from credit_default.validation.schema import DataValidationError, validate

    try:
        validate(df, cfg)
        return {"passed": True, "detail": "Pandera contract passed"}
    except DataValidationError as exc:
        return {"passed": False, "detail": str(exc)[:2000]}


def _columns(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "n_unique": int(s.nunique()),
                "role": (
                    "id"
                    if c == ID_COLUMN
                    else "target"
                    if c == TARGET
                    else "categorical"
                    if c in CATEGORICAL
                    else "ordinal"
                    if c in PAY_STATUS
                    else "numeric"
                ),
            }
        )
    return rows


def _duplicates(df: pd.DataFrame) -> dict:
    features = df[FEATURE_COLUMNS]
    dup_mask = features.duplicated(keep=False)
    groups = df[dup_mask].groupby(FEATURE_COLUMNS, dropna=False)[TARGET].nunique()
    return {
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns=[ID_COLUMN]).duplicated().sum()),
        "duplicate_feature_vectors": int(features.duplicated().sum()),
        "rows_in_duplicate_groups": int(dup_mask.sum()),
        "groups_with_conflicting_labels": int((groups > 1).sum()),
        "note": (
            "Duplicates are kept: identical feature vectors are plausible for distinct customers "
            "(e.g. unused cards with all-zero balances), and IDs are unique."
        ),
    }


def _codes(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col, codes in UNDOCUMENTED.items():
        for code in codes:
            mask = df[col] == code
            rows.append(
                {
                    "column": col,
                    "code": code,
                    "n": int(mask.sum()),
                    "default_rate": float(df.loc[mask, TARGET].mean()) if mask.any() else None,
                    "handling": f"folded into 'others' ({4 if col == 'EDUCATION' else 3})",
                }
            )
    status = df[PAY_STATUS].to_numpy().ravel()
    for code, label in ((-2, "no consumption"), (0, "revolving credit / minimum paid")):
        rows.append(
            {
                "column": "PAY_*",
                "code": code,
                "n": int((status == code).sum()),
                "default_rate": None,
                "handling": f"kept as an ordinal level ({label}; not in the original paper)",
            }
        )
    return rows


def _outliers(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in NUMERIC:
        s = df[c].astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append(
            {
                "column": c,
                "lower_fence": float(lo),
                "upper_fence": float(hi),
                "n_outside": n,
                "share_outside": n / len(s),
            }
        )
    return rows


def _domain_flags(df: pd.DataFrame) -> list[dict]:
    bills, pays = df[BILL_AMT].to_numpy(), df[PAY_AMT].to_numpy()
    limit = df["LIMIT_BAL"].to_numpy()[:, None]
    flags = {
        "negative bill (credit balance) in any month": (bills < 0).any(axis=1),
        "bill above credit limit in any month": (bills > limit).any(axis=1),
        "all six bills and payments zero (inactive card)": (bills == 0).all(axis=1)
        & (pays == 0).all(axis=1),
        "payment larger than the previous bill (PAY_AMT1 > BILL_AMT2 > 0)": (
            (pays[:, 0] > bills[:, 1]) & (bills[:, 1] > 0)
        ),
    }
    y = df[TARGET].to_numpy()
    return [
        {
            "check": name,
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "default_rate": float(y[mask].mean()) if mask.any() else None,
        }
        for name, mask in flags.items()
    ]


def _distributions(df: pd.DataFrame) -> dict:
    numeric = {}
    for c in [*NUMERIC, *PAY_STATUS]:
        s = df[c].astype(float)
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        numeric[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p01": float(q.iloc[0]),
            "p05": float(q.iloc[1]),
            "p25": float(q.iloc[2]),
            "median": float(q.iloc[3]),
            "p75": float(q.iloc[4]),
            "p95": float(q.iloc[5]),
            "p99": float(q.iloc[6]),
            "max": float(s.max()),
            "skew": float(s.skew()),
        }
    categorical = {}
    for c in [*CATEGORICAL, "PAY_0"]:
        g = df.groupby(c)[TARGET].agg(["size", "mean"])
        categorical[c] = [
            {"value": int(v), "n": int(r["size"]), "default_rate": float(r["mean"])}
            for v, r in g.iterrows()
        ]
    return {"numeric": numeric, "categorical": categorical}


def _univariate_auc(df: pd.DataFrame) -> list[dict]:
    y = df[TARGET].to_numpy()
    rows = []
    for c in FEATURE_COLUMNS:
        auc = roc_auc_score(y, df[c].to_numpy(dtype=float))
        rows.append({"feature": c, "auc": float(max(auc, 1 - auc)), "direction": int(auc >= 0.5)})
    return sorted(rows, key=lambda r: -r["auc"])


def _rowwise_features(df: pd.DataFrame, n: int = 50) -> bool:
    from credit_default.features.build import engineer

    batch = engineer(df[FEATURE_COLUMNS].iloc[:n]).to_numpy()
    single = np.vstack([engineer(df[FEATURE_COLUMNS].iloc[[i]]).to_numpy() for i in range(n)])
    return bool(np.allclose(batch, single, equal_nan=True))


def _leakage(df: pd.DataFrame, cfg) -> dict:
    splits = make_splits(df, cfg.train.splits, cfg.seed)
    key = FEATURE_COLUMNS

    def keys(frame):
        return pd.MultiIndex.from_frame(frame[key])

    train_keys = set(keys(splits.train))
    test = splits.test
    in_train = np.fromiter((k in train_keys for k in keys(test)), bool, len(test))
    uni = _univariate_auc(df)
    suspicious = [r for r in uni if r["auc"] >= LEAKAGE_AUC_WARN]
    parts = {
        "train": splits.train,
        "calibration": splits.calibration,
        "validation": splits.validation,
        "test": splits.test,
    }
    split_rates = {
        name: {"n": len(part), "default_rate": float(part[TARGET].mean())}
        for name, part in parts.items()
    }
    ids = [set(p[ID_COLUMN]) for p in parts.values()]
    overlap = sum(len(a & b) for i, a in enumerate(ids) for b in ids[i + 1 :])
    checks = [
        {
            "check": "ID and target are not model inputs",
            "passed": ID_COLUMN not in FEATURE_COLUMNS and TARGET not in FEATURE_COLUMNS,
        },
        {
            "check": "train / calibration / validation / test are disjoint by ID",
            "passed": overlap == 0,
        },
        {
            "check": f"no single feature has univariate ROC-AUC >= {LEAKAGE_AUC_WARN}",
            "passed": not suspicious,
        },
        {
            "check": "feature engineering is row-wise (a row gets the same features alone as in "
            "a batch), so it cannot carry statistics from other rows or splits",
            "passed": _rowwise_features(df),
        },
    ]
    return {
        "checks": checks,
        "split_default_rates": split_rates,
        "test_rows_with_identical_feature_vector_in_train": int(in_train.sum()),
        "test_rows_with_identical_feature_vector_in_train_share": float(in_train.mean()),
        "univariate_auc_top": uni[:8],
        "temporal_note": (
            "The table has no application date: every row covers the same April-September 2005 "
            "statement window and the label is default in October 2005. A time-based split is "
            "therefore impossible; the split is stratified random (seeded). All features are "
            "observed before the label month, so there is no look-ahead inside a row, but the "
            "test set comes from the same period as training, so the reported metrics measure "
            "in-period generalisation, not performance on a later cohort."
        ),
    }


def build_report(df: pd.DataFrame, cfg, data_info: dict | None = None) -> dict:
    return {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "data": data_info or {},
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "default_rate": float(df[TARGET].mean()),
        "schema": _schema_check(df, cfg),
        "columns": _columns(df),
        "duplicates": _duplicates(df),
        "undocumented_codes": _codes(df),
        "outliers": _outliers(df),
        "domain_flags": _domain_flags(df),
        "distributions": _distributions(df),
        "leakage": _leakage(df, cfg),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def to_markdown(r: dict) -> str:
    d, lk = r["duplicates"], r["leakage"]
    banner = (
        "> **SYNTHETIC FIXTURE: these numbers describe the test fixture, not the real data.**\n\n"
        if r["data"].get("synthetic")
        else ""
    )
    lines = [
        "# Data-quality report",
        "",
        banner + f"Generated {r['generated_at']} at commit `{r['git_commit'][:10]}` from "
        f"`{r['data'].get('path', '?')}` (sha256 `{str(r['data'].get('sha256', '?'))[:12]}...`).",
        "",
        f"- Rows: {r['n_rows']:,}; columns: {r['n_columns']}; default rate: {_pct(r['default_rate'])}",
        f"- Schema contract (Pandera): **{'PASS' if r['schema']['passed'] else 'FAIL'}**",
        f"- Missing values: {sum(c['n_missing'] for c in r['columns'])} in total",
        f"- Duplicate IDs: {d['duplicate_ids']}; duplicate rows (excluding ID): "
        f"{d['duplicate_rows_excluding_id']}; duplicate feature vectors: "
        f"{d['duplicate_feature_vectors']} ({d['groups_with_conflicting_labels']} groups with "
        "conflicting labels). " + d["note"],
        "",
        "## Undocumented codes",
        "",
        "| Column | Code | Rows | Default rate | Handling |",
        "|---|---|---|---|---|",
    ]
    for c in r["undocumented_codes"]:
        rate = "-" if c["default_rate"] is None else _pct(c["default_rate"])
        lines.append(f"| {c['column']} | {c['code']} | {c['n']:,} | {rate} | {c['handling']} |")
    lines += [
        "",
        "## Domain flags (kept, but worth knowing)",
        "",
        "| Check | Rows | Share | Default rate |",
        "|---|---|---|---|",
    ]
    for f in r["domain_flags"]:
        rate = "-" if f["default_rate"] is None else _pct(f["default_rate"])
        lines.append(f"| {f['check']} | {f['n']:,} | {_pct(f['share'])} | {rate} |")
    lines += [
        "",
        f"## Outliers (outside Tukey fences at {OUTLIER_IQR_K} x IQR)",
        "",
        "Kept: they are genuine large balances and payments, not entry errors (the contract "
        "bounds catch unit errors). Tree models split on ranks and are insensitive to them; "
        "the linear baseline sees them standardised but unclipped. Stateful preprocessing "
        "(scaler, one-hot encoder, SMOTE) is fitted inside the model pipeline on training "
        "rows only.",
        "",
        "| Column | Upper fence | Rows outside | Share |",
        "|---|---|---|---|",
    ]
    for o in sorted(r["outliers"], key=lambda o: -o["n_outside"]):
        lines.append(
            f"| {o['column']} | {o['upper_fence']:,.0f} | {o['n_outside']:,} | "
            f"{_pct(o['share_outside'])} |"
        )
    lines += [
        "",
        "## Leakage and split checks",
        "",
        *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in lk["checks"]],
        f"- Test rows whose exact feature vector also occurs in train: "
        f"{lk['test_rows_with_identical_feature_vector_in_train']} "
        f"({_pct(lk['test_rows_with_identical_feature_vector_in_train_share'])})",
        "- Default rate per split: "
        + ", ".join(
            f"{k} {v['n']:,} rows / {_pct(v['default_rate'])}"
            for k, v in lk["split_default_rates"].items()
        ),
        "- Strongest single features (univariate ROC-AUC): "
        + ", ".join(f"{u['feature']} {u['auc']:.3f}" for u in lk["univariate_auc_top"][:5]),
        "",
        lk["temporal_note"],
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = write_json(report, out_dir / "data_quality.json")
    m = out_dir / "data_quality.md"
    m.write_text(to_markdown(report))
    return j, m


def main(argv: list[str] | None = None) -> int:
    from credit_default.pipeline.steps import data_fingerprint

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    path = REPO_ROOT / cfg.data.raw_path
    df = pd.read_csv(path)
    info = data_fingerprint(cfg, path)
    out = next((f.split("=", 1)[1] for f in flags if f.startswith("--out=")), None)
    default_dir = cfg.paths.results_dir if not info["synthetic"] else cfg.paths.artifacts_dir
    out_dir = Path(out) if out else REPO_ROOT / default_dir
    report = build_report(df, cfg, info)
    j, m = write_report(report, out_dir)
    log.info("data-quality report: %s, %s (schema %s)", j, m, report["schema"]["passed"])
    return 0 if report["schema"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
