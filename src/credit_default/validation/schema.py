"""Pandera contract for the canonical table. Training stops here if the data breaks it."""

from __future__ import annotations

import logging
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from credit_default.data.schema import (
    AGE_RANGE,
    BILL_AMT,
    DOMAINS,
    ID_COLUMN,
    LIMIT_BAL_RANGE,
    PAY_AMT,
    PAY_STATUS,
    RAW_COLUMNS,
    TARGET,
)

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input table violates the data contract."""


def build_schema(
    min_rows: int = 1000,
    default_rate_min: float = 0.10,
    default_rate_max: float = 0.40,
    require_target: bool = True,
) -> pa.DataFrameSchema:
    def col(*checks, unique=False):
        return pa.Column("int64", checks=list(checks), nullable=False, unique=unique, coerce=True)

    columns = {
        ID_COLUMN: col(pa.Check.gt(0), unique=True),
        "LIMIT_BAL": col(pa.Check.in_range(*LIMIT_BAL_RANGE)),
        "SEX": col(pa.Check.isin(DOMAINS["SEX"])),
        "EDUCATION": col(pa.Check.isin(DOMAINS["EDUCATION"])),
        "MARRIAGE": col(pa.Check.isin(DOMAINS["MARRIAGE"])),
        "AGE": col(pa.Check.in_range(*AGE_RANGE)),
        **{c: col(pa.Check.isin(DOMAINS["PAY_STATUS"])) for c in PAY_STATUS},
        # Bills can be negative (overpayment / credit balance); bounded to catch unit errors.
        **{c: col(pa.Check.in_range(-1_000_000, 2_000_000)) for c in BILL_AMT},
        **{c: col(pa.Check.in_range(0, 2_000_000)) for c in PAY_AMT},
    }
    checks = [pa.Check(lambda df: len(df) >= min_rows, error=f"fewer than {min_rows} rows")]
    if require_target:
        columns[TARGET] = col(pa.Check.isin((0, 1)))
        checks.append(
            pa.Check(
                lambda df: default_rate_min <= df[TARGET].mean() <= default_rate_max,
                error=f"default rate outside [{default_rate_min}, {default_rate_max}]",
            )
        )
    return pa.DataFrameSchema(
        columns=columns, checks=checks, strict=True, ordered=False, name="credit_default"
    )


def validate(df: pd.DataFrame, cfg=None, lazy: bool | None = None) -> pd.DataFrame:
    """Validate and coerce; raise ``DataValidationError`` with a readable summary on failure."""
    v = cfg.validation if cfg is not None else None
    schema = build_schema(
        min_rows=v.min_rows if v is not None else 1000,
        default_rate_min=v.default_rate_min if v is not None else 0.10,
        default_rate_max=v.default_rate_max if v is not None else 0.40,
    )
    lazy = (v.lazy if v is not None else True) if lazy is None else lazy
    try:
        out = schema.validate(df, lazy=lazy)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        summary = (
            cases.assign(failure_case=cases["failure_case"].astype(str))
            .groupby(["column", "check"], dropna=False)["failure_case"]
            .agg(n_failures="size", examples=lambda s: ", ".join(s.unique()[:3]))
            .reset_index()
        )
        raise DataValidationError(
            f"{len(cases)} contract violations:\n{summary.to_string(index=False)}"
        ) from exc
    except SchemaError as exc:
        raise DataValidationError(str(exc)) from exc
    log.info("validated %d rows x %d columns", len(out), len(out.columns))
    return out[RAW_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    from credit_default.config import REPO_ROOT, load_config, split_cli

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, _ = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    df = pd.read_csv(REPO_ROOT / cfg.data.raw_path)
    try:
        validate(df, cfg)
    except DataValidationError as exc:
        log.error("data contract failed:\n%s", exc)
        return 1
    log.info("data contract passed: %s", cfg.data.raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
