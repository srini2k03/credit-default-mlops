import numpy as np
import pytest

from credit_default.data.schema import TARGET
from credit_default.validation import DataValidationError, validate


def test_valid_fixture_passes(small_df):
    out = validate(small_df)
    assert len(out) == len(small_df)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.assign(SEX=3), "SEX"),
        (lambda d: d.assign(PAY_0=9), "PAY_0"),
        (lambda d: d.assign(AGE=15), "AGE"),
        (lambda d: d.assign(LIMIT_BAL=-1), "LIMIT_BAL"),
        (lambda d: d.assign(PAY_AMT1=-5), "PAY_AMT1"),
        (lambda d: d.assign(**{TARGET: 2}), TARGET),
        (lambda d: d.drop(columns=["BILL_AMT3"]), "BILL_AMT3"),
        (lambda d: d.assign(extra_col=1), "extra_col"),
        (lambda d: d.assign(ID=1), "ID"),
        (lambda d: d.assign(**{TARGET: 0}), "default rate"),
    ],
)
def test_contract_violations_fail(small_df, mutate, match):
    with pytest.raises(DataValidationError, match=match):
        validate(mutate(small_df.copy()))


def test_nulls_fail(small_df):
    bad = small_df.astype({"AGE": "float64"})
    bad.loc[0, "AGE"] = np.nan
    with pytest.raises(DataValidationError):
        validate(bad)


def test_too_few_rows_fail(small_df):
    with pytest.raises(DataValidationError, match="fewer than"):
        validate(small_df.head(100))


def test_fail_fast_mode_stops_at_first_error(small_df):
    with pytest.raises(DataValidationError):
        validate(small_df.assign(SEX=3, AGE=15), lazy=False)
