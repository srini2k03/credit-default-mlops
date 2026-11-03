"""Column names and value domains of the UCI Default of Credit Card Clients table.

The canonical CSV keeps the UCI column names; only the target is renamed from
"default payment next month" to ``default_payment_next_month``.
"""

from __future__ import annotations

ID_COLUMN = "ID"
TARGET = "default_payment_next_month"
UCI_TARGET_NAMES = ("default payment next month", "default.payment.next.month")

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # UCI skips PAY_1
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS,
    *BILL_AMT,
    *PAY_AMT,
]
RAW_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]

# Domains observed in the published data (documented codes plus the undocumented ones that occur).
DOMAINS = {
    "SEX": (1, 2),  # 1 male, 2 female
    "EDUCATION": (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ),  # 1 grad school, 2 university, 3 high school, 4 others; 0/5/6 unlabelled
    "MARRIAGE": (0, 1, 2, 3),  # 1 married, 2 single, 3 others; 0 unlabelled
    "PAY_STATUS": tuple(range(-2, 9)),  # -2 no use, -1 paid duly, 0 revolving, 1..8 months late
}
AGE_RANGE = (21, 79)
LIMIT_BAL_RANGE = (10_000, 1_000_000)
