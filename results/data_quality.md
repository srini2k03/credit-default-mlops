# Data-quality report

Generated 2026-11-24T11:00:49Z at commit `1faa0a8f8f` from `data/raw/credit_default.csv` (sha256 `af6aa9d50511...`).

- Rows: 30,000; columns: 25; default rate: 22.12%
- Schema contract (Pandera): **PASS**
- Missing values: 0 in total
- Duplicate IDs: 0; duplicate rows (excluding ID): 35; duplicate feature vectors: 56 (21 groups with conflicting labels). Duplicates are kept: identical feature vectors are plausible for distinct customers (e.g. unused cards with all-zero balances), and IDs are unique.

## Undocumented codes

| Column | Code | Rows | Default rate | Handling |
|---|---|---|---|---|
| EDUCATION | 0 | 14 | 0.00% | folded into 'others' (4) |
| EDUCATION | 5 | 280 | 6.43% | folded into 'others' (4) |
| EDUCATION | 6 | 51 | 15.69% | folded into 'others' (4) |
| MARRIAGE | 0 | 54 | 9.26% | folded into 'others' (3) |
| PAY_* | -2 | 24,415 | - | kept as an ordinal level (no consumption; not in the original paper) |
| PAY_* | 0 | 95,919 | - | kept as an ordinal level (revolving credit / minimum paid; not in the original paper) |

## Domain flags (kept, but worth knowing)

| Check | Rows | Share | Default rate |
|---|---|---|---|
| negative bill (credit balance) in any month | 1,930 | 6.43% | 16.48% |
| bill above credit limit in any month | 3,931 | 13.10% | 30.09% |
| all six bills and payments zero (inactive card) | 795 | 2.65% | 37.61% |
| payment larger than the previous bill (PAY_AMT1 > BILL_AMT2 > 0) | 2,757 | 9.19% | 10.41% |

## Outliers (outside Tukey fences at 3.0 x IQR)

Kept: they are genuine large balances and payments, not entry errors (the contract bounds catch unit errors). Tree models split on ranks and are insensitive to them; the linear baseline sees them standardised but unclipped. Stateful preprocessing (scaler, one-hot encoder, SMOTE) is fitted inside the model pipeline on training rows only.

| Column | Upper fence | Rows outside | Share |
|---|---|---|---|
| PAY_AMT6 | 15,647 | 1,639 | 5.46% |
| PAY_AMT2 | 17,501 | 1,638 | 5.46% |
| PAY_AMT1 | 17,024 | 1,629 | 5.43% |
| PAY_AMT4 | 15,165 | 1,583 | 5.28% |
| PAY_AMT3 | 16,850 | 1,560 | 5.20% |
| PAY_AMT5 | 15,368 | 1,544 | 5.15% |
| BILL_AMT5 | 195,473 | 949 | 3.16% |
| BILL_AMT6 | 193,025 | 930 | 3.10% |
| BILL_AMT4 | 211,044 | 900 | 3.00% |
| BILL_AMT3 | 232,660 | 858 | 2.86% |
| BILL_AMT2 | 247,071 | 791 | 2.64% |
| BILL_AMT1 | 257,688 | 786 | 2.62% |
| LIMIT_BAL | 810,000 | 1 | 0.00% |
| AGE | 80 | 0 | 0.00% |

## Leakage and split checks

- PASS: ID and target are not model inputs
- PASS: train / calibration / validation / test are disjoint by ID
- PASS: no single feature has univariate ROC-AUC >= 0.9
- PASS: feature engineering is row-wise (a row gets the same features alone as in a batch), so it cannot carry statistics from other rows or splits
- Test rows whose exact feature vector also occurs in train: 11 (0.18%)
- Default rate per split: train 18,000 rows / 22.12%, calibration 3,000 rows / 22.13%, validation 3,000 rows / 22.10%, test 6,000 rows / 22.12%
- Strongest single features (univariate ROC-AUC): PAY_0 0.690, PAY_2 0.639, PAY_3 0.624, LIMIT_BAL 0.618, PAY_AMT1 0.611

The table has no application date: every row covers the same April-September 2005 statement window and the label is default in October 2005. A time-based split is therefore impossible; the split is stratified random (seeded). All features are observed before the label month, so there is no look-ahead inside a row, but the test set comes from the same period as training, so the reported metrics measure in-period generalisation, not performance on a later cohort.
