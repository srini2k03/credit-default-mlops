# Data card

**Source.** UCI Default of Credit Card Clients (id 350), Yeh & Lien (2009), CC BY 4.0. Downloaded
from the UCI archive or one of two commit-pinned mirrors; all produce the same canonical CSV
(sha256 `af6aa9d5...`, pinned in `configs/data/uci.yaml`).

**Rows and label.** 30,000 credit-card holders of a Taiwanese bank. Label
`default_payment_next_month` = default in October 2005 (22.12% positive). No missing values,
no duplicate IDs (`results/data_quality.md`).

## Features

| Group | Columns | Used by the model |
|---|---|---|
| Credit limit | `LIMIT_BAL` | yes |
| Demographics | `SEX`, `MARRIAGE`, `AGE` | **no** (accepted by the API, used only for fairness reports) |
| Education | `EDUCATION` (codes 0/5/6 folded into "others") | yes |
| Repayment status | `PAY_0`, `PAY_2`..`PAY_6` (−2 no use, −1 paid in full, 0 revolving, 1-8 months late) | yes |
| Bills | `BILL_AMT1`..`BILL_AMT6` | yes |
| Payments | `PAY_AMT1`..`PAY_AMT6` | yes |
| Engineered (row-wise) | utilisation (latest, mean, trend), payment ratios, max delay, months delayed, months paid in full | yes |

`ID` and the label are never inputs. Why the three demographics are excluded and what that does
and does not achieve: [fairness](fairness.md).

## Preprocessing

All inside the sklearn/imblearn pipeline, so training and serving share one code path:
row-wise feature engineering → (logistic regression only) one-hot + standard scaling →
optional SMOTE (training folds only) → estimator → calibrator. Outliers are kept (genuine
large balances; the contract bounds catch unit errors).

## Split and the temporal question

There is no application date: every row covers the same April-September 2005 statement window
and the label is the next month. A time-based split is therefore impossible. The split is
stratified random and seeded: 60% train, 10% calibration, 10% validation, 20% test, with equal
default rates (22.1% in each). All inputs are observed before the label month, so there is no
look-ahead inside a row; but test rows come from the same period as training rows, so the
metrics measure in-period generalisation, not performance on a later cohort.

## Quality and leakage checks (committed results)

- Pandera contract: pass. 35 duplicate rows and 56 duplicate feature vectors (21 groups with
  conflicting labels), kept: plausible for distinct customers such as unused cards.
- Undocumented codes: EDUCATION 0/5/6 (345 rows), MARRIAGE 0 (54 rows), PAY_* −2 and 0.
- Domain flags: 13.1% of customers had a bill above their limit in some month (30.1% default
  rate); 2.65% inactive cards (37.6% default rate).
- Leakage: splits disjoint by ID; feature engineering verified row-wise; strongest single
  feature PAY_0 has univariate ROC-AUC 0.690 (no feature ≥ 0.9); 11 test rows (0.18%) share an
  exact feature vector with a training row.
