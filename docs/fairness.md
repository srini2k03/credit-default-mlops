# Fairness and responsible modelling

Source: `results/fairness.json` / `fairness.md` (test split, deployed cutoff 0.16, n = 6,000,
stratified bootstrap 95% intervals). Groups use attributes recorded in the data; nothing is
inferred.

## Decision: SEX, MARRIAGE and AGE are not model inputs

Sex, marital status and age are protected attributes in many credit regulations (for example
the US Equal Credit Opportunity Act). Every training run measures what excluding them costs by
refitting the champion family with the same hyper-parameters *with* them: in the committed run
that model scores test PR-AUC 0.5577 and cost 0.5543 versus 0.5594 and 0.5578 for the served
model, differences well inside the bootstrap intervals (about ±0.025). The API still accepts the three fields (backward-compatible contract); the model
ignores them (`tests/test_decision_fairness.py` asserts identical predictions when they change).

Removing an attribute does not make a model fair: other features (credit limit, education,
balances) can act as proxies. That is why the group metrics below are reported for the served
model.

## Group metrics (served model)

| Attribute | Group | n | Default rate | Decline rate | FPR | FNR | Calibration gap |
|---|---|---|---|---|---|---|---|
| SEX | female | 3,598 | 0.213 | 0.471 | 0.391 | 0.232 | +0.006 |
| SEX | male | 2,402 | 0.234 | 0.501 | 0.413 | 0.210 | −0.006 |
| MARRIAGE | married | 2,767 | 0.241 | 0.468 | 0.373 | 0.232 | −0.020 |
| MARRIAGE | single | 3,158 | 0.205 | 0.495 | 0.420 | 0.215 | +0.018 |
| AGE | 21-29 | 1,912 | 0.223 | 0.523 | 0.442 | 0.197 | +0.015 |
| AGE | 30-39 | 2,198 | 0.206 | 0.447 | 0.368 | 0.247 | 0.000 |
| AGE | 40-49 | 1,341 | 0.233 | 0.472 | 0.381 | 0.228 | −0.013 |
| AGE | 50-59 | 473 | 0.237 | 0.514 | 0.429 | 0.214 | −0.002 |
| EDUCATION | graduate school | 2,130 | 0.190 | 0.404 | 0.324 | 0.257 | +0.002 |
| EDUCATION | university | 2,774 | 0.236 | 0.529 | 0.445 | 0.198 | +0.004 |
| EDUCATION | high school | 1,014 | 0.257 | 0.556 | 0.479 | 0.222 | −0.011 |

Small groups (MARRIAGE "others" 75, age 60+ 76, EDUCATION "others" 82) are reported on the
dashboard but flagged: their intervals span tens of percentage points.

Reading the table:

- **Sex:** approval-rate ratio 0.94, FPR and FNR gaps 0.022; calibration within ±0.006.
- **Age:** the youngest band is declined most (52.3%) with the highest FPR (0.442): the
  approval-rate ratio across the four sizeable bands is 0.86. The age-aware model has slightly
  smaller gaps (FPR gap 0.062 vs 0.075), so excluding age did not reduce the age gap here:
  proxies carry it.
- **Education:** the largest gap (FPR 0.324 vs 0.479), and EDUCATION *is* a model input. The
  decline rates largely follow the observed default rates (0.190 vs 0.257) and calibration
  within each group is within ±0.011, but equal error rates are not achieved; whether that is
  acceptable is a policy question, not a modelling one.

## Limitations

- Equalised odds, demographic parity and calibration within groups cannot all hold when default
  rates differ; all three are reported instead of declaring one "fair".
- Passing a metric is not evidence of legal or regulatory compliance.
- A 2005 sample from one bank says nothing about other populations; there are no protected
  attributes beyond the four recorded ones.
