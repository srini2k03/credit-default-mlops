# Drift scenarios and monitoring decisions

Generated 2026-11-24T11:19:15Z at commit `1faa0a8f8f`. Reference: training split (n = 18,000); batches of 3,000 rows simulated from the test split. Model: lightgbm (cutoff 0.160).

Policy: block if any column misses > 2% or unseen codes appear; retrain if > 30% of features drift (PSI >= 0.2, or >= 0.1 with p < 0.01), prediction PSI >= 0.2, or (with >= 1000 labels) ROC-AUC drops > 0.03, the calibration gap exceeds +/-0.05, or the deployed cutoff costs > 5% more than the batch-optimal one.

| Scenario | Drifted / significant features | Prediction PSI | Decline rate | Default rate | ROC-AUC | Calibration gap | Cost regret | Action without labels | Action with labels |
|---|---|---|---|---|---|---|---|---|---|
| control | 0 / 0 | 0.005 | 0.486 | 0.213 | 0.777 | +0.010 | 1.5% | none | **none** |
| covariate_shift | 3 / 9 | 0.223 | 0.675 | 0.220 | 0.763 | +0.042 | 3.5% | retrain | **monitor** |
| missing_values | 0 / 0 | 0.040 | 0.525 | 0.213 | 0.802 | +0.026 | 2.0% | block | **block** |
| new_categories | 2 / 2 | 0.002 | 0.477 | 0.223 | 0.782 | -0.001 | 1.4% | block | **block** |
| riskier_mix | 3 / 18 | 0.414 | 0.714 | 0.353 | 0.796 | +0.006 | 0.8% | retrain | **monitor** |
| label_shift | 0 / 8 | 0.035 | 0.541 | 0.350 | 0.774 | -0.096 | 15.9% | monitor | **retrain** |
| concept_drift | 0 / 0 | 0.004 | 0.487 | 0.276 | 0.745 | -0.051 | 21.5% | monitor | **retrain** |

## Why

- **control** (Untouched random sample of the test split.): nothing moved
- **covariate_shift** (Credit limits x0.7, ages -6 years, PAY_0 one month worse for 35% of rows, bill amounts x1.25.): prediction PSI 0.223 >= 0.2; labelled performance within limits: the model handles this shift
- **missing_values** (15% of BILL_AMT1-3 and PAY_AMT1 values missing (upstream feed failure).): missing rate 15.9% > 2% on BILL_AMT1, BILL_AMT2, BILL_AMT3, PAY_AMT1; fix the upstream feed; scores unreliable
- **new_categories** (10% of rows carry EDUCATION 7/8 or MARRIAGE 4, codes the model never saw.): unseen category codes {'EDUCATION': [7, 8], 'MARRIAGE': [4]}; fix the upstream feed; scores unreliable
- **riskier_mix** (Resampled so 60% of applicants are already 1+ months late on PAY_0 (about 23% in the reference).): prediction PSI 0.414 >= 0.2; labelled performance within limits: the model handles this shift; default rate 35.3% (p = 3.6e-55)
- **label_shift** (Resampled to a ~35% default rate, features drawn from the same per-class distributions (prior shift).): calibration gap -0.096 (mean predicted - observed) beyond +/-0.05; deployed cutoff costs 16% more than the batch-optimal cutoff 0.105 (limit 5%)
- **concept_drift** (Same applicants, but 30% of non-defaulters with utilisation >= 80% and PAY_0 <= 0 now default.): ROC-AUC dropped 0.036 > 0.03; calibration gap -0.051 (mean predicted - observed) beyond +/-0.05; deployed cutoff costs 22% more than the batch-optimal cutoff 0.110 (limit 5%)

Limitations: batches are simulated, not production traffic; PSI depends on the binning; univariate tests miss joint changes (concept_drift has no feature drift at all); significance with thousands of rows says little about impact, which is why actions need effect sizes or labelled performance.
