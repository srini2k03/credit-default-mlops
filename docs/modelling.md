# Modelling

Source of every number: `results/metrics.json` / `results/metrics.md` from train run
[45371941736](https://github.com/srini2k03/credit-default-mlops/actions/runs/45371941736)
(commit `48a6a54`), test split n = 6,000.

## One protocol for every family

1. Optuna TPE (seed 42, 30 trials, defaults enqueued as trial 0) maximises stratified 3-fold CV
   PR-AUC on the training split. PR-AUC because it is threshold-free and focuses on the minority
   class.
2. Fit on the training split with the chosen imbalance strategy (class weights; the ablation
   below compares none and SMOTE).
3. Calibrate on the calibration split: none, Platt (sigmoid) and isotonic are all fitted; the one
   with the lowest validation Brier (log loss breaks ties) is kept.
4. Choose the cutoff that minimises expected cost on the validation split.
5. Score once on the test split: ROC-AUC, PR-AUC, Brier, log loss, ECE, precision, recall,
   specificity, F1, balanced accuracy, confusion matrix, expected cost; stratified bootstrap 95%
   CIs (500 resamples) and paired bootstrap differences against the champion.

**Selection rule:** the champion is the deployable family with the lowest expected cost per
applicant on validation; validation PR-AUC breaks ties. ROC-AUC is reported but never used to
choose. CatBoost is compared but not deployable (optional dependency, ~100 MB more in a 512 MB
free-tier container).

## Results (test split)

| Family | ROC-AUC | PR-AUC | Brier | ECE | Cost / applicant | Cutoff | Calibration |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.733 | 0.474 | 0.1435 | 0.014 | 0.620 | 0.11 | isotonic |
| random forest | 0.779 | 0.555 | 0.1357 | 0.019 | 0.562 | 0.145 | Platt |
| **LightGBM (champion)** | **0.781** | **0.559** | **0.1346** | **0.015** | **0.558** | **0.16** | **Platt** |
| XGBoost | 0.780 | 0.555 | 0.1357 | 0.020 | 0.569 | 0.17 | Platt |
| CatBoost (research only) | 0.782 | 0.563 | 0.1352 | 0.021 | 0.559 | 0.17 | Platt |

Paired bootstrap (champion minus other): the PR-AUC and cost differences to random forest,
XGBoost and CatBoost all have 95% intervals that include zero; logistic regression is worse by
0.086 PR-AUC [0.071, 0.100]. The honest reading: the four tree families are indistinguishable
on this data; the protocol picks one reproducibly.

**Previous champion** (train run #1, 3 families, isotonic, demographics as inputs): ROC-AUC
0.778, PR-AUC 0.536, cost 0.5625. The new one improves PR-AUC by 0.023, mostly because Platt
scaling does not create the ties that isotonic regression does (isotonic on the new champion:
PR-AUC 0.534).

## Imbalance ablation (CV PR-AUC, training split)

Class weights and no correction are within one standard deviation for every family; SMOTE is
never better and is clearly worse for XGBoost (0.549 vs 0.568). Class weights stay the default;
calibration then fixes the probability scale.

## Calibration (champion)

| Method | Val Brier | Test Brier | Test ECE | Test PR-AUC |
|---|---|---|---|---|
| none | 0.1702 | 0.1766 | 0.190 | 0.559 |
| Platt (chosen) | 0.1312 | 0.1346 | 0.015 | 0.559 |
| isotonic | 0.1323 | 0.1348 | 0.019 | 0.534 |

Uncalibrated class-weighted scores are inflated (ECE 0.19); both calibrators fix that, and
Platt keeps the ranking intact.

## Thresholds

At FN:FP = 5:1 the cost-optimal cutoff on validation is 0.16 (Bayes cutoff for calibrated scores:
1/6 = 0.167). On test it costs 0.558 per applicant against 0.752 at 0.5 (−25.9%) and catches
77.7% of defaults instead of 35.4%. Max-F1 (0.33) costs 0.629. Details and sensitivity:
[decision engine](decision_engine.md).

## Explainability and errors

- SHAP mean |value| and LightGBM split gain agree closely (Spearman 0.97); permutation importance
  on raw inputs ranks PAY_0 first (PR-AUC drop 0.20) and gives exactly 0 to the excluded inputs.
- 97.6% of missed defaults at the deployed cutoff had no delay on their latest bill
  (PAY_0 ≤ 0): their risk is not visible in six months of history. Segment tables are in
  `results/evaluation.json` and on the dashboard. These are associations, not causes.
