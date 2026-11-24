# Drift monitoring

The UCI table has no timestamps and there is no live labelled traffic, so incoming batches are
simulated from the test split (`credit_default.monitor.scenarios`). Every scenario is a
documented failure mode; the policy turns its measurements into an action.

## What is measured per batch

- **Data quality:** missing rate per input, category codes outside the contract.
- **Feature drift:** PSI (quantile bins of the training split; exact shares for discrete
  columns) plus KS for continuous and chi-square (with Cramér's V) for categorical/ordinal inputs.
- **Prediction drift:** PSI of scores on fixed 0.1-wide bins and KS, against held-out
  (validation) scores, because scores on rows the model trained on are over-confident.
- **Label rate:** two-proportion z-test against the training default rate.
- **Labelled performance** (once outcomes arrive): ROC-AUC drop vs the champion's test value,
  calibration gap (mean predicted − observed), and the cost regret of the deployed cutoff vs the
  batch's own optimum. PR-AUC and cost per applicant move with the default rate even when the
  model is fine, so they are reported but never trigger.

## Policy (`configs/monitor/default.yaml` → `policy`)

`block` if any input misses > 2% or unseen codes appear (fix the feed; do not retrain on broken
data). Without labels, `retrain` (a challenger that must still pass the gate) when > 30% of
features drift (PSI ≥ 0.2, or ≥ 0.1 with p < 0.01) or prediction PSI ≥ 0.2. With ≥ 1,000 labels,
`retrain` when ROC-AUC drops > 0.03, |calibration gap| > 0.05 or cost regret > 5%; drift that
labelled performance shows the model handles is only `monitor`ed. Significance alone never acts.

## Results (`results/monitoring/scenarios.md`, batches of 3,000 rows)

| Batch | Drifted / significant features | Prediction PSI | ROC-AUC | Calibration gap | Cost regret | Without labels | With labels |
|---|---|---|---|---|---|---|---|
| control | 0 / 0 | 0.005 | 0.777 | +0.010 | 1.5% | none | none |
| covariate shift | 3 / 9 | 0.223 | 0.763 | +0.042 | 3.5% | retrain | monitor |
| missing values | 0 / 0 | 0.040 | 0.802 | +0.026 | 2.0% | block | block |
| new category codes | 2 / 2 | 0.002 | 0.782 | −0.001 | 1.4% | block | block |
| riskier applicant mix | 3 / 18 | 0.414 | 0.796 | +0.006 | 0.8% | retrain | monitor |
| label (prior) shift | 0 / 8 | 0.035 | 0.774 | −0.096 | 15.9% | monitor | retrain |
| concept drift | 0 / 0 | 0.004 | 0.745 | −0.051 | 21.5% | monitor | retrain |

What the table teaches:

- The bill amounts ×1.25 in the covariate-shift batch are significant on KS (p < 0.01) but stay
  in the stable PSI band: significance with thousands of rows is not impact.
- A riskier applicant mix moves the score distribution a lot (PSI 0.41) but the calibrated model
  is right about the riskier applicants (gap +0.006); retraining would not help.
- Concept drift is invisible without labels: no feature and no score moves, yet ROC-AUC falls
  0.036 and the deployed cutoff costs 21.5% more than the batch optimum.
- New category codes are folded into "others" silently by the feature code; only the
  data-quality check catches them.

## Scheduled path

`retrain.yml` runs `check_drift` weekly on the control batch (expected: no action) and on demand
on any batch; it calls `train.yml` only on `retrain` and raises a warning on `block`. The same
check runs on the API's prediction log (`--slice=log`). Evidently HTML reports are still built
for the control and drifted slices for people to read; they use Evidently's own tests and are
not what the policy acts on.

## Limitations

Simulated batches; PSI depends on binning; univariate tests miss joint changes; the label-based
checks assume outcomes arrive, which for credit takes months.
