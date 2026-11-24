# Cost-sensitive decision engine

`credit_default.decision.engine` turns a calibrated probability of default into approve /
decline under a stated cost policy. It is an educational decision-support component, not an
approved lending engine; costs are relative units, not money.

## Policy

```yaml
# configs/cost/default.yaml
fn_cost: 5.0            # approving a customer who defaults (exposure x loss given default)
fp_cost: 1.0            # declining a customer who would have repaid (lost margin)
threshold: null         # fixed cutoff instead of the cost minimiser
min_approval_rate: null # business constraint, e.g. 0.70
sensitivity_ratios: [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
```

Decline when P(default) ≥ cutoff. With calibrated probabilities the Bayes-optimal cutoff is
fp / (fp + fn); the engine still chooses the cutoff empirically on validation (robust to
residual miscalibration) and reports both. Why 5:1: card loss given default is roughly 60-80%
of the balance while annual net margin is 10-15% of it, i.e. 4:1 to 8:1; 5:1 is the conservative
end. It is an assumption, which is why the sensitivity analysis exists.

```bash
python -m credit_default.decision.engine --fn-cost=8 --fp-cost=1 --min-approval=0.7
```

The API applies a request-level policy with `?threshold=` or `?fn_cost=&fp_cost=`.

## Results (test split, cutoffs chosen on validation; `results/decision.md`)

| Rule | Cutoff | Cost / applicant | Recall | Precision | Approval | FPR | FNR |
|---|---|---|---|---|---|---|---|
| deployed (cost-optimal) | 0.160 | 0.5578 | 77.7% | 35.6% | 51.7% | 40.0% | 22.3% |
| Bayes cutoff | 0.167 | 0.5530 | 76.3% | 36.7% | 54.0% | 37.4% | 23.7% |
| 0.5 | 0.500 | 0.7523 | 35.4% | 67.2% | 88.3% | 4.9% | 64.6% |
| max F1 | 0.334 | 0.6292 | 51.2% | 55.9% | 79.8% | 11.4% | 48.8% |
| Youden J | 0.225 | 0.5747 | 64.0% | 44.5% | 68.2% | 22.6% | 36.0% |

The theoretical Bayes cutoff happens to be slightly cheaper on test than the validation optimum
(0.553 vs 0.558): a reminder that the validation split (3,000 rows) is noisy and that the two
agree within that noise.

## Sensitivity to the cost ratio

| FN:FP | Optimal cutoff | Test cost at optimum | Extra cost of keeping 0.16 |
|---|---|---|---|
| 1:1 | 0.395 | 0.184 | +0.177 |
| 3:1 | 0.225 | 0.415 | +0.044 |
| 5:1 (configured) | 0.160 | 0.558 | 0 |
| 8:1 | 0.125 | 0.663 | +0.042 |
| 10:1 | 0.080 | 0.698 | +0.107 |
| 20:1 | 0.040 | 0.757 | +0.541 |

The optimal cutoff tracks the Bayes cutoff closely, as calibrated scores should. A deployed
cutoff tuned for 5:1 is robust for ratios between about 4:1 and 6:1 (regret ≤ 0.013) and
expensive far outside it, so the ratio is a business decision worth revisiting, not a modelling
detail. The dashboard's evaluation page recomputes all of this live from the held-out predictions.
