# Quality gate

**Decision: PASS**

- challenger passed every promotion check against production

Challenger: models:/credit-default@staging (v2)  
Production: models:/credit-default@production (v1)

| Metric | Challenger | Production |
|---|---|---|
| pr_auc | 0.5594 | 0.5594 |
| roc_auc | 0.7807 | 0.7807 |
| business_cost | 0.5578 | 0.5578 |
| brier | 0.1346 | 0.1346 |
| ece | 0.0146 | 0.0146 |
| recall | 0.7769 | 0.7769 |
| f1 | 0.4880 | 0.4880 |
| threshold | 0.1600 | 0.1600 |

## Promotion checks

| Check | Group | Result | Value | Limit |
|---|---|---|---|---|
| roc_auc_floor | performance | pass | 0.7807 | >= 0.7 |
| pr_auc_floor | performance | pass | 0.5594 | >= 0.45 |
| ece_ceiling | calibration | pass | 0.0146 | <= 0.05 |
| pr_auc_vs_champion | performance | pass | 0.5594 | >= champion 0.5594 - 0.005 |
| brier_vs_champion | calibration | pass | 0.1346 | <= champion 0.1346 + 0.002 |
| cost_vs_champion | cost | pass | 0.5578 | <= champion 0.5578 x (1 + 0.01) |
| input_contract | operational | pass | True | API columns unchanged |
| probabilities_valid | operational | pass | True | finite, in [0, 1] |
| deterministic | operational | pass | True | same input -> same output |
| threshold_valid | operational | pass | 0.1600 | in (0, 1) |
| p95_latency_ms | operational | pass | 7.3449 | <= 50 ms (single row, in-process) |
| bundle_size_mb | operational | pass | 0.2999 | <= 50 MB |
