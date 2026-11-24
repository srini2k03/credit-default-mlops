# Champion/challenger simulation

Generated 2026-11-24T11:19:31Z at commit `1faa0a8f8f`. Simulation in a temporary MLflow registry on the real train/test splits; the served model is not touched.

| # | Event | Candidate | Version | Production before -> after | PR-AUC | Brier | ECE | Cost / applicant | Failed checks |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **promote** | v1 baseline | v1 | none -> v1 | 0.5269 | 0.1377 | 0.0140 | 0.5858 | - |
| 2 | **promote** | v2 retrained | v2 | v1 -> v2 | 0.5599 | 0.1346 | 0.0142 | 0.5562 | - |
| 3 | **reject** | v3 noisy labels | v3 | v2 -> v2 | 0.5417 | 0.1366 | 0.0232 | 0.5638 | pr_auc_vs_champion, brier_vs_champion, cost_vs_champion |
| 4 | **reject** | v4 uncalibrated | v4 | v2 -> v2 | 0.5599 | 0.1764 | 0.1903 | 0.5553 | ece_ceiling, brier_vs_champion |
| 5 | **rollback** | - | - | v2 -> v1 | - | - | - | - | - |

## Candidates

- **v1 baseline**: First model: default hyper-parameters, 50% of the training rows. -> promote
- **v2 retrained**: Tuned hyper-parameters, all training rows (new labelled outcomes). -> promote
- **v3 noisy labels**: Same recipe, but the outcome feed recorded 40% of defaults as repaid. -> reject
- **v4 uncalibrated**: Same base model as v2 served without calibration (scores inflated by class weights). -> reject
- **rollback**: simulated incident: a downstream system reports an unexpected decline-rate jump after the latest promotion; roll back while it is investigated

Final aliases: @production -> v1, @previous-production -> v2.
