# MLflow run summary

Experiment `credit-default`, exported after the training run.

| runName | status | start_time | test_pr_auc | test_roc_auc | test_business_cost | test_brier | cv_pr_auc | champion | data_source |
|---|---|---|---|---|---|---|---|---|---|
| xgboost | FINISHED | 2026-11-03 13:39:18.829000+00:00 | 0.5351 | 0.7799 | 0.5682 | 0.1358 | 0.5683 |  |  |
| lightgbm | FINISHED | 2026-11-03 13:38:37.428000+00:00 | 0.5362 | 0.7781 | 0.5625 | 0.1353 | 0.5640 |  |  |
| logreg | FINISHED | 2026-11-03 13:38:29.192000+00:00 | 0.4782 | 0.7403 | 0.6012 | 0.1431 | 0.5133 |  |  |
| train-uci-2026-11-03T13:38:29Z | FINISHED | 2026-11-03 13:38:29.163000+00:00 | 0.5362 | 0.7781 | 0.5625 | 0.1353 |  | lightgbm | uci |

## Registered versions

| Version | Aliases | Run |
|---|---|---|
| 1 | production, staging | `b8bfce8214cc4d2bbc9cc79fe1f74d0b` |
