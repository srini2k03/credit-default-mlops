# MLflow run summary

Experiment `credit-default`, exported after the training run.

| runName | status | start_time | test_pr_auc | test_roc_auc | test_business_cost | test_brier | cv_pr_auc | champion | data_source |
|---|---|---|---|---|---|---|---|---|---|
| catboost | FINISHED | 2026-11-24 10:21:18.882000+00:00 | 0.5631 | 0.7823 | 0.5593 | 0.1352 | 0.5658 |  |  |
| xgboost | FINISHED | 2026-11-24 10:20:39.623000+00:00 | 0.5549 | 0.7797 | 0.5685 | 0.1357 | 0.5679 |  |  |
| lightgbm | FINISHED | 2026-11-24 10:19:41.809000+00:00 | 0.5594 | 0.7807 | 0.5578 | 0.1346 | 0.5647 |  |  |
| random_forest | FINISHED | 2026-11-24 10:10:52.822000+00:00 | 0.5551 | 0.7788 | 0.5617 | 0.1357 | 0.5668 |  |  |
| logreg | FINISHED | 2026-11-24 10:10:43.424000+00:00 | 0.4739 | 0.7331 | 0.6198 | 0.1435 | 0.5123 |  |  |
| train-uci-2026-11-24T10:10:43Z | FINISHED | 2026-11-24 10:10:43.397000+00:00 | 0.5594 | 0.7807 | 0.5578 | 0.1346 |  | lightgbm | uci |
| bootstrap-production | FINISHED | 2026-11-24 10:10:36.670000+00:00 |  |  |  |  |  |  |  |

## Registered versions

| Version | Aliases | Run |
|---|---|---|
| 1 | previous-production | `d4e36e733dcd4674b565a533e8ff4fd9` |
| 2 | production, staging | `b92962b32072458c89ab9d81948b0363` |
