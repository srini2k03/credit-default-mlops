# MLflow run summary

Experiment `credit-default`, exported after the training run.

| runName | status | start_time | test_pr_auc | test_roc_auc | test_business_cost | test_brier | cv_pr_auc | champion | data_source |
|---|---|---|---|---|---|---|---|---|---|
| catboost | FINISHED | 2026-11-24 11:11:50.960000+00:00 | 0.5631 | 0.7823 | 0.5593 | 0.1352 | 0.5658 |  |  |
| xgboost | FINISHED | 2026-11-24 11:11:11.775000+00:00 | 0.5549 | 0.7797 | 0.5685 | 0.1357 | 0.5679 |  |  |
| lightgbm | FINISHED | 2026-11-24 11:10:13.177000+00:00 | 0.5594 | 0.7807 | 0.5578 | 0.1346 | 0.5647 |  |  |
| random_forest | FINISHED | 2026-11-24 11:01:12.845000+00:00 | 0.5551 | 0.7788 | 0.5617 | 0.1357 | 0.5668 |  |  |
| logreg | FINISHED | 2026-11-24 11:01:03.281000+00:00 | 0.4739 | 0.7331 | 0.6198 | 0.1435 | 0.5123 |  |  |
| train-uci-2026-11-24T11:01:03Z | FINISHED | 2026-11-24 11:01:03.253000+00:00 | 0.5594 | 0.7807 | 0.5578 | 0.1346 |  | lightgbm | uci |
| bootstrap-production | FINISHED | 2026-11-24 11:00:56.155000+00:00 |  |  |  |  |  |  |  |

## Registered versions

| Version | Aliases | Run |
|---|---|---|
| 1 | previous-production | `e1f43ce8967b4cf588ba23394d7d38bf` |
| 2 | production, staging | `38eecc3f5025460a9c18a799bd3f0d85` |
