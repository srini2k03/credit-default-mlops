# results/

Everything in this folder except this README and the placeholders is written by the `train`
GitHub Actions workflow from a run on the **real UCI data** that passed the quality gate:

| File | Content |
|---|---|
| `metrics.json`, `metrics.md` | test-holdout metrics for every model family, threshold rules, imbalance ablation |
| `gate.json`, `gate.md` | challenger vs production decision |
| `figures/` | cost-vs-threshold curve, reliability plot, SHAP bar and beeswarm |
| `loadtest.md` | Locust p50/p95/p99 and RPS per endpoint |
| `mlflow_summary.md` | MLflow runs and registered versions/aliases from that run |

`python -m credit_default.pipeline.publish` refuses to write synthetic-fixture runs here.
