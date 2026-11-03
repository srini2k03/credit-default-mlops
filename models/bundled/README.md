# Bundled production model

`model.joblib` and `metadata.json` in this folder are written **only** by the `train` GitHub
Actions workflow, after it trains on the real UCI data and the quality gate passes
(`python -m credit_default.pipeline.publish` refuses synthetic-fixture runs).

The API loads this file when no MLflow registry is reachable (Render, CI smoke tests,
`docker run` without compose). `metadata.json` records the commit, data checksum, test metrics
and threshold of the model in this folder.

Until the first `train` run has finished, this folder holds only this README and the API's
`/health` returns 503.
