# API

FastAPI service (`credit_default.serve.app`), live at
https://credit-default-api-fkqg.onrender.com/docs. Every endpoint and Swagger example of the
first release still works; new fields and parameters are additive.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `status`, `model_loaded`, `model_source`, `model_version`, `model_family`, `threshold`; 503 without a model (Render's health check) |
| GET | `/ready` | `ready`, `fast_scoring`, `explainer_ready`; 503 until a model is loaded |
| GET | `/model` | version, URI, family, threshold, calibration, excluded inputs, explanation space, disclaimer, training metadata (commit, data sha256, test metrics) |
| POST | `/predict` | `request_id`, `probability_of_default`, `default_prediction`, `decision`, `threshold`, `threshold_source`, `policy_threshold`, `model_version`, `model_source` |
| POST | `/predict_batch` | `{"applications": [...]}` (1-1,000) → `n`, `predictions` (request IDs `<id>-<n>`) |
| POST | `/explain?top_k=1..31` | probability, decision, `base_value`, ranked `contributions` (feature, value, SHAP value, direction), `raw_score` (= base + all contributions), `uncalibrated_probability`, `other_features_contribution`, `calibration`, `explanation_space` |

Query parameters on the POST endpoints: `threshold` in (0, 1), or `fn_cost` and `fp_cost`
together (cutoff = fp / (fp + fn), the Bayes cutoff for calibrated probabilities). Passing both
forms, or only one cost, is a 422.

## Input validation

One Pydantic model with the UCI column names (`extra="forbid"`): integer types, credit limit
1,000-5,000,000, SEX ∈ {1, 2}, EDUCATION 0-6, MARRIAGE 0-3, AGE 18-100, PAY_* −2..8, bills
−1,000,000..2,000,000, payments 0..2,000,000. Unknown or missing fields → 422.

## Hardening

| Concern | Behaviour | Setting |
|---|---|---|
| Request IDs | `X-Request-ID` accepted if it matches `[A-Za-z0-9._-]{1,64}`, else generated; echoed on every response and in error bodies | - |
| Logs | one JSON object per request (ID, method, path, status, latency, model version); applicant data is never logged | `LOG_FORMAT`, `LOG_LEVEL` |
| Body size | 413 above the limit | `MAX_BODY_BYTES` (2,000,000) |
| Rate limit | sliding one-minute window per client (first `X-Forwarded-For` hop on Render), 429 with `Retry-After`; health checks exempt | `RATE_LIMIT_PER_MINUTE` (120; 0 disables, used by load tests) |
| CORS | only the dashboard origin | `CORS_ORIGINS` |
| Errors | JSON `{"detail", "request_id"}`; 500s say "internal error" and log the exception class only | - |
| Model loading | `@production` from MLflow if `MLFLOW_TRACKING_URI` is set (5 s timeout, no retries), else the committed bundle | `MODEL_SOURCE`, `MODEL_URI`, `MODEL_BUNDLE_PATH` |
| Fast path | NumPy scorer enabled only if bit-identical to the pipeline on a verification sample at startup | `FAST_SCORING` |
| Warm-up | SHAP explainer built at startup so the first `/explain` is not slow | `EXPLAINER_WARMUP` |

The rate limiter is in-process: with several workers or instances each has its own window, and
`X-Forwarded-For` can be spoofed where no trusted proxy sets it. Behind a real gateway the limit
belongs there.

## Prediction log

Every prediction is appended to SQLite (`PREDICTION_LOG_PATH`) with its inputs, score and model
version; `check_drift --slice=log` compares it with the training data. On Render the disk is
ephemeral, so the log lives as long as the instance.
