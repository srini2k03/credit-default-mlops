# Common tasks. Real-data targets need network access to the UCI archive or the GitHub mirrors.
PY ?= python
export MLFLOW_DISABLE_TELEMETRY := true
export MLFLOW_DISABLE_AGENT_HINT := 1
export DO_NOT_TRACK := 1
export PREFECT_SERVER_ANALYTICS_ENABLED := false
SYN := data=synthetic train=fast
PORT ?= 8000

.PHONY: help install lint format test data fixtures validate train train-synthetic gate gate-synthetic \
        bootstrap publish serve smoke drift drift-synthetic check-drift flow flow-synthetic \
        loadtest docker-build compose-up compose-down site clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

install:  ## editable install with every extra
	$(PY) -m pip install -e ".[dev]"

lint:  ## ruff lint + format check
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:  ## unit + integration tests on the synthetic fixture (no network)
	$(PY) -m pytest

# ---- data -------------------------------------------------------------------------------------
data:  ## real UCI data: DVC pull from the remote, else download + verify; then push
	dvc pull -q data/raw/credit_default.csv 2>/dev/null || dvc repro download
	$(PY) -m credit_default.validation.schema
	dvc push -q || true

fixtures:  ## synthetic fixture with the UCI schema
	$(PY) -m credit_default.data.synthetic

validate:  ## Pandera contract on the real data
	$(PY) -m credit_default.validation.schema

# ---- training and registry --------------------------------------------------------------------
bootstrap:  ## register models/bundled as @production in an empty registry
	$(PY) -m credit_default.registry.bootstrap

train: data  ## full search on real data, registers @staging
	$(PY) -m credit_default.pipeline.train

train-synthetic: fixtures  ## quick run on the fixture
	$(PY) -m credit_default.pipeline.train $(SYN)

gate:  ## staging vs production on the holdout; promotes on pass, exits 1 on reject
	$(PY) -m credit_default.registry.gate --promote

gate-synthetic:
	$(PY) -m credit_default.registry.gate $(SYN) --promote

publish:  ## copy the approved real-data run into results/ and models/bundled/
	$(PY) -m credit_default.pipeline.publish

flow:  ## Prefect flow on real data
	$(PY) flows/training_flow.py

flow-synthetic:
	$(PY) flows/training_flow.py $(SYN)

# ---- serving ----------------------------------------------------------------------------------
serve:  ## API on :$(PORT) (registry if MLFLOW_TRACKING_URI is set, else models/bundled)
	uvicorn credit_default.serve.app:app --host 0.0.0.0 --port $(PORT)

smoke:  ## curl /health, /predict, /predict_batch, /explain on a running API
	bash scripts/smoke_api.sh http://localhost:$(PORT)

loadtest:  ## 60 s headless Locust run against a running API -> artifacts/loadtest.md
	mkdir -p artifacts/locust
	locust -f locust/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:$(PORT) \
	  --csv artifacts/locust/run --only-summary
	$(PY) locust/summarize.py artifacts/locust/run_stats.csv > artifacts/loadtest.md
	cat artifacts/loadtest.md

# ---- monitoring -------------------------------------------------------------------------------
drift:  ## Evidently reports for the control and drifted slices (real data)
	$(PY) -m credit_default.monitor.drift

drift-synthetic:
	$(PY) -m credit_default.monitor.drift data=synthetic

check-drift:  ## exit 1 if the simulated production slice drifts
	$(PY) -m credit_default.monitor.check_drift --slice=drifted

site:  ## static site for GitHub Pages (reports, results, MLflow summary)
	$(PY) scripts/build_site.py

# ---- containers -------------------------------------------------------------------------------
docker-build:
	docker build -f docker/Dockerfile --target api -t credit-default-api .

compose-up:  ## mlflow + api + report server
	docker compose up -d --build

compose-down:
	docker compose down

clean:
	rm -rf artifacts outputs reports site mlruns logs .pytest_cache .ruff_cache
