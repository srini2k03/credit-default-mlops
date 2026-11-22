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
        loadtest docker-build compose-up compose-down site clean \
        infra-bootstrap infra-validate app-init up destroy infra-plan nuke \
        kind-up kind-loadtest kind-down

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

# ---- AWS (see docs/aws_setup.md; run from AWS CloudShell or with admin credentials) -------------
TF ?= terraform
INFRA_BOOT := infra/bootstrap
INFRA_APP := infra/app

infra-bootstrap:  ## one-time AWS bootstrap: state bucket, OIDC deploy role, ECR, S3 buckets, budget
	@test -f $(INFRA_BOOT)/terraform.tfvars || { echo "cp $(INFRA_BOOT)/terraform.tfvars.example $(INFRA_BOOT)/terraform.tfvars and edit it first"; exit 1; }
	cd $(INFRA_BOOT) && $(TF) init -input=false && $(TF) apply
	@# the local state is the source of truth; keep a copy in the state bucket
	cd $(INFRA_BOOT) && aws s3 cp terraform.tfstate "s3://$$($(TF) output -raw state_bucket)/bootstrap/terraform.tfstate" --only-show-errors
	@echo; echo "Set these GitHub repository variables:"; cd $(INFRA_BOOT) && $(TF) output -raw gh_variable_commands

# Values default to the bootstrap outputs (its local state lives in CloudShell); export them to override.
boot_out = $(shell $(TF) -chdir=$(INFRA_BOOT) output -raw $(1) 2>/dev/null)
AWS_REGION ?= $(or $(call boot_out,region),us-east-1)
TF_STATE_BUCKET ?= $(call boot_out,state_bucket)
MLFLOW_ARTIFACT_BUCKET ?= $(call boot_out,mlflow_artifact_bucket)
ALARM_EMAIL ?= $(call boot_out,alarm_email)
IMAGE ?=
APP_VARS = -var "region=$(AWS_REGION)" -var "mlflow_artifact_bucket=$(MLFLOW_ARTIFACT_BUCKET)" \
           -var "alarm_email=$(ALARM_EMAIL)" $(if $(IMAGE),-var "container_image=$(IMAGE)")

app-init:
	@test -n "$(TF_STATE_BUCKET)" || { echo "TF_STATE_BUCKET unknown: run make infra-bootstrap or export the bootstrap outputs"; exit 1; }
	cd $(INFRA_APP) && $(TF) init -input=false -reconfigure \
	  -backend-config="bucket=$(TF_STATE_BUCKET)" -backend-config="region=$(AWS_REGION)"

up: app-init  ## create the app stack (ALB + Fargate, about 7 cents an hour); IMAGE=... picks the image
	cd $(INFRA_APP) && $(TF) apply -input=false -auto-approve $(APP_VARS)
	@cd $(INFRA_APP) && echo "API: $$($(TF) output -raw alb_url)/docs  (run make destroy when done)"

destroy: app-init  ## destroy the app stack (bootstrap, buckets and images stay)
	cd $(INFRA_APP) && $(TF) destroy -input=false -auto-approve $(APP_VARS)

infra-plan: app-init  ## terraform plan for the app stack
	cd $(INFRA_APP) && $(TF) plan -input=false $(APP_VARS)

nuke:  ## destroy the app stack, empty the state bucket, then destroy the bootstrap stack
	@printf "This deletes the app stack, all buckets (state, MLflow, DVC) and ECR images. Type nuke: "; \
	  read ans; [ "$$ans" = nuke ] || { echo aborted; exit 1; }
	$(MAKE) destroy
	bash scripts/aws/empty_bucket.sh "$(TF_STATE_BUCKET)"
	cd $(INFRA_BOOT) && $(TF) destroy -input=false -auto-approve

# ---- Kubernetes on a laptop (kind + Helm + HPA; needs docker, kind, kubectl, helm) -------------
KIND_CLUSTER ?= credit-default

kind-up:  ## create a local kind cluster (NodePort 30080 -> localhost:8080)
	kind create cluster --name $(KIND_CLUSTER) --config deploy/kind/kind-config.yaml

kind-loadtest:  ## build, install the chart, stepped Locust load with HPA sampling -> results/k8s/
	CLUSTER=$(KIND_CLUSTER) PY=$(PY) CONTEXT="Local run (make kind-loadtest)." bash scripts/k8s/run_loadtest.sh

kind-down:  ## delete the local kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

infra-validate:  ## terraform fmt check + validate for both stacks (no AWS credentials needed)
	$(TF) fmt -check -recursive infra
	cd $(INFRA_BOOT) && $(TF) init -backend=false -input=false >/dev/null && $(TF) validate
	@if [ -d $(INFRA_APP) ]; then cd $(INFRA_APP) && $(TF) init -backend=false -input=false >/dev/null && $(TF) validate; fi

clean:
	rm -rf artifacts outputs reports site mlruns logs .pytest_cache .ruff_cache
