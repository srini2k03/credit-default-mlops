#!/usr/bin/env bash
# kind + Helm + HPA load test. Used by .github/workflows/k8s-loadtest.yml and `make kind-loadtest`.
# Needs docker, kind, kubectl, helm and a Python with locust, pandas and matplotlib.
#
# Environment (defaults in brackets):
#   CLUSTER [credit-default]   kind cluster (created by `make kind-up` or helm/kind-action)
#   STEPS [10,25,50,100,150]   Locust users per step     STEP_SECONDS [120]
#   COOLDOWN_SECONDS [300]     sampling after the load, to see the scale-down
#   OUT [results/k8s]          report directory          RAW [/tmp/k8s-loadtest] raw files
#   CONTEXT [""]               one line describing where it ran (goes into summary.md)
#   SKIP_BUILD [false]         reuse an already loaded credit-default-api:kind image
set -euo pipefail

CLUSTER="${CLUSTER:-credit-default}"
RELEASE="${RELEASE:-credit-default}"
STEPS="${STEPS:-10,25,50,100,150}"
STEP_SECONDS="${STEP_SECONDS:-120}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-300}"
OUT="${OUT:-results/k8s}"
RAW="${RAW:-/tmp/k8s-loadtest}"
BASE="http://localhost:8080"
PY="${PY:-python}"
export RELEASE

mkdir -p "$RAW" "$OUT"

echo "== metrics-server"
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/ >/dev/null 2>&1 || true
helm repo update metrics-server >/dev/null
helm upgrade --install metrics-server metrics-server/metrics-server -n kube-system \
  -f deploy/kind/metrics-server-values.yaml --wait --timeout 5m

if [ "${SKIP_BUILD:-false}" != "true" ]; then
  echo "== build and load the api image"
  docker build -f docker/Dockerfile --target api -t credit-default-api:kind .
  kind load docker-image credit-default-api:kind --name "$CLUSTER"
fi

echo "== install the chart"
helm lint deploy/helm/credit-default -f deploy/kind/values-loadtest.yaml
helm upgrade --install "$RELEASE" deploy/helm/credit-default -f deploy/kind/values-loadtest.yaml \
  --wait --timeout 5m
kubectl rollout status "deployment/$RELEASE" --timeout=5m

echo "== wait for the API and for pod metrics"
for i in $(seq 1 60); do curl -fsS -o /dev/null "$BASE/health" && break; sleep 2; done
curl -fsS "$BASE/health"; echo
bash scripts/smoke_api.sh "$BASE"
for i in $(seq 1 36); do
  kubectl top pods -l "app.kubernetes.io/instance=$RELEASE" >/dev/null 2>&1 && break
  sleep 5
done
kubectl get hpa "$RELEASE"

echo "== stepped load: $STEPS users, ${STEP_SECONDS}s each, then ${COOLDOWN_SECONDS}s cool-down"
bash scripts/k8s/sample_replicas.sh "$RELEASE" "$RAW/replicas.csv" 10 &
sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT
LOCUST_STEPS="$STEPS" LOCUST_STEP_SECONDS="$STEP_SECONDS" \
  locust -f locust/step_load.py --headless --host "$BASE" \
  --csv "$RAW/locust" --csv-full-history --only-summary --loglevel WARNING --exit-code-on-error 0
sleep "$COOLDOWN_SECONDS"
kill "$sampler" 2>/dev/null || true

kubectl get hpa "$RELEASE" -o wide > "$RAW/hpa.txt" || true
kubectl describe hpa "$RELEASE" > "$RAW/hpa-describe.txt" || true
kubectl get events --sort-by=.lastTimestamp > "$RAW/events.txt" || true

echo "== report"
"$PY" scripts/k8s/loadtest_report.py --locust-prefix "$RAW/locust" --replicas "$RAW/replicas.csv" \
  --out "$OUT" --steps "$STEPS" --step-seconds "$STEP_SECONDS" --context "${CONTEXT:-}"
