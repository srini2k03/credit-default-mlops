#!/usr/bin/env bash
# Sample the HPA and pod counts every INTERVAL seconds into a CSV until killed.
# Usage: sample_replicas.sh <release fullname> <out.csv> [interval]
set -uo pipefail
name="${1:?usage: sample_replicas.sh <name> <out.csv> [interval]}"
out="${2:?}"
interval="${3:-10}"
selector="app.kubernetes.io/instance=${RELEASE:-credit-default}"
echo "timestamp,hpa_current_replicas,hpa_desired_replicas,cpu_utilization_percent,ready_pods,total_pods" > "$out"
while :; do
  ts=$(date -u +%s)
  hpa=$(kubectl get hpa "$name" -o jsonpath='{.status.currentReplicas},{.status.desiredReplicas},{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null || echo ",,")
  pods=$(kubectl get pods -l "$selector" -o jsonpath='{range .items[*]}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null)
  total=$(grep -c . <<<"$pods" || true)
  ready=$(grep -c '^true$' <<<"$pods" || true)
  echo "${ts},${hpa},${ready},${total}" >> "$out"
  sleep "$interval"
done
