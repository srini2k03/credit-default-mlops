#!/usr/bin/env bash
# Drive the app stack (infra/app) from CI or CloudShell.
#   app_stack.sh up      terraform apply  (IMAGE=<ecr image> optional)
#   app_stack.sh down    terraform destroy
#   app_stack.sh status  print "up <age-minutes>" or "down"
#   app_stack.sh check   destroy when older than AWS_MAX_UP_HOURS (default 4), unless AWS_KEEP_UP=true
# Needs TF_STATE_BUCKET, AWS_REGION, MLFLOW_ARTIFACT_BUCKET; ALARM_EMAIL for "up".
set -euo pipefail

PROJECT="${PROJECT:-credit-default}"
CREATED_PARAM="/${PROJECT}/app/created_at"
MAX_UP_HOURS="${AWS_MAX_UP_HOURS:-4}"
KEEP_UP="${AWS_KEEP_UP:-false}"
: "${TF_STATE_BUCKET:?TF_STATE_BUCKET is required}"
: "${AWS_REGION:?AWS_REGION is required}"
: "${MLFLOW_ARTIFACT_BUCKET:?MLFLOW_ARTIFACT_BUCKET is required}"
APP_DIR="$(cd "$(dirname "$0")/../../infra/app" && pwd)"

summary() { echo "$*"; [ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$*" >> "$GITHUB_STEP_SUMMARY"; return 0; }

tf() { terraform -chdir="$APP_DIR" "$@"; }

tf_init() {
  tf init -input=false -reconfigure \
    -backend-config="bucket=${TF_STATE_BUCKET}" -backend-config="region=${AWS_REGION}" >/dev/null
}

tf_vars() {
  # destroy does not use the email, but the variable is validated, so give it a valid placeholder
  local email="${ALARM_EMAIL:-unused@example.com}"
  VARS=(-var "region=${AWS_REGION}" -var "mlflow_artifact_bucket=${MLFLOW_ARTIFACT_BUCKET}"
        -var "alarm_email=${email}")
  if [ -n "${IMAGE:-}" ]; then VARS+=(-var "container_image=${IMAGE}"); fi
}

# Minutes since the stack was created, or empty if there is no record.
age_minutes() {
  local created
  created=$(aws ssm get-parameter --name "$CREATED_PARAM" --query Parameter.Value --output text 2>/dev/null) || return 0
  echo $(( ( $(date -u +%s) - $(date -u -d "$created" +%s) ) / 60 ))
}

# Resources in the state even without a creation record (a failed first apply).
state_count() {
  tf_init
  tf state list 2>/dev/null | wc -l
}

case "${1:-}" in
  up)
    : "${ALARM_EMAIL:?ALARM_EMAIL is required for up}"
    tf_init; tf_vars
    tf apply -input=false -auto-approve -lock-timeout=15m "${VARS[@]}"
    ;;
  down)
    tf_init; tf_vars
    tf destroy -input=false -auto-approve -lock-timeout=15m "${VARS[@]}"
    summary "App stack destroyed."
    ;;
  status)
    age=$(age_minutes)
    if [ -n "$age" ]; then echo "up $age"; elif [ "$(state_count)" -gt 0 ]; then echo "partial"; else echo "down"; fi
    ;;
  check)
    age=$(age_minutes)
    if [ -z "$age" ]; then
      if [ "$(state_count)" -gt 0 ]; then
        summary "App stack has resources but no creation record (failed apply?); destroying."
        "$0" down
      else
        summary "App stack is down; nothing to do."
      fi
      exit 0
    fi
    limit=$(( MAX_UP_HOURS * 60 ))
    if [ "$KEEP_UP" = "true" ]; then
      summary "App stack up for ${age} min; AWS_KEEP_UP=true, so it stays up."
    elif [ "$age" -ge "$limit" ]; then
      summary "App stack up for ${age} min (limit ${MAX_UP_HOURS} h); destroying."
      "$0" down
    else
      summary "App stack up for ${age} min; auto-destroy after ${MAX_UP_HOURS} h (in $(( limit - age )) min)."
    fi
    ;;
  *)
    sed -n '2,7p' "$0"; exit 2
    ;;
esac
