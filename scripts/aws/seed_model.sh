#!/usr/bin/env bash
# Upload models/bundled/model.joblib as the production model in S3 if none is there yet, so a
# fresh stack serves from S3 before train.yml has published anything.
set -euo pipefail
: "${MLFLOW_ARTIFACT_BUCKET:?MLFLOW_ARTIFACT_BUCKET is required}"
key="models/production/model.joblib"
if aws s3api head-object --bucket "$MLFLOW_ARTIFACT_BUCKET" --key "$key" >/dev/null 2>&1; then
  echo "s3://$MLFLOW_ARTIFACT_BUCKET/$key exists"
  exit 0
fi
aws s3 cp models/bundled/model.joblib "s3://$MLFLOW_ARTIFACT_BUCKET/$key" --only-show-errors
jq -n --arg uri "s3://$MLFLOW_ARTIFACT_BUCKET/$key" \
      --arg sha "$(sha256sum models/bundled/model.joblib | cut -d' ' -f1)" \
      --arg commit "${GITHUB_SHA:-$(git rev-parse HEAD)}" \
      --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{s3_uri: $uri, sha256: $sha, source: "seeded from models/bundled", published_commit: $commit, published_at: $at}' \
  | aws s3 cp - "s3://$MLFLOW_ARTIFACT_BUCKET/models/production/model.json" --content-type application/json --only-show-errors
echo "seeded s3://$MLFLOW_ARTIFACT_BUCKET/$key from models/bundled"
