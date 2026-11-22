#!/usr/bin/env bash
# Delete every object version and delete marker in a versioned S3 bucket (used by `make nuke`
# before destroying the bootstrap stack). Needs the AWS CLI and jq (both preinstalled in CloudShell).
# Usage: scripts/aws/empty_bucket.sh <bucket>
set -euo pipefail
bucket="${1:?usage: empty_bucket.sh <bucket>}"
if ! aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
  echo "bucket $bucket not found; nothing to empty"; exit 0
fi
echo "emptying s3://$bucket (all versions)"
while :; do
  batch=$(aws s3api list-object-versions --bucket "$bucket" --max-items 1000 --output json \
    | jq -c '{Objects: ([(.Versions // [])[], (.DeleteMarkers // [])[]] | map({Key, VersionId})), Quiet: true}')
  [ "$(jq '.Objects | length' <<<"$batch")" -eq 0 ] && break
  aws s3api delete-objects --bucket "$bucket" --delete "$batch" >/dev/null
done
echo "s3://$bucket is empty"
