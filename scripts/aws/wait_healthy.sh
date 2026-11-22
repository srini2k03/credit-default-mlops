#!/usr/bin/env bash
# Poll <base>/health until it returns 200 (new ALB DNS names can take a few minutes to resolve).
set -euo pipefail
base="${1:?usage: wait_healthy.sh <base url> [attempts]}"
attempts="${2:-40}"
for i in $(seq 1 "$attempts"); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$base/health" || true)
  echo "attempt $i: HTTP $code"
  [ "$code" = "200" ] && { curl -fsS "$base/health"; echo; exit 0; }
  sleep 15
done
echo "::error::$base/health did not return 200"
exit 1
