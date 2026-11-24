#!/usr/bin/env bash
# Smoke test a running API: health, readiness, metadata, predict (+ cutoff parameters),
# predict_batch, explain, request-ID echo and a rejected payload.
set -euo pipefail
BASE="${1:-http://localhost:8000}"
P='{"LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":35,"PAY_0":2,"PAY_2":2,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":46990,"BILL_AMT2":48233,"BILL_AMT3":49291,"BILL_AMT4":28314,"BILL_AMT5":28959,"BILL_AMT6":29547,"PAY_AMT1":2000,"PAY_AMT2":2019,"PAY_AMT3":1200,"PAY_AMT4":1100,"PAY_AMT5":1069,"PAY_AMT6":1000}'
H='content-type: application/json'

echo "== GET /health";        curl -fsS "$BASE/health"; echo
echo "== GET /ready";         curl -fsS "$BASE/ready"; echo
echo "== GET /model";         curl -fsS "$BASE/model" | head -c 400; echo
echo "== POST /predict";      curl -fsS -X POST "$BASE/predict" -H "$H" -d "$P"; echo
echo "== POST /predict?fn_cost=8&fp_cost=1"
curl -fsS -X POST "$BASE/predict?fn_cost=8&fp_cost=1" -H "$H" -d "$P" | grep -q '"threshold_source":"request costs"'
echo "== POST /predict_batch"; curl -fsS -X POST "$BASE/predict_batch" -H "$H" -d "{\"applications\":[$P,$P]}"; echo
echo "== POST /explain";      curl -fsS -X POST "$BASE/explain?top_k=5" -H "$H" -d "$P"; echo
rid=$(curl -fsS -o /dev/null -D - -X POST "$BASE/predict" -H "$H" -H 'x-request-id: smoke-test-1' -d "$P" \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="x-request-id"{print $2}')
echo "== X-Request-ID echoed: $rid"
[ "$rid" = "smoke-test-1" ]
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" -H "$H" -d '{"LIMIT_BAL":-1}')
echo "== invalid payload -> HTTP $code"
[ "$code" = "422" ]
echo "smoke OK"
