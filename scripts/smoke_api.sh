#!/usr/bin/env bash
# Smoke test a running API: health, predict, predict_batch, explain, and a rejected payload.
set -euo pipefail
BASE="${1:-http://localhost:8000}"
P='{"LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":35,"PAY_0":2,"PAY_2":2,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":46990,"BILL_AMT2":48233,"BILL_AMT3":49291,"BILL_AMT4":28314,"BILL_AMT5":28959,"BILL_AMT6":29547,"PAY_AMT1":2000,"PAY_AMT2":2019,"PAY_AMT3":1200,"PAY_AMT4":1100,"PAY_AMT5":1069,"PAY_AMT6":1000}'
H='content-type: application/json'

echo "== GET /health";        curl -fsS "$BASE/health"; echo
echo "== POST /predict";      curl -fsS -X POST "$BASE/predict" -H "$H" -d "$P"; echo
echo "== POST /predict_batch"; curl -fsS -X POST "$BASE/predict_batch" -H "$H" -d "{\"applications\":[$P,$P]}"; echo
echo "== POST /explain";      curl -fsS -X POST "$BASE/explain?top_k=5" -H "$H" -d "$P"; echo
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" -H "$H" -d '{"LIMIT_BAL":-1}')
echo "== invalid payload -> HTTP $code"
[ "$code" = "422" ]
echo "smoke OK"
