#!/usr/bin/env bash
# Render build step. The normal path uses the CI-trained model committed by train.yml
# (models/bundled/model.joblib). If it is missing (train.yml has not run yet), train a quick
# fallback on the real UCI data during the build: no Optuna search and no quality gate, so it is
# a bootstrap only, and /model reports it.
set -euo pipefail
BUNDLE=models/bundled/model.joblib
if [ -f "$BUNDLE" ]; then
  echo "Using committed CI-trained model: $BUNDLE"
  exit 0
fi
echo "No committed model; training a fast fallback on the real data for this build"
pip install -e ".[train]"
python -m credit_default.data.download
python -m credit_default.pipeline.train train=fast --no-register
mkdir -p models/bundled
cp artifacts/run/model.joblib "$BUNDLE"
