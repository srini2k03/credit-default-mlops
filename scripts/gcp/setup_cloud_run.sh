#!/usr/bin/env bash
# One-time Google Cloud setup for deploying the credit-default API to Cloud Run from GitHub Actions.
#
# Run it in Cloud Shell (console.cloud.google.com -> the ">_" icon), after creating a project:
#   bash setup_cloud_run.sh YOUR_PROJECT_ID
#
# It creates: an Artifact Registry Docker repository (keeping only the 3 newest images), a runtime
# service account with no roles, a deployer service account, and a Workload Identity Federation
# pool that only this GitHub repository can use. No keys are created. It is safe to re-run.
set -euo pipefail

PROJECT_ID=${1:?usage: bash setup_cloud_run.sh PROJECT_ID}
REGION=${REGION:-asia-south1}
GITHUB_REPO=srini2k03/credit-default-mlops
REPOSITORY=credit-default
POOL=github
PROVIDER=github-actions

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
DEPLOYER_SA="github-deployer@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME_SA="credit-api-runtime@$PROJECT_ID.iam.gserviceaccount.com"

echo "Enabling APIs (takes a minute)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com

echo "Artifact Registry repository"
gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker \
    --location="$REGION" --description="credit-default API images"
cat > /tmp/cleanup-policy.json <<'JSON'
[
  {"name": "keep-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any", "olderThan": "1d"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$REPOSITORY" --location="$REGION" \
  --policy=/tmp/cleanup-policy.json --no-dry-run >/dev/null

echo "Service accounts"
gcloud iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create credit-api-runtime --display-name="credit-default API runtime (no roles)"
gcloud iam service-accounts describe "$DEPLOYER_SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$DEPLOYER_SA" \
    --role="$role" --condition=None >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:$DEPLOYER_SA" --role=roles/iam.serviceAccountUser >/dev/null

echo "Workload Identity Federation (GitHub OIDC, restricted to $GITHUB_REPO)"
gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
  --workload-identity-pool="$POOL" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --display-name="GitHub Actions" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='$GITHUB_REPO'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$GITHUB_REPO" >/dev/null

cat <<OUT

Done. Add these three repository VARIABLES (not secrets) on GitHub:
  https://github.com/$GITHUB_REPO/settings/variables/actions/new

  GCP_PROJECT_ID       = $PROJECT_ID
  GCP_WIF_PROVIDER     = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER
  GCP_SERVICE_ACCOUNT  = $DEPLOYER_SA

Then run the "gcp-deploy" workflow (Actions tab -> gcp-deploy -> Run workflow), or let the next
change to the API deploy it.
OUT
