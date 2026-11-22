# AWS setup (on-demand ECS Fargate)

Status: **written but not yet applied.** The AWS account does not exist yet, so none of this has
run against AWS. Every AWS workflow in this repository skips with a notice until the repository
variables below are set, so CI stays green without an account.

Everything is on demand: the one-time bootstrap holds only things that cost nothing or almost
nothing while idle (S3 buckets, an empty ECR repository, an IAM role, a budget). The app stack
(ALB + Fargate, about $0.07/hour) is created with `make up` or the `aws-infra` workflow and is
destroyed automatically after a few hours (see [Auto-destroy](#auto-destroy)).

## 1. Create the account

1. Sign up at <https://aws.amazon.com/> (choose the free plan with credits for a new account).
2. Sign in to the console as the root user once, turn on MFA for root, then create an admin user
   in IAM Identity Center (or an IAM user with `AdministratorAccess`) and use that from now on.
3. Pick the region **us-east-1** (top right of the console). Everything below assumes it.

## 2. Bootstrap from AWS CloudShell (once)

Open CloudShell (the `>_` icon in the console header). `$HOME` in CloudShell persists between
sessions (1 GB), which is where the bootstrap's local Terraform state lives.

```bash
# Terraform into ~/bin (persists across CloudShell sessions)
TF_VERSION=1.17.2
ARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')
mkdir -p ~/bin && cd /tmp
curl -fsSLO "https://releases.hashicorp.com/terraform/${TF_VERSION}/terraform_${TF_VERSION}_linux_${ARCH}.zip"
unzip -o "terraform_${TF_VERSION}_linux_${ARCH}.zip" -d ~/bin && export PATH=~/bin:$PATH
terraform version

# The repository (public, so no credentials needed to clone)
cd ~ && git clone https://github.com/srini2k03/credit-default-mlops.git && cd credit-default-mlops

# Settings: region and the email for budget and alarm notifications
cp infra/bootstrap/terraform.tfvars.example infra/bootstrap/terraform.tfvars
sed -i 's/REPLACE_ME@example.com/you@example.com/' infra/bootstrap/terraform.tfvars   # your email

# Apply (shows the plan and asks for "yes"), back up the state, print the GitHub variables
make infra-bootstrap
```

The bootstrap creates:

| Resource | Purpose | Idle cost |
|---|---|---|
| S3 `credit-default-tfstate-<account>-<region>` | Terraform state of the app stack (versioned, encrypted, public access blocked, S3-native locking) | pennies |
| S3 `credit-default-mlflow-<account>-<region>` | MLflow artifacts (`mlflow/`), registry db (`registry/mlflow.db`), production model (`models/production/`) | pennies |
| S3 `credit-default-dvc-<account>-<region>` | DVC remote (`dvc/`) | pennies |
| ECR `credit-default-api` | API images, scan on push, keeps the last 10 | a few cents per GB-month |
| IAM OIDC provider + role `credit-default-github-deploy` | GitHub Actions sign-in without access keys | free |
| IAM policy `credit-default-app-boundary` | permissions boundary for the roles the app stack creates | free |
| AWS Budgets `credit-default-monthly` ($1/month) | emails at 50% and 100% actual, 100% forecast | free (first two budgets) |

Confirm the email AWS Budgets and SNS send you, or you will not receive alerts.

### Deploy role: what GitHub can do

The role trusts only `repo:srini2k03/credit-default-mlops` on `refs/heads/main` or in the GitHub
environment `aws`. Its policy allows: the app stack's Terraform state (`app/` prefix), pushing
to the one ECR repository, the `mlflow/`, `registry/`, `models/` and `dvc/` prefixes of the two
data buckets, creating and destroying the app stack in one region (VPC, ALB, ECS, autoscaling),
IAM roles named `credit-default-app-*` only when they carry the permissions boundary,
`iam:PassRole` on the task roles to ECS only, and the stack's log group, alarms, SNS topic and SSM
parameters by name.

## 3. Connect GitHub

`make infra-bootstrap` prints `gh variable set ...` commands. Run them on a machine with the
GitHub CLI logged in, or set the values by hand under **Settings > Secrets and variables >
Actions > Variables**:

| Variable | Value |
|---|---|
| `AWS_ROLE_ARN` | output `deploy_role_arn` |
| `AWS_REGION` | `us-east-1` |
| `TF_STATE_BUCKET` | output `state_bucket` |
| `ECR_REPOSITORY` | output `ecr_repository_url` |
| `MLFLOW_ARTIFACT_BUCKET` | output `mlflow_artifact_bucket` |
| `DVC_BUCKET` | output `dvc_bucket` |
| `ALARM_EMAIL` | your email (SNS alarm subscription of the app stack) |

Optional: `AWS_MAX_UP_HOURS` (default 4) and `AWS_KEEP_UP` (`true` disables auto-destroy).

Then create the environment: **Settings > Environments > New environment > `aws`**. Adding
yourself as a required reviewer makes every infra change and branch deploy wait for a click.

No secrets are needed: workflows get short-lived credentials through OIDC.

## 4. MLflow and DVC on S3

Once the variables exist, `train.yml` (on main) switches to S3 by itself. Without them it runs as
before: local DVC remote cached in Actions, an empty MLflow registry recreated from the committed
bundle each run.

| What | Where | How |
|---|---|---|
| DVC remote | `s3://<DVC_BUCKET>/dvc` | the `s3` remote in `.dvc/config` holds a placeholder URL; CI writes the real bucket and makes it the default in `.dvc/config.local` (`local` stays the committed default, so forks keep working) |
| MLflow artifacts | `s3://<MLFLOW_ARTIFACT_BUCKET>/mlflow` | `MLFLOW_ARTIFACT_LOCATION`, read when the experiment is first created |
| MLflow registry | `s3://<MLFLOW_ARTIFACT_BUCKET>/registry/mlflow.db` | SQLite backend kept; downloaded before training, uploaded after the gate (also after a rejection). The bucket is versioned, so earlier registry states can be restored |
| Production model | `s3://<MLFLOW_ARTIFACT_BUCKET>/models/production/model.joblib` + `model.json` | uploaded after a promotion; `model.json` records the registry version, commit, sha256 and run. The ECS task loads it with `MODEL_SOURCE=s3` |

The train job uses `role-duration-seconds: 7200` (the role allows 2 hours) and refreshes the
credentials before the S3 uploads at the end.

To use the S3 remote from a laptop with AWS credentials:

```bash
pip install "dvc-s3>=3.2"
dvc remote modify --local s3 url s3://<DVC_BUCKET>/dvc
dvc pull -r s3
```

## 5. The app stack (on demand)

`infra/app` (state: `s3://<TF_STATE_BUCKET>/app/terraform.tfstate`, S3-native lock file):

- VPC with two public subnets in two AZs, no NAT gateway; the default security group is emptied.
- Internet-facing ALB on HTTP :80, target group health-checking `/health` every 15 s,
  20 s deregistration delay (matches uvicorn's graceful shutdown).
- ECS cluster + Fargate service: 0.5 vCPU / 1 GB, desired 1, autoscaling 1-3 tasks on CPU 60% and
  ALB requests per target (1000/min), deployment circuit breaker with rollback. Tasks run in the
  public subnets with a public IP; their security group accepts only the ALB, egress only 443.
- Task environment: `LOG_FORMAT=json`, `MODEL_SOURCE=s3`,
  `MODEL_S3_URI=s3://<MLFLOW_ARTIFACT_BUCKET>/models/production/model.joblib`,
  `PREDICTION_LOG_PATH=/tmp/predictions.sqlite`. If the object is missing the task serves the
  model baked into the image and says so in its logs and in `/health`.
- IAM: execution role (pull from ECR, write logs) and task role (`s3:GetObject` on `models/*`
  only), both with the bootstrap permissions boundary.
- CloudWatch log group `/ecs/credit-default-api` (7 days) via the awslogs driver; alarms on 5XX
  count (>= 5 in 5 min), 5XX rate (> 5%), p95 target response time (> 500 ms in 3 of 5 minutes)
  and unhealthy hosts, all to an SNS topic with an email subscription.
- An SSM parameter `/credit-default/app/stack` (standard tier, free) with the creation time, ALB
  URL and service names, read by the deploy and auto-destroy workflows.

The $1 budget lives in the bootstrap stack instead, so it keeps watching while the app is down.

From CloudShell (values default to the bootstrap outputs):

```bash
make up          # ~5 minutes; prints the API URL
make infra-plan  # preview changes
make destroy     # remove the app stack (buckets, ECR and the deploy role stay)
```

`make up` uses the public GHCR image from `main` unless you pass `IMAGE=...`; the workflows always
deploy the ECR image of the commit.

**HTTPS** needs a domain name: request an ACM certificate (free) for it, add an HTTPS listener
with the certificate on the ALB and redirect :80 to :443, then point the domain at the ALB.
Without a domain the stack stays on HTTP.

## 6. Workflows: up, deploy, down

All AWS workflows use OIDC (`aws-actions/configure-aws-credentials`, `id-token: write`) and skip
with a `::notice::` and a green check when the repository variables are missing.

| Workflow | Trigger | What it does |
|---|---|---|
| `aws-infra.yml` | dispatch (`action=up/down`), or push a change to `.github/aws-infra-request.json` (`{"action": "up"}`) on any branch | **up**: build and push the image of the commit to ECR (skipped if the tag exists), seed `models/production/model.joblib` from `models/bundled/` if S3 has none, `terraform apply`, wait for `/health` through the ALB and run the smoke test. **down**: `terraform destroy` |
| `aws-infra.yml` (schedule) | every 3 hours | auto-destroy check (below) |
| `deploy-aws.yml` | after a green `ci` run on main, dispatch, or a change to `.github/aws-deploy-request.json` | if the stack is up: build/push the image, register a task definition with it, update the service and wait until it is stable (circuit breaker rolls back failures), smoke-test `/health` and `/predict` through the ALB. Otherwise a notice |
| `live-smoke.yml` | weekly, dispatch | smoke-tests Render as before; with the `aws_url` input it also tests the ALB |
| `infra-ci.yml` | push or PR touching `infra/**` | `terraform fmt`, `validate` of both stacks, checkov; no credentials |

The jobs that touch AWS run in the GitHub environment `aws` (the deploy role trusts it), so
required reviewers on that environment gate them. `train.yml` runs on main and is trusted by
branch instead.

### Auto-destroy

`terraform apply` writes the creation time to the SSM parameter `/credit-default/app/created_at`
(early in the apply, so even a failed apply leaves a record). The scheduled `aws-infra` run reads
it every 3 hours and destroys the stack once it is older than `AWS_MAX_UP_HOURS` (whole hours,
default 4), unless `AWS_KEEP_UP=true`. A stack with resources in its state but no record (a
failed first apply) is destroyed too.

Scheduled workflows only run from the default branch. Until this work is merged to main, an
`up` started from another branch also starts a `timer` job that sleeps `AWS_MAX_UP_HOURS` (capped
at 5.5 hours, the job limit) and then runs the same check; with a limit above 5 hours it destroys
at the cap. Worst case with the defaults: the stack lives about 4-7 hours, roughly $0.30-$0.50.

## Teardown

Remove everything, including the bootstrap (run in CloudShell from the repository directory):

```bash
make nuke
```

`make nuke` destroys the app stack (if up), empties the versioned state bucket, then destroys
the bootstrap stack. The artifact and DVC buckets and the ECR repository are deleted with their
contents (`force_destroy` / `force_delete`). Afterwards delete the GitHub variables and the `aws`
environment. To close the account entirely: **Account > Close account** in the console.
