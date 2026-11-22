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

## Teardown

Remove everything, including the bootstrap (run in CloudShell from the repository directory):

```bash
make nuke
```

`make nuke` destroys the app stack (if up), empties the versioned state bucket, then destroys
the bootstrap stack. The artifact and DVC buckets and the ECR repository are deleted with their
contents (`force_destroy` / `force_delete`). Afterwards delete the GitHub variables and the `aws`
environment. To close the account entirely: **Account > Close account** in the console.
