data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
  # checkov:skip=CKV_AWS_394:only the first two zones are used (slice below), so new zones change nothing
}

locals {
  name           = "${var.project}-app"
  account_id     = data.aws_caller_identity.current.account_id
  partition      = data.aws_partition.current.partition
  container_name = "api"
  container_port = 8000
  azs            = slice(data.aws_availability_zones.available.names, 0, 2)
  model_s3_uri   = "s3://${var.mlflow_artifact_bucket}/models/production/model.joblib"
  # Created by infra/bootstrap; the deploy role may only create roles that carry it.
  boundary_arn = "arn:${local.partition}:iam::${local.account_id}:policy/${var.project}-app-boundary"
  # Read by the workflows (deploy, auto-destroy) without a Terraform init.
  ssm_stack_parameter = "/${var.project}/app/stack"

  tags = {
    project    = var.project
    managed-by = "terraform"
    stack      = "app"
  }
}

# Creation time of this stack, recorded once; the auto-destroy job compares it with the clock.
resource "time_static" "created" {}
