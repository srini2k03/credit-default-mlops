data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  # Bucket names are global: account id + region keep them unique and predictable.
  suffix = "${local.account_id}-${var.region}"

  state_bucket    = "${var.project}-tfstate-${local.suffix}"
  artifact_bucket = "${var.project}-mlflow-${local.suffix}"
  dvc_bucket      = "${var.project}-dvc-${local.suffix}"

  tags = {
    project    = var.project
    managed-by = "terraform"
    stack      = "bootstrap"
  }
}

# ---------------------------------------------------------------- Terraform state bucket

resource "aws_s3_bucket" "state" {
  bucket = local.state_bucket
  # Not force_destroy: `make nuke` empties it explicitly before destroying this stack.
  # checkov:skip=CKV_AWS_18:server access logs need a second bucket; private, single-user demo buckets
  # checkov:skip=CKV_AWS_144:cross-region replication doubles storage cost; versioning covers mistakes
  # checkov:skip=CKV_AWS_145:SSE-S3 instead of a customer-managed KMS key ($1/month)
  # checkov:skip=CKV2_AWS_62:no consumers for bucket event notifications
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-old-state-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# ---------------------------------------------------------------- data buckets

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.artifact_bucket
  force_destroy = true
  # checkov:skip=CKV_AWS_18:server access logs need a second bucket; private, single-user demo buckets
  # checkov:skip=CKV_AWS_144:cross-region replication doubles storage cost; versioning covers mistakes
  # checkov:skip=CKV_AWS_145:SSE-S3 instead of a customer-managed KMS key ($1/month)
  # checkov:skip=CKV2_AWS_62:no consumers for bucket event notifications
}

resource "aws_s3_bucket" "dvc" {
  bucket        = local.dvc_bucket
  force_destroy = true
  # checkov:skip=CKV_AWS_18:server access logs need a second bucket; private, single-user demo buckets
  # checkov:skip=CKV_AWS_144:cross-region replication doubles storage cost; versioning covers mistakes
  # checkov:skip=CKV_AWS_145:SSE-S3 instead of a customer-managed KMS key ($1/month)
  # checkov:skip=CKV2_AWS_62:no consumers for bucket event notifications
}

locals {
  data_buckets = {
    artifacts = aws_s3_bucket.artifacts.id
    dvc       = aws_s3_bucket.dvc.id
  }
  all_buckets = merge(local.data_buckets, { state = aws_s3_bucket.state.id })
}

resource "aws_s3_bucket_versioning" "data" {
  for_each = local.data_buckets
  bucket   = each.value
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  for_each = local.data_buckets
  bucket   = each.value

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  depends_on = [aws_s3_bucket_versioning.data]
}

# ---------------------------------------------------------------- shared bucket hardening

resource "aws_s3_bucket_server_side_encryption_configuration" "all" {
  for_each = local.all_buckets
  bucket   = each.value

  rule {
    # SSE-S3: free. A customer-managed KMS key would cost $1/month plus requests.
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "all" {
  for_each = local.all_buckets
  bucket   = each.value

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "all" {
  for_each = local.all_buckets
  bucket   = each.value
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

data "aws_iam_policy_document" "tls_only" {
  for_each = local.all_buckets

  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${each.value}",
      "arn:${data.aws_partition.current.partition}:s3:::${each.value}/*",
    ]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "tls_only" {
  for_each = local.all_buckets
  bucket   = each.value
  policy   = data.aws_iam_policy_document.tls_only[each.key].json

  depends_on = [aws_s3_bucket_public_access_block.all]
}

# ---------------------------------------------------------------- container registry

resource "aws_ecr_repository" "api" {
  name                 = "${var.project}-api"
  image_tag_mutability = "IMMUTABLE" # one tag per commit SHA; deploys skip the push if it exists
  force_delete         = true
  # checkov:skip=CKV_AWS_136:AES256 instead of a customer-managed KMS key ($1/month)

  image_scanning_configuration {
    scan_on_push = true # basic scanning: free
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------- budget (stays while the app is down)

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = format("%.2f", var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alarm_email]
  }
}
