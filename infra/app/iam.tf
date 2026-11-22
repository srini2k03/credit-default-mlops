# Both roles carry the bootstrap permissions boundary and are named <project>-app-task*, the only
# names the deploy role may create and pass to ECS.

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

# Execution role: used by the ECS agent to pull the image and write logs.
resource "aws_iam_role" "execution" {
  name                 = "${local.name}-task-exec"
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_trust.json
  permissions_boundary = local.boundary_arn
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["arn:${local.partition}:ecr:${var.region}:${local.account_id}:repository/${var.project}-api"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "pull-and-log"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

# Task role: what the application itself may do. Only read the production model.
resource "aws_iam_role" "task" {
  name                 = "${local.name}-task"
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_trust.json
  permissions_boundary = local.boundary_arn
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ReadProductionModel"
    actions   = ["s3:GetObject"]
    resources = ["arn:${local.partition}:s3:::${var.mlflow_artifact_bucket}/models/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "read-models"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}
