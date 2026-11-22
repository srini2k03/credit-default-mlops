# GitHub Actions -> AWS through OIDC: no long-lived access keys anywhere.

locals {
  partition = data.aws_partition.current.partition
  oidc_host = "token.actions.githubusercontent.com"
  oidc_provider_arn = (
    var.create_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
    : "arn:${local.partition}:iam::${local.account_id}:oidc-provider/${local.oidc_host}"
  )

  # Names the app stack (infra/app) uses; the deploy role is scoped to them.
  app_role_prefix   = "${var.project}-app-"
  boundary_name     = "${var.project}-app-boundary"
  state_key_prefix  = "app/"
  ssm_prefix        = "/${var.project}/"
  region_condition  = { test = "StringEquals", variable = "aws:RequestedRegion", values = [var.region] }
  arn_prefix_region = "arn:${local.partition}:%s:${var.region}:${local.account_id}"
}

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_oidc_provider ? 1 : 0
  url            = "https://${local.oidc_host}"
  client_id_list = ["sts.amazonaws.com"]
  # AWS verifies GitHub's OIDC endpoint against its trusted CA store, so no thumbprint is needed.
}

data "aws_iam_policy_document" "deploy_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only this repository: jobs on main (train.yml, deploy on main) or jobs that run in the
    # GitHub environment "aws" (infra up/down, deploys requested from a branch). Add required
    # reviewers or branch rules to that environment in GitHub to gate it further.
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values = [
        "repo:${var.github_repository}:ref:refs/heads/main",
        "repo:${var.github_repository}:environment:${var.github_environment}",
      ]
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.project}-github-deploy"
  description          = "Assumed by GitHub Actions (OIDC) for ${var.github_repository}"
  assume_role_policy   = data.aws_iam_policy_document.deploy_trust.json
  max_session_duration = 7200 # train.yml can run up to 2 hours
}

# ---------------------------------------------------------------- boundary for roles the app stack creates

# Every role the deploy role creates must carry this boundary, so a compromised workflow cannot
# mint a more powerful role and pass it to ECS.
data "aws_iam_policy_document" "app_boundary" {
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:/ecs/${var.project}*"]
  }
  statement {
    sid       = "ReadModels"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/models/*"]
  }
}

resource "aws_iam_policy" "app_boundary" {
  name        = local.boundary_name
  description = "Permissions boundary for ECS task and execution roles of the app stack"
  policy      = data.aws_iam_policy_document.app_boundary.json
}

# ---------------------------------------------------------------- deploy role permissions

data "aws_iam_policy_document" "deploy" {
  # --- Terraform remote state for the app stack (S3-native locking: the .tflock object)
  statement {
    sid       = "StateList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.state.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.state_key_prefix}*"]
    }
  }
  statement {
    sid       = "StateObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.state.arn}/${local.state_key_prefix}*"]
  }

  # --- ECR: push the api image
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
      "ecr:DescribeImages", "ecr:DescribeRepositories", "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload", "ecr:ListImages", "ecr:PutImage", "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # --- S3: MLflow artifacts, registry db, production model, DVC remote
  statement {
    sid       = "DataBucketsList"
    actions   = ["s3:ListBucket", "s3:ListBucketVersions", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.artifacts.arn, aws_s3_bucket.dvc.arn]
  }
  statement {
    sid     = "DataObjects"
    actions = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "${aws_s3_bucket.artifacts.arn}/mlflow/*",
      "${aws_s3_bucket.artifacts.arn}/registry/*",
      "${aws_s3_bucket.artifacts.arn}/models/*",
      "${aws_s3_bucket.dvc.arn}/dvc/*",
    ]
  }

  # --- App stack resources (Terraform apply/destroy, ECS deploys), pinned to one region.
  # These services have little resource-level support for create calls, so they are scoped by
  # region; IAM, logs, alarms, SNS and SSM below are scoped by name.
  statement {
    sid = "Network"
    actions = [
      "ec2:Describe*",
      "ec2:CreateVpc", "ec2:DeleteVpc", "ec2:ModifyVpcAttribute",
      "ec2:CreateSubnet", "ec2:DeleteSubnet", "ec2:ModifySubnetAttribute",
      "ec2:CreateInternetGateway", "ec2:DeleteInternetGateway",
      "ec2:AttachInternetGateway", "ec2:DetachInternetGateway",
      "ec2:CreateRouteTable", "ec2:DeleteRouteTable", "ec2:CreateRoute", "ec2:DeleteRoute",
      "ec2:ReplaceRoute", "ec2:AssociateRouteTable", "ec2:DisassociateRouteTable",
      "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup",
      "ec2:AuthorizeSecurityGroupIngress", "ec2:AuthorizeSecurityGroupEgress",
      "ec2:RevokeSecurityGroupIngress", "ec2:RevokeSecurityGroupEgress",
      "ec2:ModifySecurityGroupRules", "ec2:UpdateSecurityGroupRuleDescriptionsIngress",
      "ec2:UpdateSecurityGroupRuleDescriptionsEgress",
      "ec2:CreateTags", "ec2:DeleteTags",
    ]
    resources = ["*"]
    condition {
      test     = local.region_condition.test
      variable = local.region_condition.variable
      values   = local.region_condition.values
    }
  }
  statement {
    sid       = "LoadBalancingEcsAutoscaling"
    actions   = ["elasticloadbalancing:*", "ecs:*", "application-autoscaling:*"]
    resources = ["*"]
    condition {
      test     = local.region_condition.test
      variable = local.region_condition.variable
      values   = local.region_condition.values
    }
  }
  statement {
    sid       = "ServiceLinkedRoles"
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "iam:AWSServiceName"
      values = [
        "ecs.amazonaws.com",
        "elasticloadbalancing.amazonaws.com",
        "ecs.application-autoscaling.amazonaws.com",
      ]
    }
  }

  # --- IAM: only roles named <project>-app-*, only with the boundary, PassRole only to ECS tasks
  statement {
    sid       = "AppRolesCreateWithBoundary"
    actions   = ["iam:CreateRole", "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:AttachRolePolicy", "iam:DetachRolePolicy"]
    resources = ["arn:${local.partition}:iam::${local.account_id}:role/${local.app_role_prefix}*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PermissionsBoundary"
      values   = [aws_iam_policy.app_boundary.arn]
    }
  }
  statement {
    sid = "AppRolesManage"
    actions = [
      "iam:GetRole", "iam:DeleteRole", "iam:UpdateRole", "iam:UpdateAssumeRolePolicy",
      "iam:TagRole", "iam:UntagRole", "iam:GetRolePolicy", "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies", "iam:ListInstanceProfilesForRole",
    ]
    resources = ["arn:${local.partition}:iam::${local.account_id}:role/${local.app_role_prefix}*"]
  }
  statement {
    sid       = "PassAppTaskRoles"
    actions   = ["iam:PassRole"]
    resources = ["arn:${local.partition}:iam::${local.account_id}:role/${local.app_role_prefix}task*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
  statement {
    sid       = "ReadBoundary"
    actions   = ["iam:GetPolicy", "iam:GetPolicyVersion"]
    resources = [aws_iam_policy.app_boundary.arn]
  }

  # --- Observability and bookkeeping, scoped by name
  statement {
    sid = "LogGroups"
    actions = [
      "logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy",
      "logs:DeleteRetentionPolicy", "logs:TagResource", "logs:UntagResource",
      "logs:ListTagsForResource", "logs:TagLogGroup", "logs:ListTagsLogGroup",
    ]
    resources = [
      format("${local.arn_prefix_region}:log-group:/ecs/${var.project}*", "logs"),
    ]
  }
  statement {
    sid       = "LogsDescribe"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["*"]
  }
  statement {
    sid = "Alarms"
    actions = [
      "cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms", "cloudwatch:TagResource",
      "cloudwatch:UntagResource", "cloudwatch:ListTagsForResource",
    ]
    resources = [format("${local.arn_prefix_region}:alarm:${var.project}-*", "cloudwatch")]
  }
  statement {
    sid       = "AlarmsDescribe"
    actions   = ["cloudwatch:DescribeAlarms"]
    resources = ["*"]
  }
  statement {
    sid = "AlertTopic"
    actions = [
      "sns:CreateTopic", "sns:DeleteTopic", "sns:GetTopicAttributes", "sns:SetTopicAttributes",
      "sns:Subscribe", "sns:Unsubscribe", "sns:GetSubscriptionAttributes",
      "sns:ListSubscriptionsByTopic", "sns:TagResource", "sns:UntagResource",
      "sns:ListTagsForResource",
    ]
    # topic ARN and its subscription ARNs (<topic arn>:<id>)
    resources = [format("${local.arn_prefix_region}:${var.project}-*", "sns")]
  }
  statement {
    sid = "StackParameters"
    actions = [
      "ssm:PutParameter", "ssm:GetParameter", "ssm:GetParameters", "ssm:DeleteParameter",
      "ssm:AddTagsToResource", "ssm:RemoveTagsFromResource", "ssm:ListTagsForResource",
    ]
    resources = [format("${local.arn_prefix_region}:parameter${local.ssm_prefix}*", "ssm")]
  }
  statement {
    sid       = "ParametersDescribe"
    actions   = ["ssm:DescribeParameters"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
