output "region" {
  value = var.region
}

output "deploy_role_arn" {
  description = "AWS_ROLE_ARN for GitHub Actions"
  value       = aws_iam_role.deploy.arn
}

output "state_bucket" {
  value = aws_s3_bucket.state.id
}

output "mlflow_artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "dvc_bucket" {
  value = aws_s3_bucket.dvc.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "app_permissions_boundary_arn" {
  value = aws_iam_policy.app_boundary.arn
}

output "github_variables" {
  description = "Repository variables to set (Settings > Secrets and variables > Actions > Variables)"
  value = {
    AWS_ROLE_ARN           = aws_iam_role.deploy.arn
    AWS_REGION             = var.region
    TF_STATE_BUCKET        = aws_s3_bucket.state.id
    ECR_REPOSITORY         = aws_ecr_repository.api.repository_url
    MLFLOW_ARTIFACT_BUCKET = aws_s3_bucket.artifacts.id
    DVC_BUCKET             = aws_s3_bucket.dvc.id
  }
}

output "gh_variable_commands" {
  description = "Copy-paste: set the repository variables with the GitHub CLI"
  value       = <<-EOT
    gh variable set AWS_ROLE_ARN           --repo ${var.github_repository} --body '${aws_iam_role.deploy.arn}'
    gh variable set AWS_REGION             --repo ${var.github_repository} --body '${var.region}'
    gh variable set TF_STATE_BUCKET        --repo ${var.github_repository} --body '${aws_s3_bucket.state.id}'
    gh variable set ECR_REPOSITORY         --repo ${var.github_repository} --body '${aws_ecr_repository.api.repository_url}'
    gh variable set MLFLOW_ARTIFACT_BUCKET --repo ${var.github_repository} --body '${aws_s3_bucket.artifacts.id}'
    gh variable set DVC_BUCKET             --repo ${var.github_repository} --body '${aws_s3_bucket.dvc.id}'
    gh variable set ALARM_EMAIL            --repo ${var.github_repository} --body '${var.alarm_email}'
  EOT
}
