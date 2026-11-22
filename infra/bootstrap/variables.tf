variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "credit-default"
}

variable "github_repository" {
  description = "owner/repo allowed to assume the deploy role through GitHub OIDC."
  type        = string
  default     = "srini2k03/credit-default-mlops"
}

variable "github_environment" {
  description = "GitHub environment whose jobs may assume the deploy role (besides refs/heads/main)."
  type        = string
  default     = "aws"
}

variable "create_oidc_provider" {
  description = "Create the GitHub OIDC provider. Set false if the account already has one."
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Email address for budget alerts (confirm the subscription email AWS sends)."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alarm_email))
    error_message = "alarm_email must be an email address."
  }
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget; alerts at 50%, 100% (actual) and 100% (forecast)."
  type        = number
  default     = 1
}

variable "noncurrent_version_days" {
  description = "Days to keep noncurrent object versions in the artifact and DVC buckets."
  type        = number
  default     = 30
}
