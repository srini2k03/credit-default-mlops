variable "region" {
  description = "AWS region (same as the bootstrap stack)."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix; must match the bootstrap stack (IAM and the deploy role are scoped to it)."
  type        = string
  default     = "credit-default"
}

variable "container_image" {
  description = <<-EOT
    API image. Workflows pass the ECR image tagged with the commit SHA. The default (the public
    GHCR image that ci.yml pushes from main) lets `make up` work without building anything.
  EOT
  type        = string
  default     = "ghcr.io/srini2k03/credit-default-mlops:latest"
}

variable "mlflow_artifact_bucket" {
  description = "Bootstrap output mlflow_artifact_bucket; the task reads models/production/model.joblib from it."
  type        = string
}

variable "alarm_email" {
  description = "Email subscribed to the alarm topic (confirm the subscription email)."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alarm_email))
    error_message = "alarm_email must be an email address."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "task_cpu" {
  description = "Fargate CPU units (512 = 0.5 vCPU)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate memory in MiB."
  type        = number
  default     = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "min_tasks" {
  type    = number
  default = 1
}

variable "max_tasks" {
  type    = number
  default = 3
}

variable "cpu_target_percent" {
  description = "Target-tracking goal for average service CPU."
  type        = number
  default     = 60
}

variable "requests_per_target" {
  description = "Target-tracking goal for ALB requests per task per minute."
  type        = number
  default     = 1000
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "latency_p95_threshold_seconds" {
  type    = number
  default = 0.5
}

variable "error_rate_threshold_percent" {
  type    = number
  default = 5
}
