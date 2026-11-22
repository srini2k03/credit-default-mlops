terraform {
  # 1.10+ for S3-native state locking (use_lockfile) in the app stack that uses this bucket.
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # This stack creates the state bucket, so its own state stays local (in CloudShell's
  # persistent $HOME). `make infra-bootstrap` also copies it to the state bucket as a backup.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}
