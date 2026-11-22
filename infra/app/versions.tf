terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.12"
    }
  }

  # State in the bootstrap bucket with S3-native locking (no DynamoDB). Supply the bucket and
  # region at init time:
  #   terraform init -backend-config="bucket=$TF_STATE_BUCKET" -backend-config="region=us-east-1"
  backend "s3" {
    key          = "app/terraform.tfstate"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}
