#!/usr/bin/env bash
# Build the api image and push it to ECR as <ECR_REPOSITORY>:<tag>, unless that tag exists
# (tags are immutable). Prints the image reference on the last line.
# Needs ECR_REPOSITORY (repository URL) and a docker login to ECR.
set -euo pipefail
tag="${1:?usage: push_image.sh <tag>}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY is required}"
repo_name="${ECR_REPOSITORY#*/}"
image="${ECR_REPOSITORY}:${tag}"
if aws ecr describe-images --repository-name "$repo_name" --image-ids "imageTag=${tag}" >/dev/null 2>&1; then
  echo "image ${image} already in ECR" >&2
else
  docker build -f docker/Dockerfile --target api -t "$image" . >&2
  docker push "$image" >&2
fi
echo "$image"
