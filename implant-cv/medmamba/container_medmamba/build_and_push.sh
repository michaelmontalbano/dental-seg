#!/bin/bash

# Set variables
GITHUB_TOKEN="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"
ACCOUNT_ID=552401576656
REGION=us-east-1
REPO_NAME=medmamba
TAG=latest
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

# Ensure buildx is bootstrapped
docker buildx create --use --name medmamba-builder 2>/dev/null || echo "Builder exists"
docker buildx inspect medmamba-builder --bootstrap

echo "🛠️ Building image for linux/amd64..."
docker buildx build --platform linux/amd64 -t medmamba:latest --load .

echo "🔑 Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | \
docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "🏷️ Tagging image as ${ECR_URI}..."
docker tag ${REPO_NAME}:latest ${ECR_URI}

echo "🚀 Pushing image to ECR..."
docker push ${ECR_URI}

echo "✅ Done! Image available at: ${ECR_URI}"

