#!/bin/bash

# Set variables
# GITHUB_TOKEN should be set as an environment variable, not hardcoded!
# export GITHUB_TOKEN=your_token_here
ACCOUNT_ID=552401576656
REGION=us-west-2  # Updated to match SageMaker region
REPO_NAME=medmamba
TAG=${1:-latest}  # Allow tag to be passed as argument, default to latest
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

# Check if GITHUB_TOKEN is set (if needed for private repos)
# if [ -z "$GITHUB_TOKEN" ]; then
#     echo "⚠️  Warning: GITHUB_TOKEN not set. This may be needed for private repos."
# fi

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository if not exists..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}

# Ensure buildx is bootstrapped
docker buildx create --use --name medmamba-builder 2>/dev/null || echo "Builder exists"
docker buildx inspect medmamba-builder --bootstrap

echo "🔑 Logging in to ECR first..."
aws ecr get-login-password --region ${REGION} | \
docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "🛠️ Building and pushing image for linux/amd64 directly to ECR..."
echo "💡 This avoids the export error and uses cached layers"
docker buildx build --platform linux/amd64 -t ${ECR_URI} --push .

echo "✅ Done! Image available at: ${ECR_URI}"
echo ""
echo "� To use this image in SageMaker, run:"
echo "   python ../sagemaker_launcher_medmamba.py --class-group <group-name>"
echo ""
echo "🔐 Security Note: If you need a GitHub token for private repos,"
echo "   set it as an environment variable: export GITHUB_TOKEN=your_token_here"
