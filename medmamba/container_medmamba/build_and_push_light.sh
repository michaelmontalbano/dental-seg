#!/bin/bash

# Set variables
ACCOUNT_ID=552401576656
REGION=us-west-2
REPO_NAME=medmamba
TAG=${1:-latest}  # Allow tag to be passed as argument, default to latest
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

echo "🔍 First, let's check Docker disk usage..."
docker system df

echo ""
echo "🏗️ Building lighter Docker image..."

# Build using regular docker build (not buildx) for simplicity
docker build -f Dockerfile.light -t ${REPO_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Create ECR repository if it doesn't exist
    echo "📦 Creating ECR repository if not exists..."
    aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}
    
    echo "🔑 Logging in to ECR..."
    aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    
    echo "🏷️ Tagging image as ${ECR_URI}..."
    docker tag ${REPO_NAME}:${TAG} ${ECR_URI}
    
    echo "🚀 Pushing image to ECR..."
    docker push ${ECR_URI}
    
    echo "✅ Done! Image available at: ${ECR_URI}"
    echo ""
    echo "📋 To use this image in SageMaker, run:"
    echo "   python ../sagemaker_launcher_medmamba.py --class-group <group-name>"
else
    echo "❌ Build failed! Try these steps:"
    echo "   1. Run: chmod +x fix_docker_space.sh && ./fix_docker_space.sh"
    echo "   2. Check Docker Desktop settings to increase disk allocation"
    echo "   3. Consider removing large unused images manually"
fi
