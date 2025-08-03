#!/bin/bash

# Remote Docker Build Script
# Builds Docker images on EC2 and pushes to ECR

set -e

# Configuration
KEY_PATH="~/.ssh/medmamba-builder-key.pem"
REPO_URL="https://github.com/yourusername/dental-seg.git"  # Update this!
BRANCH="main"
ECR_REGION="us-west-2"
ECR_REPO="medmamba"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Get instance info
if [ ! -f "instance_info.txt" ]; then
    echo -e "${RED}❌ Error: instance_info.txt not found. Run setup_ec2_builder.sh first.${NC}"
    exit 1
fi

INSTANCE_ID=$(grep "Instance ID:" instance_info.txt | awk '{print $3}')
REGION=$(grep "Region:" instance_info.txt | awk '{print $2}')

# Check if instance is running
STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION \
    --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "$STATE" != "running" ]; then
    echo -e "${BLUE}▶️  Starting instance...${NC}"
    ./manage_builder.sh start
    sleep 30  # Wait for SSH to be ready
fi

# Get current public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo -e "${GREEN}🔗 Connecting to EC2 Builder at $PUBLIC_IP${NC}"

# Create remote build script
cat > /tmp/ec2_build_commands.sh << 'BUILDSCRIPT'
#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

cd ~/builds

# Clone or update repository
if [ ! -d "dental-seg" ]; then
    echo -e "${BLUE}📥 Cloning repository...${NC}"
    git clone REPO_URL_PLACEHOLDER dental-seg
else
    echo -e "${BLUE}🔄 Updating repository...${NC}"
    cd dental-seg
    git fetch origin
    git checkout BRANCH_PLACEHOLDER
    git pull origin BRANCH_PLACEHOLDER
    cd ..
fi

cd dental-seg/medmamba/container_medmamba

# Get ECR login
echo -e "${BLUE}🔑 Logging into ECR...${NC}"
aws ecr get-login-password --region ECR_REGION_PLACEHOLDER | \
    docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.ECR_REGION_PLACEHOLDER.amazonaws.com

# Build the image
echo -e "${GREEN}🏗️  Building Docker image...${NC}"
START_TIME=$(date +%s)

# Use regular docker build (no buildx needed on native x86_64)
docker build -t ECR_REPO_PLACEHOLDER:latest .

END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))
echo -e "${GREEN}✅ Build completed in ${BUILD_TIME} seconds${NC}"

# Tag and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.ECR_REGION_PLACEHOLDER.amazonaws.com/ECR_REPO_PLACEHOLDER:latest"

echo -e "${BLUE}🏷️  Tagging image...${NC}"
docker tag ECR_REPO_PLACEHOLDER:latest $ECR_URI

echo -e "${BLUE}🚀 Pushing to ECR...${NC}"
docker push $ECR_URI

echo -e "${GREEN}✅ Image available at: $ECR_URI${NC}"

# Show docker status
echo ""
~/docker_status.sh
BUILDSCRIPT

# Replace placeholders
sed -i.bak "s|REPO_URL_PLACEHOLDER|$REPO_URL|g" /tmp/ec2_build_commands.sh
sed -i.bak "s|BRANCH_PLACEHOLDER|$BRANCH|g" /tmp/ec2_build_commands.sh
sed -i.bak "s|ECR_REGION_PLACEHOLDER|$ECR_REGION|g" /tmp/ec2_build_commands.sh
sed -i.bak "s|ECR_REPO_PLACEHOLDER|$ECR_REPO|g" /tmp/ec2_build_commands.sh

# Copy and execute script on EC2
echo -e "${BLUE}📤 Uploading build script...${NC}"
scp -i $KEY_PATH -o StrictHostKeyChecking=no /tmp/ec2_build_commands.sh ec2-user@$PUBLIC_IP:~/build_medmamba.sh

echo -e "${GREEN}🚀 Starting remote build...${NC}"
ssh -i $KEY_PATH -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP "chmod +x ~/build_medmamba.sh && ~/build_medmamba.sh"

# Clean up
rm /tmp/ec2_build_commands.sh*

echo -e "${GREEN}✅ Remote build complete!${NC}"
echo ""
echo -e "${BLUE}📝 Next steps:${NC}"
echo "  1. Run training: python ../sagemaker_launcher_medmamba.py --class-group bone-loss"
echo "  2. Stop EC2 instance: ./manage_builder.sh stop"
echo ""
echo -e "${BLUE}💡 To connect to the instance:${NC}"
echo "  ./connect_to_builder.sh"
