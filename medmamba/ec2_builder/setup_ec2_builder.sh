#!/bin/bash

# EC2 Builder Setup Script for MedMamba Docker builds
# This script sets up an EC2 instance for building Docker images

set -e

# Configuration
INSTANCE_TYPE="t3.medium"
REGION="us-west-2"
KEY_NAME="medmamba-builder-key"
SECURITY_GROUP_NAME="medmamba-builder-sg"
INSTANCE_NAME="MedMamba-Docker-Builder"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up EC2 Docker Builder Instance${NC}"

# Check if key pair exists, create if not
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION 2>/dev/null; then
    echo -e "${GREEN}🔑 Creating new key pair: $KEY_NAME${NC}"
    aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION \
        --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo -e "${GREEN}✅ Key saved to ~/.ssh/${KEY_NAME}.pem${NC}"
else
    echo -e "${BLUE}🔑 Using existing key pair: $KEY_NAME${NC}"
fi

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text --region $REGION)

# Create security group if it doesn't exist
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query "SecurityGroups[0].GroupId" --output text --region $REGION 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo -e "${GREEN}🔒 Creating security group: $SECURITY_GROUP_NAME${NC}"
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for MedMamba Docker builder" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' --output text)
    
    # Allow SSH from your IP
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr ${MY_IP}/32 \
        --region $REGION
    
    echo -e "${GREEN}✅ Security group created with SSH access from ${MY_IP}${NC}"
else
    echo -e "${BLUE}🔒 Using existing security group: $SG_ID${NC}"
fi

# Get latest Amazon Linux 2 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region $REGION)

# Create user data script for instance initialization
cat > /tmp/ec2_user_data.sh << 'EOF'
#!/bin/bash
# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Git
yum install git -y

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Install CloudWatch agent for monitoring
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm
rm amazon-cloudwatch-agent.rpm

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CWCONFIG'
{
  "metrics": {
    "namespace": "MedMambaBuilder",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
CWCONFIG

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a query -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Create build directory
mkdir -p /home/ec2-user/builds
chown ec2-user:ec2-user /home/ec2-user/builds

# Add helper scripts
cat > /home/ec2-user/docker_status.sh << 'SCRIPT'
#!/bin/bash
echo "🐳 Docker Status:"
docker --version
echo ""
echo "📊 Docker System Info:"
docker system df
echo ""
echo "🏃 Running Containers:"
docker ps
echo ""
echo "💾 Disk Usage:"
df -h /
SCRIPT

chmod +x /home/ec2-user/docker_status.sh
chown ec2-user:ec2-user /home/ec2-user/docker_status.sh

# Configure Docker to use full disk space
cat > /etc/docker/daemon.json << 'DOCKERCONFIG'
{
  "data-root": "/home/ec2-user/docker",
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
DOCKERCONFIG

mkdir -p /home/ec2-user/docker
chown -R ec2-user:ec2-user /home/ec2-user/docker
systemctl restart docker

# Set up auto-shutdown after 2 hours of inactivity (optional, disabled by default)
# echo "shutdown -h +120" | at now

echo "EC2 Builder instance initialized successfully!" > /home/ec2-user/setup_complete.txt
EOF

# Launch instance
echo -e "${GREEN}🚀 Launching EC2 instance...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP_NAME \
    --user-data file:///tmp/ec2_user_data.sh \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Purpose,Value=DockerBuilder}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✅ Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo -e "${BLUE}⏳ Waiting for instance to start...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# Create connection script
cat > connect_to_builder.sh << EOF
#!/bin/bash
echo "🔗 Connecting to MedMamba Docker Builder..."
ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}
EOF
chmod +x connect_to_builder.sh

# Create instance management script
cat > manage_builder.sh << EOF
#!/bin/bash
INSTANCE_ID="$INSTANCE_ID"
REGION="$REGION"

case "\$1" in
    start)
        echo "▶️  Starting instance..."
        aws ec2 start-instances --instance-ids \$INSTANCE_ID --region \$REGION
        aws ec2 wait instance-running --instance-ids \$INSTANCE_ID --region \$REGION
        PUBLIC_IP=\$(aws ec2 describe-instances --instance-ids \$INSTANCE_ID --region \$REGION \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
        echo "✅ Instance started. IP: \$PUBLIC_IP"
        # Update connect script
        sed -i.bak "s/ec2-user@.*/ec2-user@\$PUBLIC_IP/" connect_to_builder.sh
        ;;
    stop)
        echo "⏹️  Stopping instance..."
        aws ec2 stop-instances --instance-ids \$INSTANCE_ID --region \$REGION
        aws ec2 wait instance-stopped --instance-ids \$INSTANCE_ID --region \$REGION
        echo "✅ Instance stopped"
        ;;
    status)
        STATE=\$(aws ec2 describe-instances --instance-ids \$INSTANCE_ID --region \$REGION \
            --query 'Reservations[0].Instances[0].State.Name' --output text)
        echo "📊 Instance state: \$STATE"
        if [ "\$STATE" == "running" ]; then
            PUBLIC_IP=\$(aws ec2 describe-instances --instance-ids \$INSTANCE_ID --region \$REGION \
                --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
            echo "🌐 Public IP: \$PUBLIC_IP"
        fi
        ;;
    terminate)
        read -p "⚠️  Are you sure you want to terminate the instance? (y/N) " -n 1 -r
        echo
        if [[ \$REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Terminating instance..."
            aws ec2 terminate-instances --instance-ids \$INSTANCE_ID --region \$REGION
            echo "✅ Instance terminated"
        fi
        ;;
    *)
        echo "Usage: \$0 {start|stop|status|terminate}"
        exit 1
        ;;
esac
EOF
chmod +x manage_builder.sh

# Save instance info
cat > instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Region: $REGION
Public IP: $PUBLIC_IP
Key Name: $KEY_NAME
Security Group: $SG_ID
Instance Type: $INSTANCE_TYPE

To connect: ./connect_to_builder.sh
To manage: ./manage_builder.sh {start|stop|status|terminate}

CloudWatch Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=MedMambaBuilder
EOF

echo -e "${GREEN}✅ EC2 Docker Builder Setup Complete!${NC}"
echo ""
echo -e "${BLUE}Instance Details:${NC}"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Region: $REGION"
echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "  1. Wait ~2 minutes for initialization to complete"
echo "  2. Connect: ./connect_to_builder.sh"
echo "  3. Check status: ./manage_builder.sh status"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "  - Stop instance when not in use: ./manage_builder.sh stop"
echo "  - View monitoring: Check CloudWatch dashboard"
echo "  - Instance has 100GB disk for Docker images"

# Clean up
rm /tmp/ec2_user_data.sh
