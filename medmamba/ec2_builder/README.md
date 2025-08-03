# EC2 Docker Builder for MedMamba

This setup provides a fast, reliable way to build Docker images on EC2 (native x86_64) instead of dealing with cross-platform builds on Mac.

## 🚀 Quick Start

1. **Set up the EC2 instance** (one-time):
```bash
cd ec2_builder
chmod +x setup_ec2_builder.sh
./setup_ec2_builder.sh
```

This will:
- Create an EC2 instance (t3.medium, 100GB disk)
- Install Docker, Git, AWS CLI
- Set up CloudWatch monitoring
- Create SSH key and save to `~/.ssh/medmamba-builder-key.pem`
- Generate helper scripts for managing the instance

2. **Update the Git repository URL**:
Edit `remote_build.sh` and update the `REPO_URL` variable with your repository.

3. **Build and push to ECR**:
```bash
./remote_build.sh
```

This will:
- Start the EC2 instance if stopped
- Clone/update your repository
- Build the Docker image (10x faster than Mac!)
- Push to ECR
- Show build time and status

4. **Run training** (from your Mac):
```bash
cd /Users/michaelmontalbano/work/repos/dental-seg/medmamba
python sagemaker_launcher_medmamba.py --class-group bone-loss
```

## 📊 Monitoring

- **CloudWatch Dashboard**: Monitor CPU, memory, and disk usage
- **SSH Access**: `./connect_to_builder.sh` to connect directly
- **Docker Status**: Run `~/docker_status.sh` on the instance

## 💰 Cost Management

- **Instance Type**: t3.medium (~$0.04/hour)
- **Auto-stop**: Stop when not in use with `./manage_builder.sh stop`
- **Typical build**: ~$0.01-0.02 per build

## 🛠️ Management Commands

```bash
# Check instance status
./manage_builder.sh status

# Start instance
./manage_builder.sh start

# Stop instance (saves money)
./manage_builder.sh stop

# Terminate instance (deletes everything)
./manage_builder.sh terminate

# Connect via SSH
./connect_to_builder.sh
```

## 🔧 Advanced Usage

### Building specific branches
```bash
# Edit remote_build.sh
BRANCH="feature/my-branch"
```

### Using different instance types
```bash
# Edit setup_ec2_builder.sh before first run
INSTANCE_TYPE="t3.large"  # More CPU
INSTANCE_TYPE="t3.xlarge" # Even more CPU
```

### Manual builds on EC2
```bash
# Connect to instance
./connect_to_builder.sh

# Navigate to builds
cd ~/builds/dental-seg/medmamba/container_medmamba

# Build manually
docker build -t medmamba:test .
```

## 🚨 Troubleshooting

### SSH connection refused
- Wait 2-3 minutes after starting instance
- Check security group allows your IP

### Build fails
- Connect via SSH and check logs
- Ensure ECR repository exists
- Check IAM permissions

### Instance won't start
- Check AWS service limits
- Verify region matches your config

## 📝 Architecture

```
Your Mac                    EC2 Instance               ECR
--------                    ------------               ---
remote_build.sh  ----SSH--> Clone repo      
                           Build Docker image  -----> Push image
                           Monitor with CloudWatch
                           
sagemaker_launcher.py --------------------------------> Use image
```

## 🔒 Security Notes

- SSH key stored in `~/.ssh/medmamba-builder-key.pem`
- Security group only allows SSH from your IP
- Instance has IAM role for ECR access
- Stop instance when not in use
