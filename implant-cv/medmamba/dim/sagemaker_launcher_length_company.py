import sagemaker
from sagemaker.pytorch import PyTorch
import os
from datetime import datetime

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Configuration
bucket = sagemaker_session.default_bucket()
prefix = "implants/length-company-classification"

# Training script configuration
train_instance_type = "ml.g4dn.xlarge"  # GPU instance
train_instance_count = 1
train_volume_size = 50  # GB

# Hyperparameters
hyperparameters = {
    "train_json": "/opt/ml/input/data/train_augmented/augmented_train.json",
    "val_json": "/opt/ml/input/data/train_augmented/augmented_val.json",
    "company_checkpoint": "/opt/ml/input/data/company_model/vit_company.pth",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "model_size": "base",
    "warmup_epochs": 5,
    "validate_every": 1,
    "save_best_only": "true",
    "output_dir": "/opt/ml/model",
    "image_root": "/opt/ml/input/data/train_augmented",
    "bin_strategy": "adaptive",
    "n_bins": 10,
    "use_attention": "true"
}

# Create timestamp for unique job name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"length-company-classifier-{timestamp}"

# Create PyTorch estimator
estimator = PyTorch(
    entry_point="train_length_with_company.py",
    source_dir=".",
    role=role,
    instance_type=train_instance_type,
    instance_count=train_instance_count,
    volume_size=train_volume_size,
    framework_version="2.0",
    py_version="py310",
    hyperparameters=hyperparameters,
    sagemaker_session=sagemaker_session,
    disable_profiler=True,
    base_job_name="length-company-classifier"
)

# Define data channels
# S3 path for augmented training data
s3_train_path = f"s3://{bucket}/implants/augmented-data/aug_dataset"

# S3 path for company model - you'll need to upload the pretrained model here
s3_company_model_path = f"s3://{bucket}/implants/models/company_vit"

data_channels = {
    "train_augmented": s3_train_path,
    "company_model": s3_company_model_path
}

print(f"Starting training job: {job_name}")
print(f"Training data: {s3_train_path}")
print(f"Company model: {s3_company_model_path}")
print(f"Instance type: {train_instance_type}")
print(f"Hyperparameters: {hyperparameters}")

# IMPORTANT: Before running this script, ensure you've uploaded:
# 1. The augmented dataset to s3://{bucket}/implants/augmented-data/aug_dataset
# 2. The pretrained company model (vit_company.pth) to s3://{bucket}/implants/models/company_vit/

# Start training
estimator.fit(data_channels, job_name=job_name, wait=True)

print(f"Training completed! Model artifacts saved to: {estimator.model_data}")
