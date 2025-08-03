import os
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

# Get SageMaker session and role
session = sagemaker.Session()
role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'

# S3 paths
bucket = session.default_bucket()
base_job_name = f"hierarchical-model-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
output_path = f"s3://{bucket}/sagemaker/hierarchical-model"

# Define the estimator
estimator = PyTorch(
    entry_point='train_hierarchical_model.py',
    source_dir='.',
    role=role,
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    base_job_name=base_job_name,
    output_path=output_path,
    sagemaker_session=session,
    environment={
        'WANDB_DISABLED': 'true',
        'TOKENIZERS_PARALLELISM': 'false'
    },
    hyperparameters={
        'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
        'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json',
        'image_root': '/opt/ml/input/data/train_augmented',
        'output_dir': '/opt/ml/model',
        'model_size': 'base',
        'epochs': 75,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'hierarchy_weight': 0.5,
        'confidence_threshold': 0.8,
        'validate_every': 1,
        'save_best_only': 'true'
    },
    metric_definitions=[
        {
            'Name': 'train_loss',
            'Regex': r'\[(\d+)\] train_loss = ([0-9\.]+);'
        },
        {
            'Name': 'train_company_accuracy',
            'Regex': r'\[(\d+)\] train_company_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'train_model_accuracy',
            'Regex': r'\[(\d+)\] train_model_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'val_loss',
            'Regex': r'\[(\d+)\] val_loss = ([0-9\.]+);'
        },
        {
            'Name': 'val_company_accuracy',
            'Regex': r'\[(\d+)\] val_company_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'val_model_accuracy',
            'Regex': r'\[(\d+)\] val_model_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'best_combined_accuracy',
            'Regex': r'best_combined_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'best_company_accuracy',
            'Regex': r'best_company_accuracy = ([0-9\.]+);'
        },
        {
            'Name': 'best_model_accuracy',
            'Regex': r'best_model_accuracy = ([0-9\.]+);'
        }
    ]
)

# Data configuration
data_channels = {
    'train_augmented': "s3://codentist-general/datasets/aug_dataset"
}

print(f"\n🚀 Starting hierarchical model training job...")
print(f"📊 Model: Hierarchical Company → Model Classifier")
print(f"🔧 Architecture: Attention-based hierarchical structure")
print(f"💾 Results will be saved to: {output_path}")

print(f"\n📋 Training Configuration:")
print(f"  - Model size: base")
print(f"  - Epochs: 75")
print(f"  - Batch size: 32")
print(f"  - Learning rate: 1e-4")
print(f"  - Hierarchy weight: 0.5")
print(f"  - Confidence threshold: 0.8")

# Start the training job
estimator.fit(data_channels)

print(f"\n✅ Training job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"⏰ Estimated time: 2-3 hours")
