from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()
bucket = 'codentist-general'

# Training configuration
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'model_size': 'base',  # tiny, small, base, large
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'output_dir': '/opt/ml/model',
    'image_root': '/opt/ml/input/data/train_augmented',
    'validate_every': 10  # Run validation every 10 epochs
}

print("🚀 SageMaker Training Configuration")
print("=" * 40)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 40)

estimator = PyTorch(
    entry_point='train_vit_multitask.py',
    source_dir='.',  # Local directory containing your scripts
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',  # GPU instance for faster training
    instance_count=1,
    hyperparameters=hyperparameters,
    dependencies=['requirements.txt'],
    base_job_name='vit-multitask-augmented',
    output_path=f's3://{session.default_bucket()}/vit-multitask-output',
    max_run=24*60*60,  # 24 hours max runtime
    keep_alive_period_in_seconds=30*60,  # Keep instance alive for 30 min after job
)

# Input data configuration
# Make sure to upload your aug_dataset folder to S3 first:
# aws s3 sync aug_dataset/ s3://your-bucket/datasets/implants_augmented/

input_data = {
    'train_augmented': 's3://codentist-general/datasets/aug_dataset/'
}

print(f"\n📊 Input Data Mapping:")
for channel, s3_path in input_data.items():
    print(f"  {channel} -> {s3_path}")

print(f"\n🎯 Expected S3 structure:")
print(f"  s3://your-bucket/datasets/implants_augmented/")
print(f"  ├── train/")
print(f"  │   ├── Company_A/")
print(f"  │   │   ├── image_aug_001.jpg")
print(f"  │   │   └── labels/")
print(f"  │   └── Company_B/")
print(f"  ├── val/")
print(f"  │   └── Company_C/")
print(f"  ├── augmented_train.json")
print(f"  ├── augmented_val.json")
print(f"  └── augmentation_summary.json")

print(f"\n🚀 Starting training job...")

# Start the training job
estimator.fit(input_data)

print(f"\n✅ Training job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"📁 Model artifacts will be saved to: {estimator.output_path}")