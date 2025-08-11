from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()
bucket = 'codentist-general'

# Training configuration for unified classifier (all 4 tasks)
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'model_size': 'base',  # tiny, small, base, large
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'output_dir': '/opt/ml/model',
    'image_root': '/opt/ml/input/data/train_augmented',
    'validate_every': 1,
    'save_best_only': 'true',
    'warmup_epochs': 5,
    'diameter_bin_size': 0.5,
    'length_bin_strategy': 'adaptive',  # adaptive or fixed
    'task_weights': '1,1,1,1'  # weights for company,model,diameter,length
}

print("🎯 Unified Classifier - SageMaker Training Configuration")
print("=" * 50)
print("📊 Training all four tasks simultaneously:")
print("   • Company identification")
print("   • Model identification")
print("   • Diameter classification")
print("   • Length classification")
print("=" * 50)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 50)

estimator = PyTorch(
    entry_point='train_unified_classifier.py',
    source_dir='.',
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    hyperparameters=hyperparameters,
    dependencies=['requirements.txt'],
    base_job_name='unified-classifier',
    output_path=f's3://{session.default_bucket()}/unified-classifier-output',
    max_run=24*60*60,  # 24 hours max runtime
    keep_alive_period_in_seconds=30*60,  # Keep instance alive for 30 min
)

# Input data configuration
input_data = {
    'train_augmented': 's3://codentist-general/datasets/aug_dataset/'
}

print(f"\n📊 Input Data Mapping:")
for channel, s3_path in input_data.items():
    print(f"  {channel} -> {s3_path}")

print(f"\n🚀 Starting training job...")

# Start the training job
estimator.fit(input_data)

print(f"\n✅ Training job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"📁 Model artifacts will be saved to: {estimator.output_path}")
