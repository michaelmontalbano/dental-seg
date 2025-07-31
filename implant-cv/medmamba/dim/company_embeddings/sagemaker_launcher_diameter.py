from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()
bucket = 'codentist-general'

# Training configuration for DIAMETER ONLY regression with company embeddings
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'model_size': 'base',  # tiny, small, base, large
    'task': 'diameter',  # Predict ONLY diameter
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'output_dir': '/opt/ml/model',
    'image_root': '/opt/ml/input/data/train_augmented',
    'validate_every': 1,
    'save_best_only': 'true',
    'warmup_epochs': 5,
    'embed_dim': 256,  # Company embedding dimension
    'use_attention': 'true',  # Use attention mechanism
    'company_checkpoint': 'vit_company.pth',  # Using local model file in same directory
    'unfreeze_epoch': 20,  # Start unfreezing at epoch 20
    'full_unfreeze_epoch': 40,  # Fully unfreeze at epoch 40
    'company_lr_multiplier': 0.1  # Lower learning rate for company model
}

print("🎯 Diameter Regression with Company Embeddings - SageMaker Training")
print("=" * 60)
print("📊 Architecture highlights:")
print("   • Pretrained company model produces 256-dim embeddings")
print("   • Regression head predicts implant DIAMETER only")
print("   • Attention mechanism weights company embedding influence")
print("   • Progressive unfreezing: 50% at epoch 20, 100% at epoch 40")
print("=" * 60)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 60)

estimator = PyTorch(
    entry_point='train_regression_with_embeddings.py',
    source_dir='.',
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    hyperparameters=hyperparameters,
    dependencies=['../../requirements.txt'],  # Use the shared requirements
    base_job_name='regression-diameter-embeddings',
    output_path=f's3://{session.default_bucket()}/regression-diameter-embeddings-output',
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
print(f"\n💡 Expected outputs:")
print(f"   • Diameter predictions (MAE, RMSE, R²)")
print(f"   • Best model will minimize diameter prediction error")
