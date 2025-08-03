from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()
bucket = 'codentist-general'

# Training configuration for hierarchical model with company embeddings
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
    'embed_dim': 256,  # Company embedding dimension
    'use_attention': 'true',  # Use attention mechanism
    'pretrain_company_epochs': 20,  # Pretrain company model first
    'unfreeze_epoch': 40,  # When to start unfreezing company model
    'company_checkpoint': 'vit_company.pth'  # Using local model file in same directory
}

print("🎯 Hierarchical Model with Company Embeddings - SageMaker Training")
print("=" * 60)
print("📊 Architecture highlights:")
print("   • Company classifier produces 256-dim embeddings")
print("   • Model classifier uses both image features and company embeddings")
print("   • Attention mechanism weights the influence of each")
print("   • Gradual unfreezing for fine-tuning")
print("=" * 60)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 60)

estimator = PyTorch(
    entry_point='train_hierarchical.py',
    source_dir='.',
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    hyperparameters=hyperparameters,
    dependencies=['../../requirements.txt'],  # Use the shared requirements
    base_job_name='hierarchical-embeddings',
    output_path=f's3://{session.default_bucket()}/hierarchical-embeddings-output',
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
print(f"\n💡 Training strategy:")
print(f"   • Epochs 1-20: Focus on company classification")
print(f"   • Epochs 21-40: Joint training with frozen company model")
print(f"   • Epochs 41-100: Fine-tuning with gradual unfreezing")
