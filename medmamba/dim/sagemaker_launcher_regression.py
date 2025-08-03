from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()

# Cost optimization options
USE_SPOT_INSTANCES = False  # Set to False for guaranteed on-demand instances
INSTANCE_TYPE = 'ml.g6.8xlarge'  # GPU instance for faster training

# Advanced training configuration for Length + Diameter regression
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'image_root': '/opt/ml/input/data/train_augmented',
    'model_size': 'base',  # tiny, small, base, large
    'epochs': 100,  # Reduced from 400 which was too many
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'output_dir': '/opt/ml/model',
    'validate_every': 1,  # Validate every epoch
    'detailed_metrics': 'true',  # Enable R², MAPE, etc.
    'save_best_only': 'true'     # Only save best model to save space
}

print("🦷 Advanced Length + Diameter Regression Training")
print("=" * 60)
print("🎯 Production-ready regression training with advanced features:")
print("   • Length prediction (mm)") 
print("   • Diameter prediction (mm)")
print("   • Learning rate warmup & cosine annealing")
print("   • Gradient clipping & dropout regularization") 
print("   • Detailed regression metrics (MAE, RMSE, R², MAPE)")
print("   • Enhanced data augmentation")
print("   • Best model auto-saving based on MAE")
print("=" * 60)

print(f"\n💰 Cost Configuration:")
if USE_SPOT_INSTANCES:
    print(f"   🎯 Spot Instances: ENABLED (70-90% cost savings)")
    print(f"   ⚠️  Risk: Training may be interrupted (rare)")
    print(f"   🔄 Auto-restart: Enabled with checkpointing")
    print(f"   💡 Best for: Development, experimentation")
else:
    print(f"   💰 On-Demand Instances: Guaranteed but expensive")
    print(f"   ✅ No interruption risk")
    print(f"   💡 Best for: Production, critical deadlines")

print(f"   🖥️  Instance: {INSTANCE_TYPE}")

# Estimate costs
if INSTANCE_TYPE == 'ml.g6.8xlarge':
    on_demand_cost = 4.352  # $/hour
    spot_cost = on_demand_cost * 0.3  # ~70% savings typical
    training_hours = 5  # estimated for 100 epochs
    
    print(f"\n💵 Cost Estimate for ~{training_hours} hours:")
    print(f"   On-demand: ${on_demand_cost * training_hours:.2f}")
    if USE_SPOT_INSTANCES:
        print(f"   Spot price: ~${spot_cost * training_hours:.2f} (70% savings)")
        print(f"   💰 You save: ~${(on_demand_cost - spot_cost) * training_hours:.2f}")

print("\n🚀 SageMaker Training Configuration")
print("=" * 40)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 40)

# Create estimator with spot instance configuration
estimator_config = {
    'entry_point': 'train_length_diameter_model.py',
    'source_dir': '.',
    'role': role,
    'framework_version': '2.1.0',
    'py_version': 'py310',
    'instance_type': INSTANCE_TYPE,
    'instance_count': 1,
    'hyperparameters': hyperparameters,
    'dependencies': ['requirements.txt'],
    'base_job_name': 'advanced-regression-spot' if USE_SPOT_INSTANCES else 'advanced-regression',
    'output_path': f's3://{session.default_bucket()}/advanced-regression-output',
    'volume_size': 50,
}

# Add spot instance configuration
if USE_SPOT_INSTANCES:
    estimator_config.update({
        'use_spot_instances': True,
        'max_run': 16*60*60,  # 16 hours max training time
        'max_wait': 24*60*60,  # 24 hours max wait (including interruptions)
        'checkpoint_s3_uri': f's3://{session.default_bucket()}/advanced-regression-checkpoints/',
    })
    print(f"🔄 Checkpointing enabled: {estimator_config['checkpoint_s3_uri']}")
else:
    estimator_config.update({
        'max_run': 16*60*60,
        'keep_alive_period_in_seconds': 30*60,
    })

estimator = PyTorch(**estimator_config)

# Input data configuration
input_data = {
    'train_augmented': 's3://codentist-general/datasets/aug_dataset/'
}

print(f"\n📊 Input Data Mapping:")
for channel, s3_path in input_data.items():
    print(f"  {channel} -> {s3_path}")

print(f"\n🎯 Expected S3 structure:")
print(f"  s3://codentist-general/datasets/aug_dataset/")
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

print(f"\n🎯 Advanced Regression Features:")
print(f"  🔬 Learning Rate Warmup: {hyperparameters['warmup_epochs']} epochs")
print(f"  📊 Detailed Metrics: MAE, RMSE, R², MAPE")
print(f"  🎯 Target Performance: <0.5mm MAE for both length and diameter")
print(f"  💾 Smart Saving: Only best model saved (based on combined MAE)")
print(f"  🔄 Validation: Every epoch with comprehensive regression metrics")

print(f"\n🚀 Training Timeline Estimate:")
print(f"  🕐 ViT-base on ml.g6.8xlarge:")
print(f"     • 100 epochs: ~4-5 hours")
print(f"     • Warmup phase: First 10 epochs")
print(f"     • Best results: Usually epoch 60-80")

print(f"\n🚀 Starting advanced regression training job...")
if USE_SPOT_INSTANCES:
    print(f"💰 Using SPOT INSTANCES for maximum cost savings!")
    print(f"⚠️  Note: Training may be interrupted but will auto-resume")
else:
    print(f"💰 Using ON-DEMAND INSTANCES for guaranteed availability")

# Start the training job
estimator.fit(input_data)

print(f"\n✅ Advanced regression training job submitted!")
if USE_SPOT_INSTANCES:
    print(f"💰 Spot instance job running - check SageMaker console for status")
    print(f"🔄 If interrupted, job will automatically restart from checkpoint")
else:
    print(f"🔒 On-demand job guaranteed to run without interruption")

print(f"📊 Monitor progress in SageMaker console")
print(f"📁 Model artifacts will be saved to: {estimator.output_path}")
if USE_SPOT_INSTANCES:
    print(f"🔄 Checkpoints saved to: {estimator_config['checkpoint_s3_uri']}")

print(f"\n📋 Expected outputs:")
print(f"  📦 best_model.pth - Best performing model with full state")
print(f"  📈 training_history.json - Complete training metrics")
print(f"  📋 training_info.json - Final summary with regression metrics")

print(f"\n🎯 Monitoring Tips:")
if USE_SPOT_INSTANCES:
    print(f"  💰 Spot price savings: Check 'Billable seconds' vs 'Training time'")
    print(f"  🔄 Interruptions: Look for 'SpotInterruption' in logs (rare)")
    print(f"  📊 Progress preserved: Training resumes from last checkpoint")
else:
    print(f"  💰 Fixed pricing: Predictable costs, no interruptions")
    
print(f"  📊 Watch for combined MAE < 0.5mm")
print(f"  🔍 Check CloudWatch logs for detailed regression metrics")
print(f"  ⚠️  Training should converge by epoch 80-100")
print(f"  💾 Best model auto-saved when validation MAE improves")

print(f"\n💡 Cost Optimization Tips:")
if USE_SPOT_INSTANCES:
    print(f"  ✅ You're already using spot instances (70-90% savings)")
    print(f"  💡 For even more savings: Try smaller model (--model_size small)")
    print(f"  💡 Reduce epochs if convergence is fast (--epochs 50)")
else:
    print(f"  💰 Switch to spot instances: Set USE_SPOT_INSTANCES = True")
    print(f"  💡 Use smaller instance: ml.g5.2xlarge for development")
    print(f"  💡 Reduce training time: Use --epochs 50 for testing")

print(f"\n🎯 Success Criteria:")
print(f"  ✅ Length MAE: <0.5mm")
print(f"  ✅ Diameter MAE: <0.5mm") 
print(f"  ✅ Combined MAE: <0.5mm")
print(f"  ✅ R² score: >0.85 for both tasks")
print(f"  ✅ Training converged without overfitting")
if USE_SPOT_INSTANCES:
    print(f"  💰 Cost savings: 70-90% vs on-demand pricing")