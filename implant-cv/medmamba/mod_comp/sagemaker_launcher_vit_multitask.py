from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()

# Cost optimization options
USE_SPOT_INSTANCES = False  # Set to False for guaranteed on-demand instances
INSTANCE_TYPE = 'ml.g6.8xlarge'  # GPU instance for faster training

# Enhanced training configuration with early stopping and adaptive loss
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'image_root': '/opt/ml/input/data/train_augmented',
    'model_size': 'base',  # tiny, small, base, large
    'epochs': 150,  # Higher since early stopping will prevent overfitting
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'output_dir': '/opt/ml/model',
    'validate_every': 1,  # Validate every epoch
    'detailed_metrics': 'true',  # Enable F1, precision, recall
    'save_best_only': 'true',    # Only save best model to save space
    'early_stopping_patience': 20,  # Stop if no improvement for 20 epochs
    'adaptive_loss_weighting': 'true',  # Auto-balance company vs model loss
}

print("🦷 Enhanced Company + Model Classification Training")
print("=" * 60)
print("🎯 Enhanced production features:")
print("   • Company identification") 
print("   • Model identification")
print("   • Early stopping (prevents overfitting)")
print("   • Adaptive loss weighting (auto-balances tasks)")
print("   • Learning rate warmup & cosine annealing")
print("   • Label smoothing & gradient clipping") 
print("   • Detailed validation metrics (F1, precision, recall)")
print("   • Enhanced data augmentation")
print("   • Best model auto-saving")
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
    training_hours = 4  # estimated with early stopping
    
    print(f"\n💵 Cost Estimate for ~{training_hours} hours (with early stopping):")
    print(f"   On-demand: ${on_demand_cost * training_hours:.2f}")
    if USE_SPOT_INSTANCES:
        print(f"   Spot price: ~${spot_cost * training_hours:.2f} (70% savings)")
        print(f"   💰 You save: ~${(on_demand_cost - spot_cost) * training_hours:.2f}")

print("\n🚀 Enhanced Training Configuration")
print("=" * 40)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 40)

# Create estimator with spot instance configuration
estimator_config = {
    'entry_point': 'train_vit_multitask.py',
    'source_dir': '.',
    'role': role,
    'framework_version': '2.1.0',
    'py_version': 'py310',
    'instance_type': INSTANCE_TYPE,
    'instance_count': 1,
    'hyperparameters': hyperparameters,
    'dependencies': ['requirements.txt'],
    'base_job_name': 'enhanced-company-model-spot' if USE_SPOT_INSTANCES else 'enhanced-company-model',
    'output_path': f's3://{session.default_bucket()}/enhanced-company-model-output',
    'volume_size': 50,
}

# Add spot instance configuration
if USE_SPOT_INSTANCES:
    estimator_config.update({
        'use_spot_instances': True,
        'max_run': 16*60*60,  # 16 hours max training time
        'max_wait': 24*60*60,  # 24 hours max wait (including interruptions)
        'checkpoint_s3_uri': f's3://{session.default_bucket()}/enhanced-company-model-checkpoints/',
    })
    print(f"🔄 Checkpointing enabled: {estimator_config['checkpoint_s3_uri']}")
else:
    estimator_config.update({
        'max_run': 12*60*60,  # Shorter due to early stopping
        'keep_alive_period_in_seconds': 30*60,
    })

estimator = PyTorch(**estimator_config)

# Input data configuration
input_data = {
    'train_augmented': 's3://codentist-general/datasets/aug_dataset_2/'
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

print(f"\n🎯 Enhanced Training Features:")
print(f"  🛑 Early Stopping: {hyperparameters['early_stopping_patience']} epoch patience")
print(f"  🔧 Adaptive Loss Weighting: Auto-balances company vs model tasks")
print(f"  🔬 Learning Rate Warmup: {hyperparameters['warmup_epochs']} epochs")
print(f"  📊 Detailed Metrics: Classification reports, F1 scores")
print(f"  🎯 Target Performance: >95% company, >90% model accuracy")
print(f"  💾 Smart Saving: Only best model saved")
print(f"  🔄 Validation: Every epoch with comprehensive metrics")

print(f"\n🚀 Training Timeline Estimate:")
print(f"  🕐 Enhanced ViT-base on ml.g6.8xlarge:")
print(f"     • Early stopping typically triggers: ~60-80 epochs")
print(f"     • Estimated training time: ~3-4 hours")
print(f"     • Warmup phase: First 10 epochs")
print(f"     • Automatic loss balancing throughout training")

print(f"\n🚀 Starting enhanced training job...")
if USE_SPOT_INSTANCES:
    print(f"💰 Using SPOT INSTANCES for maximum cost savings!")
    print(f"⚠️  Note: Training may be interrupted but will auto-resume")
else:
    print(f"💰 Using ON-DEMAND INSTANCES for guaranteed availability")

# Start the training job
estimator.fit(input_data)

print(f"\n✅ Enhanced training job submitted!")
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
print(f"  📊 vocabularies.json - Company and model mappings") 
print(f"  📈 training_history.json - Complete training metrics with loss weights")
print(f"  📋 training_info.json - Final summary with early stopping info")

print(f"\n🎯 Enhanced Monitoring Tips:")
if USE_SPOT_INSTANCES:
    print(f"  💰 Spot price savings: Check 'Billable seconds' vs 'Training time'")
    print(f"  🔄 Interruptions: Look for 'SpotInterruption' in logs (rare)")
    print(f"  📊 Progress preserved: Training resumes from last checkpoint")
else:
    print(f"  💰 Fixed pricing: Predictable costs, no interruptions")
    
print(f"  🛑 Early stopping: Watch for 'Early stopping triggered' message")
print(f"  🔧 Loss balancing: Monitor company vs model loss weight adjustments")
print(f"  📊 Watch for validation accuracy > 0.92 combined")
print(f"  🔍 Check CloudWatch logs for detailed metrics")
print(f"  💾 Best model auto-saved when validation improves")

print(f"\n💡 Cost Optimization Tips:")
if USE_SPOT_INSTANCES:
    print(f"  ✅ You're already using spot instances (70-90% savings)")
    print(f"  ✅ Early stopping reduces training time automatically")
    print(f"  💡 For even more savings: Try smaller model (--model_size small)")
else:
    print(f"  💰 Switch to spot instances: Set USE_SPOT_INSTANCES = True")
    print(f"  ✅ Early stopping already reduces costs by ~30-50%")
    print(f"  💡 Use smaller instance: ml.g5.2xlarge for development")

print(f"\n🎯 Success Criteria:")
print(f"  ✅ Company accuracy: >95%")
print(f"  ✅ Model accuracy: >90%") 
print(f"  ✅ Combined accuracy: >92%")
print(f"  ✅ Training converged without overfitting (early stopping)")
print(f"  ✅ F1 scores > 0.90 for both tasks")
print(f"  ✅ Loss weights automatically balanced between tasks")
if USE_SPOT_INSTANCES:
    print(f"  💰 Cost savings: 70-90% vs on-demand pricing")

print(f"\n🔧 Enhanced Features Benefits:")
print(f"  🛑 Early stopping prevents overfitting and saves time/cost")
print(f"  🔧 Adaptive loss weighting improves balanced performance")
print(f"  📊 Better convergence and more stable training")
print(f"  💾 Automatic best model selection")
print(f"  📈 Detailed training history with loss weight evolution")