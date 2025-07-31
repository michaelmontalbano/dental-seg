from sagemaker.pytorch import PyTorch
import sagemaker

role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()
bucket = 'codentist-general'

# Enhanced multi-task training configuration
hyperparameters = {
    'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
    'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json', 
    'model_size': 'base',  # tiny, small, base, large
    'epochs': 200,  # Higher since early stopping will prevent overfitting
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'output_dir': '/opt/ml/model',
    'image_root': '/opt/ml/input/data/train_augmented',
    'validate_every': 5,  # Run validation every 5 epochs
    'early_stopping_patience': 20,  # Stop if no improvement for 20 epochs
    'adaptive_loss_weighting': 'true',  # Auto-balance all 4 tasks
    'detailed_metrics': 'true',  # Show per-task metrics
}

print("🦷 Enhanced Multi-Task Classification Training")
print("=" * 60)
print("🎯 Enhanced production features:")
print("   • Company identification") 
print("   • Model identification")
print("   • Diameter classification")
print("   • Length classification")
print("   • Early stopping (prevents overfitting)")
print("   • Adaptive loss weighting (auto-balances 4 tasks)")
print("   • Enhanced data augmentation")
print("   • Learning rate warmup & cosine annealing")
print("   • Label smoothing & gradient clipping") 
print("   • Best model auto-saving")
print("=" * 60)

print("🚀 Enhanced SageMaker Training Configuration")
print("=" * 40)
for key, value in hyperparameters.items():
    print(f"{key}: {value}")
print("=" * 40)

# Estimate costs with early stopping
on_demand_cost = 4.352  # $/hour for ml.g6.8xlarge
training_hours = 4  # estimated with early stopping for 4-task model

print(f"\n💵 Cost Estimate for ~{training_hours} hours (with early stopping):")
print(f"   Instance: ml.g6.8xlarge - ${on_demand_cost * training_hours:.2f}")
print(f"   💰 Early stopping typically saves 30-50% vs full training")

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
    base_job_name='enhanced-vit-multitask',
    output_path=f's3://{session.default_bucket()}/enhanced-vit-multitask-output',
    max_run=16*60*60,  # 16 hours max runtime (reduced due to early stopping)
    keep_alive_period_in_seconds=30*60,  # Keep instance alive for 30 min after job
)

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

print(f"\n🎯 Enhanced Training Features:")
print(f"  🛑 Early Stopping: {hyperparameters['early_stopping_patience']} epoch patience")
print(f"  🔧 Adaptive Loss Weighting: Auto-balances all 4 classification tasks")
print(f"  🔬 Learning Rate Warmup: {hyperparameters['warmup_epochs']} epochs")
print(f"  📊 Detailed Metrics: Per-task accuracy tracking")
print(f"  🎯 Multi-Task Performance: Company, Model, Diameter, Length")
print(f"  💾 Smart Saving: Only best model saved")
print(f"  🔄 Validation: Every {hyperparameters['validate_every']} epochs")

print(f"\n🚀 Training Timeline Estimate:")
print(f"  🕐 Enhanced 4-task ViT-base on ml.g6.8xlarge:")
print(f"     • Early stopping typically triggers: ~70-100 epochs")
print(f"     • Estimated training time: ~4-5 hours")
print(f"     • Warmup phase: First 10 epochs")
print(f"     • Automatic loss balancing throughout training")
print(f"     • All 4 tasks balanced for optimal performance")

print(f"\n🚀 Starting enhanced multi-task training job...")

# Start the training job
estimator.fit(input_data)

print(f"\n✅ Enhanced multi-task training job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"📁 Model artifacts will be saved to: {estimator.output_path}")

print(f"\n📋 Expected outputs:")
print(f"  📦 best_model.pth - Best performing model with full state")
print(f"  📊 vocabularies.json - All 4 task vocabularies") 
print(f"  📈 training_history.json - Complete training metrics with loss weights")
print(f"  📋 training_info.json - Final summary with early stopping info")

print(f"\n🎯 Enhanced Monitoring Tips:")
print(f"  🛑 Early stopping: Watch for 'Early stopping triggered' message")
print(f"  🔧 Loss balancing: Monitor 4-task loss weight adjustments")
print(f"  📊 Multi-task performance: All tasks should improve together")
print(f"  🔍 Check CloudWatch logs for detailed per-task metrics")
print(f"  💾 Best model auto-saved when validation improves")
print(f"  📈 Training typically converges faster with 4-task learning")

print(f"\n🎯 Success Criteria:")
print(f"  ✅ Company accuracy: >95%")
print(f"  ✅ Model accuracy: >90%") 
print(f"  ✅ Diameter accuracy: >85%")
print(f"  ✅ Length accuracy: >85%")
print(f"  ✅ Training converged without overfitting (early stopping)")
print(f"  ✅ Loss weights automatically balanced between all 4 tasks")
print(f"  💰 Cost savings: 30-50% vs full training duration")

print(f"\n🔧 Enhanced Features Benefits:")
print(f"  🛑 Early stopping prevents overfitting and saves time/cost")
print(f"  🔧 4-task adaptive weighting improves balanced performance")
print(f"  📊 Better convergence with multi-task learning")
print(f"  💾 Automatic best model selection")
print(f"  📈 Detailed training history with loss weight evolution")
print(f"  🎯 Simultaneous optimization of all classification tasks")