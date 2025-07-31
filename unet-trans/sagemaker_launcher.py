#!/usr/bin/env python3
"""
SageMaker Launcher for Pure TransUNet Dental Segmentation
Adapted for unet-trans implementation with segmentation-based training
Now supports model size selection (small/normal)
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch
import logging
import random
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print log level
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()

def get_default_bucket():
    """Get the default SageMaker bucket"""
    session = sagemaker.Session()
    return session.default_bucket()

def estimate_training_time(num_images: int, batch_size: int, epochs: int, model_size: str = 'normal') -> str:
    """Estimate training time based on dataset size, parameters, and model size"""
    # Adjust time estimate based on model size
    if model_size == 'small':
        time_per_epoch_minutes = (num_images / batch_size) * 0.5  # Faster with ResNet18
    else:
        time_per_epoch_minutes = (num_images / batch_size) * 0.8  # Original estimate for ResNet50
    
    total_time_hours = (time_per_epoch_minutes * epochs) / 60
    
    if total_time_hours < 1:
        return f"~{total_time_hours*60:.0f} minutes"
    elif total_time_hours < 24:
        return f"~{total_time_hours:.1f} hours"
    else:
        return f"~{total_time_hours/24:.1f} days"

def get_recommended_instance_type(dataset_size: int, model_size: str = 'normal') -> str:
    """Get recommended instance type based on dataset size and model size"""
    if model_size == 'small':
        # Smaller model can use smaller instances
        if dataset_size > 5000:
            return 'ml.g6e.4xlarge'  # Smaller GPU for large datasets
        elif dataset_size > 1000:
            return 'ml.g6e.xlarge'   # Smaller GPU for medium datasets
        else:
            return 'ml.g6e.xlarge'   # Smaller GPU for small datasets
    else:
        # Normal model (original recommendations)
        if dataset_size > 5000:
            return 'ml.g6e.12xlarge'  # Newer GPU for larger datasets
        elif dataset_size > 1000:
            return 'ml.g6e.xlarge'   # Newer GPU for medium datasets
        else:
            return 'ml.g6e.xlarge'   # Newer GPU for smaller datasets

def get_model_config(model_size: str):
    """Get model configuration based on size"""
    if model_size == 'small':
        return {
            'backbone': 'resnet18',
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6
        }
    else:  # normal
        return {
            'backbone': 'resnet50',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        }

def main():
    parser = argparse.ArgumentParser(description='Launch Pure TransUNet Dental Segmentation on SageMaker')
    
    # Model Configuration
    parser.add_argument('--model-size', type=str, default='normal',
                       choices=['small', 'normal'],
                       help='Model size: small (ResNet18, ~30M params) or normal (ResNet50, ~110M params)')
    
    # AWS Configuration
    parser.add_argument('--role', type=str, 
                       default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                       help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2', 
                       help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                       help='AWS profile to use')
    
    # Instance Configuration  
    parser.add_argument('--instance-type', type=str, default=None,
                       help='SageMaker instance type (auto-selected if not specified)')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=30,
                       help='EBS volume size in GB')
    parser.add_argument('--max-run', type=int, default=259200,  # 24 hours
                       help='Maximum runtime in seconds')
    parser.add_argument('--use-spot', action='store_true',
                       help='Use spot instances for cost savings')
    
    # Data Configuration
    parser.add_argument('--train-data', type=str, 
                       default='s3://codentist-general/datasets/master',
                       help='S3 path to training data')
    parser.add_argument('--validation-data', type=str, default=None,
                       help='S3 path to validation data (optional)')
    parser.add_argument('--train_annotations', type=str, default='train.json',
                       help='Name of the training annotations JSON file')
    parser.add_argument('--val_annotations', type=str, default='val.json',
                       help='Name of the validation annotations JSON file')
    parser.add_argument('--dataset-size', type=int, default=2000,
                       help='Estimated dataset size for instance recommendation')
    
    # Dataset Filtering Parameters
    parser.add_argument('--class-group', type=str, default='bone-loss',
                       choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth'],
                       help='Subset of classes to train on')
    parser.add_argument('--xray-type', type=str, default=None,
                   help='Filter dataset by X-ray type - use comma-separated values (e.g., "bitewing,periapical")')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (reduced default for memory efficiency)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size (reduced default: 512 for memory efficiency, 1024 for full quality)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit dataset size for testing')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                       help='Confidence threshold for detection metrics (default: 0.4)')
    
    # Memory Optimization Parameters
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient settings (batch_size=1, image_size=512)')
    parser.add_argument('--high-memory', action='store_true',
                       help='Use high-memory settings (batch_size=8, image_size=1024)')
    
    # Job Configuration
    parser.add_argument('--job-name', type=str, default=None,
                       help='Training job name')
    parser.add_argument('--output-path', type=str, default=None,
                       help='S3 output path')
    parser.add_argument('--code-location', type=str, default=None,
                       help='S3 code location')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for training to complete')
    
    # Framework
    parser.add_argument('--pytorch-version', type=str, default='2.0.1',
                       help='PyTorch version')
    parser.add_argument('--py-version', type=str, default='py310',
                       help='Python version')
    
    # Testing and Development
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with smaller settings')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without launching')
    
    args = parser.parse_args()
    
    # Get model configuration based on size
    model_config = get_model_config(args.model_size)
    
    # 💾 Apply Memory Optimization Settings
    if args.memory_efficient:
        logger.info("💾 MEMORY EFFICIENT MODE: Reducing batch size and image size")
        args.batch_size = 1
        args.image_size = 512
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Image size: {args.image_size}")
    elif args.high_memory:
        logger.info("🚀 HIGH MEMORY MODE: Increasing batch size and image size")
        args.batch_size = 8
        args.image_size = 1024
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Image size: {args.image_size}")
    
    # Adjust batch size based on model size if not explicitly set
    if args.model_size == 'small' and not (args.memory_efficient or args.high_memory):
        # Small model can handle larger batch sizes
        args.batch_size = min(args.batch_size * 2, 8)
        logger.info(f"📊 Adjusted batch size for small model: {args.batch_size}")
    
    # Setup session
    if args.profile:
        session = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=session)
    else:
        sagemaker_session = sagemaker.Session()
    
    # Default values
    default_bucket = get_default_bucket()
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Auto-configure based on mode
    if args.test_mode:
        logger.info("🧪 TEST MODE: Using reduced settings for quick testing")
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 4)
        args.dataset_size = min(args.dataset_size, 100)
        job_suffix = 'test'
        args.max_run = 3600  # 1 hour max for testing
    else:
        job_suffix = f'transunet-{args.model_size}'
    
    # Auto-select instance type if not specified
    if not args.instance_type:
        args.instance_type = get_recommended_instance_type(args.dataset_size, args.model_size)
        logger.info(f"🖥️ Auto-selected instance type: {args.instance_type}")
    
    # Validation data defaults to train data if not specified
    validation_data = args.validation_data or args.train_data
    
    # Generate job name with class group and model size info
    job_suffix = f"{args.model_size}-{args.class_group}"
    if args.xray_type:
        job_suffix += f"-{args.xray_type.replace(',', '-')}"
    if args.max_images:
        job_suffix += f"-subset{args.max_images}"
    
    if args.test_mode:
        job_suffix += "-test"
    else:
        job_suffix += "-v1"
    
    # Add random 3-digit number to ensure unique job names
    random_suffix = random.randint(100, 999)
    job_name = args.job_name or f'pure-transunet-{job_suffix}-{timestamp}-{random_suffix}'
    
    output_path = args.output_path or f's3://{default_bucket}/pure-transunet-output'
    code_location = args.code_location or f's3://{default_bucket}/pure-transunet-code'
    
    # Build hyperparameters matching train.py expectations
    hyperparameters = {
        # Core training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'confidence_threshold': args.confidence_threshold,
        
        # Model architecture parameters from model_config
        'backbone': model_config['backbone'],
        'embed_dim': model_config['embed_dim'],
        'depth': model_config['depth'],
        'num_heads': model_config['num_heads'],
        
        # Dataset configuration
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'class_group': args.class_group,
    }

    # Add xray_type if specified (though disabled by default)
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type

    # Add optional parameters
    if args.max_images:
        hyperparameters['max_images'] = args.max_images

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}
    
    # 🔍 DEBUG: Log all hyperparameters being passed to SageMaker
    logger.info(f"🔍 DEBUG SageMaker Hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"  {key}: {value} (type: {type(value).__name__})")
    
    # Estimate training time
    estimated_time = estimate_training_time(args.dataset_size, args.batch_size, args.epochs, args.model_size)
    logger.info(f"⏱️ Estimated training time: {estimated_time}")
    
    # Calculate model size estimates
    if args.model_size == 'small':
        model_params = "~30M"
        model_size_mb = "~120MB"
    else:
        model_params = "~110M"
        model_size_mb = "~440MB"
    
    # Configuration summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🦷 PURE TRANSUNET SEGMENTATION TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"📝 Job name: {job_name}")
    logger.info(f"🖥️ Instance: {args.instance_type} ({'spot' if args.use_spot else 'on-demand'})")
    logger.info(f"🧠 Model: TransUNet-{args.model_size.upper()} ({model_config['backbone']}, {model_params} params, {model_size_mb})")
    logger.info(f"   Architecture: embed_dim={model_config['embed_dim']}, depth={model_config['depth']}, heads={model_config['num_heads']}")
    logger.info(f"📊 Data: {args.train_data}")
    logger.info(f"📄 Train annotations: {args.train_annotations}")
    logger.info(f"📄 Val annotations: {args.val_annotations}")
    if args.xray_type:
        logger.info(f"🩻 X-ray type filter: {args.xray_type} (DISABLED)")
    logger.info(f"🎯 Class group: {args.class_group}")
    logger.info(f"🎛️ Batch size: {args.batch_size}, LR: {args.learning_rate}, Image size: {args.image_size}")
    logger.info(f"⏱️ Estimated time: {estimated_time}")
    logger.info(f"💡 All hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"   {key}: {value}")
    logger.info(f"{'='*60}")
    
    if args.dry_run:
        logger.info("🏃 DRY RUN: Configuration shown above, not launching actual training")
        return None
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='src',
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        max_run=args.max_run,
        framework_version=args.pytorch_version,
        py_version=args.py_version,
        hyperparameters=hyperparameters,
        output_path=output_path,
        code_location=code_location,
        sagemaker_session=sagemaker_session,
        use_spot_instances=args.use_spot,
        
        # Environment configuration
        environment={
            # 'CUDA_VISIBLE_DEVICES': '0',  # ✅ REMOVED: Allow all GPUs to be used
            'PYTHONPATH': '/opt/ml/code/src',
            'LOG_LEVEL': 'WARNING',  # Reduce logging for faster training
        },
        
        # Checkpointing for spot instances
        checkpoint_s3_uri=f's3://{default_bucket}/pure-transunet-checkpoints' if args.use_spot else None,
        
        # Debugging
        enable_cloudwatch_metrics=True,
        debugger_hook_config=False,  # Disable for performance
    )
    
    # Training inputs
    inputs = {
        'train': args.train_data,
        'validation': validation_data
    }
    
    # Launch training
    logger.info(f"\n🚀 Launching Pure TransUNet training...")
    logger.info(f"📍 Monitor at: https://console.aws.amazon.com/sagemaker/")
    
    try:
        estimator.fit(
            inputs=inputs,
            job_name=job_name,
            wait=args.wait
        )
        
        if args.wait:
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"📦 Model artifacts: {estimator.model_data}")
        else:
            logger.info(f"🚀 Training job '{job_name}' started successfully")
            logger.info(f"📊 Monitor progress: aws sagemaker describe-training-job --training-job-name {job_name}")
            
    except Exception as e:
        logger.error(f"❌ Training launch failed: {e}")
        raise
    
    return estimator

if __name__ == '__main__':
    estimator = main()