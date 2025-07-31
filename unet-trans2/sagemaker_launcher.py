#!/usr/bin/env python3
"""
SageMaker Launcher for TransUNet Dental Segmentation
Updated to use train.json/val.json and support X-ray type filtering
FIXED: Updated for new train/val file structure and bone-loss bbox handling
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_bucket():
    """Get the default SageMaker bucket"""
    session = sagemaker.Session()
    return session.default_bucket()

def estimate_training_time(num_images: int, batch_size: int, epochs: int) -> str:
    """Estimate training time based on dataset size and parameters"""
    time_per_epoch_minutes = (num_images / batch_size) * 0.5  # 0.5 min per batch estimate for TransUNet
    total_time_hours = (time_per_epoch_minutes * epochs) / 60
    
    if total_time_hours < 1:
        return f"~{total_time_hours*60:.0f} minutes"
    elif total_time_hours < 24:
        return f"~{total_time_hours:.1f} hours"
    else:
        return f"~{total_time_hours/24:.1f} days"

def get_recommended_instance_type(dataset_size: int) -> str:
    """Get recommended instance type based on dataset size"""
    if dataset_size > 5000:
        return 'ml.g6e.12xlarge'  # Newer GPU for larger datasets
    elif dataset_size > 1000:
        return 'ml.g6e.xlarge'   # Newer GPU for medium datasets
    else:
        return 'ml.g6e.xlarge'   # Newer GPU for smaller datasets

def main():
    parser = argparse.ArgumentParser(description='Launch TransUNet Dental Segmentation on SageMaker')
    
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
    parser.add_argument('--max-run', type=int, default=86400,  # 24 hours
                       help='Maximum runtime in seconds')
    parser.add_argument('--use-spot', action='store_true',
                       help='Use spot instances for cost savings')
    
    # Data Configuration
    parser.add_argument('--train-data', type=str, 
                       default='s3://codentist-general/datasets/master',
                       help='S3 path to training data')
    parser.add_argument('--validation-data', type=str, default=None,
                       help='S3 path to validation data (optional)')
    parser.add_argument('--train-annotations', type=str, default='train.json',
                       help='Name of the training annotations JSON file')
    parser.add_argument('--val-annotations', type=str, default='val.json',
                       help='Name of the validation annotations JSON file')
    parser.add_argument('--dataset-size', type=int, default=2000,
                       help='Estimated dataset size for instance recommendation')
    
    # Dataset Filtering Parameters
    parser.add_argument('--class-group', type=str, default='surfaces',
                       choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth'],
                       help='Subset of classes to train on')
    parser.add_argument('--xray-type', type=str, default='bitewing',
                       help='Filter dataset by X-ray type (comma-separated for multiple: bitewing,periapical,panoramic)')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit dataset size for testing')
    
    # Model Architecture
    parser.add_argument('--embed-dim', type=int, default=768,
                       help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                       help='Transformer depth')
    parser.add_argument('--num-heads', type=int, default=12,
                       help='Number of attention heads')
    
    # Loss Function Configuration
    parser.add_argument('--loss-function', type=str, default='dicefocal',
                       choices=['crossentropy', 'dicefocal', 'focal'],
                       help='Loss function to use')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='Whether to use class weights')
    
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
        job_suffix = 'transunet'
    
    # Auto-select instance type if not specified
    if not args.instance_type:
        args.instance_type = get_recommended_instance_type(args.dataset_size)
        logger.info(f"🖥️ Auto-selected instance type: {args.instance_type}")
    
    # Validation data defaults to train data if not specified
    validation_data = args.validation_data or args.train_data
    
    # Generate job name with class group and xray type info
    job_suffix = f"{args.class_group}"
    if args.xray_type:
        job_suffix += f"-{args.xray_type.replace(',', '-')}"
    if args.max_images:
        job_suffix += f"-subset{args.max_images}"
    
    if args.test_mode:
        job_suffix += "-test"
    else:
        job_suffix += "-v0"
    
    job_name = args.job_name or f'unet-trans-{job_suffix}'
    
    output_path = args.output_path or f's3://{default_bucket}/dental-classic-output'
    code_location = args.code_location or f's3://{default_bucket}/dental-classic-code'
    
    # Build hyperparameters matching train.py expectations
    hyperparameters = {
        # Core training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        
        # Dataset configuration - Updated for separate train/val files
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'class_group': args.class_group,
        
        # Model Architecture
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        
        # Loss function configuration
        'loss_function': args.loss_function,
        'use_class_weights': args.use_class_weights,
    }

    # Add xray_type if specified
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type

    # Add optional parameters
    if args.max_images:
        hyperparameters['max_images'] = args.max_images

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}
    
    # Estimate training time
    estimated_time = estimate_training_time(args.dataset_size, args.batch_size, args.epochs)
    logger.info(f"⏱️ Estimated training time: {estimated_time}")
    
    # Configuration summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🦷 TRANSUNET TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"📝 Job name: {job_name}")
    logger.info(f"🖥️ Instance: {args.instance_type} ({'spot' if args.use_spot else 'on-demand'})")
    logger.info(f"🧠 Model: TransUNet ({args.embed_dim}D, {args.depth} layers)")
    logger.info(f"📊 Data: {args.train_data}")
    logger.info(f"📄 Train annotations: {args.train_annotations}")
    logger.info(f"📄 Val annotations: {args.val_annotations}")
    if args.xray_type:
        logger.info(f"🩻 X-ray type filter: {args.xray_type}")
    logger.info(f"🎯 Class group: {args.class_group}")
    if args.class_group == 'bone-loss':
        logger.info(f"📦 Using BOUNDING BOXES for {args.class_group} (not segmentation)")
    logger.info(f"🎛️ Batch size: {args.batch_size}, LR: {args.learning_rate}")
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
        checkpoint_s3_uri=f's3://{default_bucket}/dental-classic-checkpoints' if args.use_spot else None,
        
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
    logger.info(f"\n🚀 Launching TransUNet training...")
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
