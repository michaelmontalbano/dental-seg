#!/usr/bin/env python3
"""
SageMaker launcher for MedSAM dental segmentation
Interactive prompt-based segmentation for dental landmarks
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch
import os

def get_default_bucket(session=None):
    """Get the default SageMaker bucket"""
    session = session or sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker MedSAM training for dental landmark segmentation')

    # AWS Configuration
    parser.add_argument('--role', type=str, 
                        default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                        help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile to use')

    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.2xlarge',
                        help='SageMaker instance type (ml.g4dn.2xlarge recommended for MedSAM)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=100,
                        help='EBS volume size in GB (larger for SAM checkpoints)')
    parser.add_argument('--max-run', type=int, default=86400,
                        help='Max training time (24h)')

    # Data Configuration
    parser.add_argument('--train-data', type=str, 
                        default='s3://codentist-general/datasets/master',
                        help='S3 path to training data')
    parser.add_argument('--validation-data', type=str, 
                        default='s3://codentist-general/datasets/master',
                        help='S3 path to validation data')
    parser.add_argument('--train-annotations', type=str, default='train.json',
                        help='Name of the training annotations JSON file')
    parser.add_argument('--val-annotations', type=str, default='val.json',
                        help='Name of the validation annotations JSON file')

    # Dataset Filtering Parameters
    parser.add_argument('--class-group', type=str, default='bone-loss',
                        choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth'],
                        help='Subset of classes to train on')
    parser.add_argument('--xray-type', type=str, default='bitewing,periapical',
                        help='Filter dataset by X-ray type (comma-separated for multiple: bitewing,periapical,panoramic)')

    # MedSAM Training Parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size (smaller for MedSAM due to memory)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--image-size', type=int, default=1024,
                        help='Input image size for MedSAM (1024 recommended)')
    parser.add_argument('--sam-model-type', type=str, default='vit_h',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM backbone model size')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Optional subset of training data for quick tests')

    # MedSAM-specific parameters
    parser.add_argument('--prompt-mode', type=str, default='auto',
                        choices=['auto', 'manual', 'mixed'],
                        help='Prompt generation strategy')
    parser.add_argument('--num-prompts', type=int, default=5,
                        help='Number of prompts per image')
    parser.add_argument('--prompt-noise', type=float, default=0.1,
                        help='Noise level for prompt augmentation')
    parser.add_argument('--negative-prompt-ratio', type=float, default=0.2,
                        help='Ratio of negative prompts')
    parser.add_argument('--freeze-image-encoder', action='store_true',
                        help='Freeze SAM image encoder during training')
    parser.add_argument('--freeze-prompt-encoder', action='store_true', default=True,
                        help='Freeze SAM prompt encoder during training')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Mask prediction threshold')

    # Advanced Training Options
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')

    # SAM Checkpoint Configuration
    parser.add_argument('--sam-checkpoint', type=str, 
                        default='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                        help='URL or path to SAM checkpoint')
    parser.add_argument('--medsam-checkpoint', type=str, default=None,
                        help='Path to pre-trained MedSAM checkpoint')

    # Job Configuration
    parser.add_argument('--job-name', type=str, default=None,
                        help='SageMaker job name (auto-generated if not provided)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='S3 path for training outputs')
    parser.add_argument('--code-location', type=str, default=None,
                        help='S3 path for source code')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for training job to complete')

    # Framework Configuration
    parser.add_argument('--pytorch-version', type=str, default='2.3.0',
                        help='PyTorch framework version')
    parser.add_argument('--py-version', type=str, default='py311',
                        help='Python version')

    args = parser.parse_args()

    # Setup AWS session
    if args.profile:
        boto_sess = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=boto_sess)
    else:
        sagemaker_session = sagemaker.Session()

    default_bucket = get_default_bucket(sagemaker_session)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Generate job name with class group and prompt mode info
    job_suffix = f"{args.class_group}-medsam-{args.prompt_mode}"
    if args.xray_type:
        job_suffix += f"-{args.xray_type.replace(',', '-')}"
    if args.subset_size:
        job_suffix += f"-subset{args.subset_size}"
    
    job_name = args.job_name or f'medsam-dental-{job_suffix}-{timestamp}'
    output_path = args.output_path or f's3://{default_bucket}/medsam-dental-output'
    code_location = args.code_location or f's3://{default_bucket}/medsam-dental-code'

    # Build hyperparameters for MedSAM
    hyperparameters = {
        # Core training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'sam_model_type': args.sam_model_type,
        'num_classes': args.num_classes,
        'workers': args.workers,
        
        # Dataset configuration
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'class_group': args.class_group,
        
        # MedSAM-specific
        'prompt_mode': args.prompt_mode,
        'num_prompts': args.num_prompts,
        'prompt_noise': args.prompt_noise,
        'negative_prompt_ratio': args.negative_prompt_ratio,
        'mask_threshold': args.mask_threshold,
        
        # Model configuration
        'sam_checkpoint': args.sam_checkpoint,
        'freeze_image_encoder': args.freeze_image_encoder,
        'freeze_prompt_encoder': args.freeze_prompt_encoder,
        
        # Training optimization
        'gradient_checkpointing': args.gradient_checkpointing,
        'mixed_precision': args.mixed_precision,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        
        # Project settings
        'project': 'dental_medsam',
        'device': 'auto',
    }

    # Add xray_type if specified
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type

    # Add optional parameters
    if args.subset_size:
        hyperparameters['subset_size'] = args.subset_size
    if args.resume:
        hyperparameters['resume'] = args.resume
    if args.medsam_checkpoint:
        hyperparameters['medsam_checkpoint'] = args.medsam_checkpoint
    
    # Only add wandb if it's True
    if args.wandb:
        hyperparameters['wandb'] = True

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Define PyTorch Estimator for MedSAM
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
        use_spot_instances=False,  # Disable spot for reliability
        environment={
            'PYTHONPATH': '/opt/ml/code',
            'CUDA_VISIBLE_DEVICES': '0',  # Single GPU for MedSAM
        }
    )

    # Set input channels
    inputs = {
        'train': args.train_data,
        'validation': args.validation_data
    }

    # Launch job with comprehensive logging
    print(f"🚀 === MedSAM Dental Segmentation Training Job Submission ===")
    print(f"📋 Job name: {job_name}")
    print(f"📂 Training data S3 path: {args.train_data}")
    print(f"📂 Validation data S3 path: {args.validation_data}")
    print(f"📄 Training annotations: {args.train_annotations}")
    print(f"📄 Validation annotations: {args.val_annotations}")
    print(f"🎯 Class group: {args.class_group}")
    print(f"🔧 Model: MedSAM with {args.sam_model_type} backbone")
    if args.xray_type:
        print(f"📷 X-ray type filter: {args.xray_type}")
    print(f"🖥️ Instance: {args.instance_type}")
    print(f"📈 Training config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Image size: {args.image_size}")
    print(f"   - Classes: {args.num_classes}")
    print(f"   - Prompt mode: {args.prompt_mode}")
    print(f"   - Prompts per image: {args.num_prompts}")
    print(f"📦 Output: {output_path}")
    print(f"⚙️ All hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    # MedSAM-specific tips
    print(f"💡 MedSAM Training Tips:")
    print(f"   - Using prompt-based segmentation with {args.prompt_mode} mode")
    print(f"   - SAM backbone: {args.sam_model_type}")
    print(f"   - Image size: {args.image_size} (optimal for SAM)")
    print(f"   - Batch size: {args.batch_size} (memory-efficient for large models)")
    
    if args.freeze_image_encoder:
        print(f"   - Image encoder: FROZEN (faster training, less adaptation)")
    else:
        print(f"   - Image encoder: TRAINABLE (slower training, better adaptation)")
    
    if args.gradient_checkpointing:
        print(f"   - Gradient checkpointing: ENABLED (memory efficient)")
    
    if args.mixed_precision:
        print(f"   - Mixed precision: ENABLED (faster training)")

    # Instance recommendations
    if args.instance_type.startswith('ml.p') or args.instance_type.startswith('ml.g'):
        print(f"✅ GPU instance selected - good for MedSAM training")
        if 'xlarge' not in args.instance_type:
            print(f"💡 Consider larger instance (e.g., ml.g4dn.2xlarge) for better performance")
    else:
        print(f"⚠️ Warning: CPU instance may be very slow for MedSAM training")

    # Dataset and prompt tips
    if args.prompt_mode == 'auto' and args.num_prompts < 3:
        print("💡 Tip: Consider using more prompts (3-10) for better automatic prompt generation.")
    
    if args.class_group == 'bone-loss' and args.prompt_mode == 'manual':
        print("💡 Tip: Manual prompts work well for precise bone loss boundaries.")
    
    if args.subset_size and args.subset_size < 50:
        print("⚠️ Warning: Very small subset size - may not be representative for MedSAM training.")

    print("💡 This configuration is optimized for MedSAM interactive segmentation!")

    estimator.fit(inputs=inputs, job_name=job_name, wait=args.wait)

    if not args.wait:
        print(f"✅ Training job '{job_name}' submitted successfully!")
        print("💡 Monitor progress in the SageMaker console:")
        print(f"🔗 https://{args.region}.console.aws.amazon.com/sagemaker/jobs/{job_name}")
        print(f"📊 Or use AWS CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
    else: 
        print(f"🎉 Training job '{job_name}' completed!")

    return estimator

if __name__ == '__main__':
    estimator = main()
