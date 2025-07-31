#!/usr/bin/env python3
"""
SageMaker launcher for YOLOv8 dental segmentation
Configured for instance segmentation of dental landmarks
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
    parser = argparse.ArgumentParser(description='Launch SageMaker YOLOv8 segmentation training for dental landmarks')

    # AWS Configuration
    parser.add_argument('--role', type=str, 
                        default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                        help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile to use')

    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type (ml.g4dn.xlarge recommended for segmentation)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=50,
                        help='EBS volume size in GB')
    parser.add_argument('--max-run', type=int, default=200000,
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
                        choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth',
                                'mesial', 'distal', 'apex', 'permanent-teeth', 'primary-teeth',
                                'bridge', 'margin', 'enamel', 'tooth-aspects', 'pulp', 'filling',
                                'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant', 'calculus',
                                'distal-surface', 'mesial-surface', 'occlusal-surface', 'dental-work','tooth-anatomy', 'decay-enamel-pulp-coronal', 'parl-rct-pulp-root',
                                'cej', 'ac'],
                        help='Subset of classes to train on')
    parser.add_argument('--xray-type', type=str, default='bitewing,periapical',
                        help='Filter dataset by X-ray type (comma-separated for multiple: bitewing,periapical,panoramic)')

    # YOLOv8 Segmentation Training Parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Input image size for YOLOv8')
    parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Optional subset of training data for quick tests')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights or None for random init')

    # NEW: Random seed parameter
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training (good seeds: 42, 123, 3407, 777, 2023, 88, 666, 1337, 9999)')

    # Segmentation-specific parameters
    parser.add_argument('--mask-ratio', type=float, default=4,
                        help='Mask downsample ratio')
    parser.add_argument('--overlap-mask', action='store_true',
                        help='Allow overlapping masks during training')

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

    # Generate job name with class group, xray type, and seed info
    job_suffix = f"{args.class_group}-seg"
    if args.xray_type:
        job_suffix += f"-{args.xray_type.replace(',', '-')}"
    job_suffix += f"-seed{args.seed}"  # Add seed to job name
    if args.subset_size:
        job_suffix += f"-subset{args.subset_size}"
    
    job_name = args.job_name or f'yolov8-dental-{job_suffix}-{timestamp}'
    output_path = args.output_path or f's3://{default_bucket}/yolov8-seg-dental-output'
    code_location = args.code_location or f's3://{default_bucket}/yolov8-seg-dental-code'

    # Build hyperparameters for segmentation
    hyperparameters = {
        # Core training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'model_size': f'"{args.model_size}"',
        'num_classes': args.num_classes,
        'workers': args.workers,
        'seed': args.seed,  # NEW: Add seed parameter
        
        # Dataset configuration
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'class_group': args.class_group,
        
        # Segmentation-specific
        'mask_ratio': args.mask_ratio,
        
        # YOLOv8 specific
        'device': 'auto',
        'project': 'dental_segmentation',
        
    }

    # Add xray_type if specified
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type

    # Add optional parameters
    if args.subset_size:
        hyperparameters['subset_size'] = args.subset_size
    if args.resume:
        hyperparameters['resume'] = args.resume
    if args.pretrained:
        hyperparameters['pretrained'] = args.pretrained
    if args.overlap_mask:
        hyperparameters['overlap_mask'] = True
    
    # Only add wandb if it's True
    if args.wandb:
        hyperparameters['wandb'] = True

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Define PyTorch Estimator for YOLOv8 Segmentation
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
        }
    )

    # Set input channels
    inputs = {
        'train': args.train_data,
        'validation': args.validation_data
    }

    # Launch job with comprehensive logging
    print(f"🚀 === YOLOv8 Dental Segmentation Training Job Submission ===")
    print(f"📋 Job name: {job_name}")
    print(f"📂 Training data S3 path: {args.train_data}")
    print(f"📂 Validation data S3 path: {args.validation_data}")
    print(f"📄 Training annotations: {args.train_annotations}")
    print(f"📄 Validation annotations: {args.val_annotations}")
    print(f"🎯 Class group: {args.class_group}")
    print(f"🔧 Task type: segmentation (automatic)")
    if args.xray_type:
        print(f"📷 X-ray type filter: {args.xray_type}")
    print(f"🎲 Random seed: {args.seed}")  # Show seed being used
    print(f"🖥️ Instance: {args.instance_type}")
    print(f"🔧 Model size: yolov8{args.model_size}-seg")
    print(f"📈 Training config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Image size: {args.image_size}")
    print(f"   - Classes: {args.num_classes}")
    print(f"   - Mask ratio: {args.mask_ratio}")
    print(f"   - Seed: {args.seed}")
    print(f"📦 Output: {output_path}")
    print(f"⚙️ All hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    # Segmentation-specific tips
    print(f"💡 Segmentation Training Tips:")
    print(f"   - Using YOLOv8 segmentation model for instance segmentation")
    print(f"   - Mask ratio {args.mask_ratio} for efficient training")
    print(f"   - Recommended batch size: 8-16 for segmentation")
    print(f"   - Seed {args.seed} for reproducibility")
    
    if args.instance_type.startswith('ml.p') or args.instance_type.startswith('ml.g'):
        print(f"✅ GPU instance selected - good for segmentation training")
    else:
        print(f"⚠️ Warning: CPU instance may be slow for segmentation training")

    # Dataset filtering tips
    if args.class_group == 'teeth' and not args.xray_type:
        print("💡 Tip: Consider filtering by --xray-type when training on teeth classes for better specificity.")
    
    if args.subset_size and args.subset_size < 100:
        print("⚠️ Warning: Very small subset size - may not be representative for effective training.")

    # Seed-specific tips
    print("🎲 Seed recommendations:")
    print("   - Good seeds to try: 42 (default), 123, 3407, 777, 2023")
    print("   - Different seeds can give 2-5% mAP variance")
    print("   - Try 3-5 different seeds to find best performance")

    print("💡 This configuration is optimized for YOLOv8 instance segmentation!")

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
