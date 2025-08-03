#!/usr/bin/env python3
"""
SageMaker launcher for MedMamba dental classification training
Supports simplified class groups on S3 dataset
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime

def get_default_bucket(session=None):
    """Get the default SageMaker bucket"""
    session = session or sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch MedMamba training on SageMaker')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Dataset parameters (simplified class groups)
    parser.add_argument('--class-group', type=str, default='bone-loss',
                        choices=['bone-loss', 'dental-work', 'decay-enamel-pulp', 
                                'surfaces', 'adult-teeth', 'primary-teeth'],
                        help='Simplified class group to train on')
    parser.add_argument('--xray-type', type=str, default=None,
                        help='Filter by X-ray type (e.g., bitewing,periapical)')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Use subset of dataset for testing')
    
    # Augmentation parameters
    parser.add_argument('--augmentation-intensity', type=str, default='medium',
                        choices=['light', 'medium', 'strong'],
                        help='Intensity of data augmentations')
    
    # SageMaker parameters
    parser.add_argument('--instance-type', type=str, default='ml.g6.8xlarge',
                        help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--use-spot', action='store_true',
                        help='Use spot instances for cost savings')
    parser.add_argument('--max-wait-time', type=int, default=172800,
                        help='Max wait time for spot instances (seconds)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Submit job without waiting for completion')
    
    # ECR image parameters
    parser.add_argument('--ecr-region', type=str, default='us-west-2',
                        help='AWS region for ECR')
    parser.add_argument('--ecr-account', type=str, default='552401576656',
                        help='AWS account ID for ECR')
    parser.add_argument('--ecr-repo', type=str, default='medmamba',
                        help='ECR repository name')
    parser.add_argument('--ecr-tag', type=str, default='latest',
                        help='ECR image tag')
    
    args = parser.parse_args()
    
    # Get SageMaker role and session
    role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
    session = sagemaker.Session()
    default_bucket = get_default_bucket(session)
    
    # Generate job name with class group
    timestamp = datetime.now().strftime('%m%d-%H%M%S')
    class_group_abbrev = {
        'bone-loss': 'bl',
        'dental-work': 'dw',
        'decay-enamel-pulp': 'dep',
        'surfaces': 'surf',
        'adult-teeth': 'at',
        'primary-teeth': 'pt'
    }
    abbrev = class_group_abbrev.get(args.class_group, args.class_group[:4])
    
    # Add xray type to job name if specified
    xray_abbrev = ''
    if args.xray_type:
        types = args.xray_type.split(',')
        xray_abbrev = '-' + '-'.join([t.strip()[:2] for t in types])
    
    job_name = f'medmamba-{abbrev}{xray_abbrev}-{timestamp}'
    
    print(f"🚀 === MedMamba Training on SageMaker ===")
    print(f"📋 Job name: {job_name}")
    print(f"🎯 Class group: {args.class_group}")
    if args.xray_type:
        print(f"📷 X-ray type filter: {args.xray_type}")
    print(f"🖥️ Instance: {args.instance_type}")
    print(f"📈 Training config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Image size: {args.image_size}")
    print(f"   - Augmentation: {args.augmentation_intensity}")
    print(f"\n🐳 Docker Image Configuration:")
    print(f"   - ECR Region: {args.ecr_region}")
    print(f"   - ECR Repository: {args.ecr_repo}:{args.ecr_tag}")
    print(f"   - Make sure to run ./build_and_push.sh first!")
    
    # Prepare hyperparameters
    hyperparameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'augmentation_intensity': args.augmentation_intensity,
        'class_group': args.class_group,
        'train_annotations': 'train.json',
        'val_annotations': 'val.json',
        'num_workers': 4,
        'weight_decay': 5e-4,
    }
    
    # Add optional parameters
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type
    if args.subset_size:
        hyperparameters['subset_size'] = args.subset_size
    
    print(f"📦 Output: s3://{default_bucket}/medmamba-output")
    
    # Configure spot instances if requested
    train_kwargs = {}
    if args.use_spot:
        train_kwargs.update({
            'use_spot_instances': True,
            'max_wait': args.max_wait_time,
            'max_run': 86400,  # 24 hours max runtime
        })
        print(f"💰 Using spot instances (max wait: {args.max_wait_time//3600} hours)")
    
    # Construct ECR image URI
    ecr_image_uri = f"{args.ecr_account}.dkr.ecr.{args.ecr_region}.amazonaws.com/{args.ecr_repo}:{args.ecr_tag}"
    print(f"🐳 Using ECR image: {ecr_image_uri}")
    
    # Get the directory of this script to find container_medmamba
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir_path = os.path.join(script_dir, 'container_medmamba')
    
    # Create PyTorch estimator with custom ECR image
    estimator = PyTorch(
        entry_point='train.py',
        source_dir=source_dir_path,  # Use absolute path to container_medmamba
        role=role,
        image_uri=ecr_image_uri,  # Use custom ECR image instead of framework/version
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        hyperparameters=hyperparameters,
        output_path=f's3://{default_bucket}/medmamba-output',
        base_job_name='medmamba-dental',
        **train_kwargs
    )
    
    # Prepare input data configuration
    inputs = {
        'train': 's3://codentist-general/datasets/master',
        'validation': 's3://codentist-general/datasets/master'
    }
    
    print(f"📂 Training data: {inputs['train']}")
    print(f"📂 Validation data: {inputs['validation']}")
    
    # Determine whether to wait for completion
    wait_for_completion = not args.no_wait
    
    if wait_for_completion:
        print("📊 Tailing logs (use --no-wait to submit without tailing)...")
    
    # Start training
    print(f"\n🚀 Starting training job...")
    estimator.fit(inputs=inputs, job_name=job_name, wait=wait_for_completion)
    
    if not wait_for_completion:
        print(f"\n✅ Training job '{job_name}' submitted successfully!")
        print(f"📊 Monitor progress in SageMaker console")
    else:
        print(f"\n🎉 Training job '{job_name}' completed!")
        print(f"📁 Model artifacts saved to: s3://{default_bucket}/medmamba-output")
    
    return estimator

if __name__ == '__main__':
    estimator = main()
