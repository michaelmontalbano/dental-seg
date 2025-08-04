#!/usr/bin/env python3
"""
SageMaker launcher for nnU-Net dental segmentation
Uses containerized nnU-Net image from ECR
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.estimator import Estimator
import os

def get_default_bucket(session=None):
    """Get the default SageMaker bucket"""
    session = session or sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker nnU-Net training for dental segmentation')

    # AWS Configuration
    parser.add_argument('--role', type=str, 
                        default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                        help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile to use')
    
    # ECR Configuration
    parser.add_argument('--ecr-repo', type=str, default='nnunet-dental-segmentation',
                        help='ECR repository name')
    parser.add_argument('--image-tag', type=str, default='latest-cuda12.1-py39',
                        help='Docker image tag')

    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g6.12xlarge',
                        help='SageMaker instance type (ml.g6.12xlarge recommended for production nnU-Net)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=200,
                        help='EBS volume size in GB (nnU-Net needs more space for preprocessing)')
    parser.add_argument('--max-run', type=int, default=432000,
                        help='Max training time (5 days for nnU-Net)')

    # Data Configuration
    parser.add_argument('--train-data', type=str, 
                        default='s3://codentist-general/datasets/nnunet_dental',
                        help='S3 path to training data')
    
    # nnU-Net Specific Parameters
    parser.add_argument('--task-name', type=str, default='Task101_DentalCBCT',
                        help='nnU-Net task name')
    parser.add_argument('--dataset-id', type=int, default=101,
                        help='nnU-Net dataset ID (must be unique)')
    parser.add_argument('--folds', type=str, default='0',
                        help='Comma-separated list of folds to train (e.g., "0,1,2,3,4")')
    parser.add_argument('--configuration', type=str, default='2d',
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                        help='nnU-Net configuration')
    parser.add_argument('--trainer-class-name', type=str, default='nnUNetTrainer',
                        help='Trainer class to use')
    parser.add_argument('--plans-name', type=str, default='nnUNetPlans',
                        help='Plans identifier')
    parser.add_argument('--num-workers-dataloader', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Training Options
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing step')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--use-compressed-data', action='store_true',
                        help='Use compressed data for training')
    parser.add_argument('--disable-lr-mirroring', action='store_true', default=True,
                        help='Disable left-right mirroring for dental data')
    
    # Job Configuration
    parser.add_argument('--job-name', type=str, default=None,
                        help='SageMaker job name (auto-generated if not provided)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='S3 path for training outputs')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for training job to complete')
    parser.add_argument('--spot-instance', action='store_true',
                        help='Use spot instances for cost optimization')

    args = parser.parse_args()

    # Setup AWS session
    if args.profile:
        boto_sess = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=boto_sess)
    else:
        sagemaker_session = sagemaker.Session()

    default_bucket = get_default_bucket(sagemaker_session)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Get AWS account ID
    account_id = boto3.client('sts').get_caller_identity()['Account']

    # Generate job name
    job_suffix = f"ds{args.dataset_id}-fold{args.folds.replace(',', '')}"
    job_name = args.job_name or f'nnunet-dental-{job_suffix}-{timestamp}'
    output_path = args.output_path or f's3://{default_bucket}/nnunet-dental-output'

    # Build ECR image URI
    ecr_image_uri = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com/{args.ecr_repo}:{args.image_tag}"

    # Build hyperparameters for nnU-Net
    hyperparameters = {
        'task_name': args.task_name,
        'dataset_id': args.dataset_id,
        'configuration': args.configuration,
        'folds': args.folds,
        'trainer_class_name': args.trainer_class_name,
        'plans_name': args.plans_name,
        'num_workers_dataloader': args.num_workers_dataloader,
        'disable_lr_mirroring': args.disable_lr_mirroring,
        'distributed': args.instance_count > 1
    }
    
    # Add optional parameters
    if args.skip_preprocessing:
        hyperparameters['skip_preprocessing'] = True
    if args.continue_training:
        hyperparameters['continue_training'] = True
    if args.use_compressed_data:
        hyperparameters['use_compressed_data'] = True

    # Configure spot instance training if requested
    train_kwargs = {}
    if args.spot_instance:
        train_kwargs['use_spot_instances'] = True
        train_kwargs['max_wait'] = args.max_run + 3600  # Add 1 hour buffer

    # Configure distributed training if multiple instances
    distribution = None
    if args.instance_count > 1:
        distribution = {
            "torch_distributed": {
                "enabled": True
            }
        }

    # Create estimator using base Estimator class (not PyTorch)
    estimator = Estimator(
        image_uri=ecr_image_uri,
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        max_run=args.max_run,
        hyperparameters=hyperparameters,
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        distribution=distribution,
        environment={
            'NCCL_DEBUG': 'INFO',
            'PYTHONPATH': '/opt/ml/code'
        },
        **train_kwargs
    )

    # Set input channels
    inputs = {
        'training': args.train_data
    }

    # Launch job with comprehensive logging
    print(f"🚀 === nnU-Net Dental Segmentation Training Job Submission ===")
    print(f"🐳 ECR Image: {ecr_image_uri}")
    print(f"📋 Job name: {job_name}")
    print(f"📂 Training data S3 path: {args.train_data}")
    print(f"🏥 nnU-Net configuration:")
    print(f"   - Task name: {args.task_name}")
    print(f"   - Dataset ID: {args.dataset_id}")
    print(f"   - Configuration: {args.configuration}")
    print(f"   - Folds: {args.folds}")
    print(f"   - Trainer: {args.trainer_class_name}")
    print(f"   - Plans: {args.plans_name}")
    print(f"   - Disable LR mirroring: {args.disable_lr_mirroring}")
    print(f"🖥️ Instance: {args.instance_type} x {args.instance_count}")
    print(f"💾 Volume size: {args.volume_size} GB")
    print(f"📈 Training config:")
    print(f"   - Max runtime: {args.max_run} seconds ({args.max_run/3600:.1f} hours)")
    print(f"   - Workers: {args.num_workers_dataloader}")
    print(f"   - Spot instances: {args.spot_instance}")
    print(f"📦 Output: {output_path}")
    print(f"⚙️ All hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    # nnU-Net specific tips
    print(f"\n💡 nnU-Net Training Tips:")
    print(f"   - Auto-configures all hyperparameters based on your data")
    print(f"   - Uses 5-fold cross-validation by default")
    print(f"   - Automatically handles preprocessing and augmentation")
    print(f"   - Recommended to train all folds for best results")
    print(f"   - Training time: ~2-4 hours per fold on GPU")
    
    if args.instance_type.startswith('ml.p') or args.instance_type.startswith('ml.g'):
        print(f"✅ GPU instance selected - required for nnU-Net training")
    else:
        print(f"❌ ERROR: nnU-Net requires GPU instance for training!")

    # Dental segmentation tips
    print("\n🦷 Dental Segmentation Tips:")
    print("   - Left-right mirroring is disabled to preserve tooth identification")
    print("   - Use 2D configuration for panoramic X-rays")
    print("   - Use 3d_fullres for CBCT volumes")
    print("   - Consider ensembling multiple folds for best accuracy")

    estimator.fit(inputs=inputs, job_name=job_name, wait=args.wait)

    if not args.wait:
        print(f"\n✅ Training job '{job_name}' submitted successfully!")
        print("💡 Monitor progress in the SageMaker console:")
        print(f"🔗 https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}")
        print(f"📊 Or use AWS CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
        
        # nnU-Net specific monitoring tips
        print("\n📊 nnU-Net Training Stages:")
        print("   1. Dataset preparation")
        print("   2. Preprocessing and planning (~30-60 min)")
        print("   3. Training (1000 epochs by default)")
        print("   4. Model export and packaging")
    else: 
        print(f"\n🎉 Training job '{job_name}' completed!")
        print("📦 To use the trained model:")
        print("   1. Download from S3 output path: {output_path}/{job_name}/output/model.tar.gz")
        print("   2. Extract and use with nnU-Net inference pipeline")
        print("   3. Consider ensemble if multiple folds trained")

    return estimator

if __name__ == '__main__':
    estimator = main()
