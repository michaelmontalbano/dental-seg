#!/usr/bin/env python3
"""
SageMaker launcher for containerized nnU-Net dental segmentation
Uses custom Docker image with nnU-Net pre-installed
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker import get_execution_role
import os

def get_default_bucket(session=None):
    """Get the default SageMaker bucket"""
    session = session or sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker nnU-Net training with custom container')

    # AWS Configuration
    parser.add_argument('--role', type=str, 
                        default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                        help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile to use')

    # Container Configuration
    parser.add_argument('--image-uri', type=str, 
                        default='552401576656.dkr.ecr.us-west-2.amazonaws.com/nnunet-dental-segmentation:latest',
                        help='Docker image URI from ECR')

    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type (ml.g4dn.xlarge recommended for nnU-Net)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=100,
                        help='EBS volume size in GB (nnU-Net needs more space for preprocessing)')
    parser.add_argument('--max-run', type=int, default=432000,
                        help='Max training time (5 days for nnU-Net)')

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

    # Dataset Configuration
    parser.add_argument('--class-group', type=str, default='bone-loss',
                        choices=['bone-loss', 'teeth', 'conditions', 'surfaces', 'dental-work', 
                                'mesial', 'tooth-anatomy', 'distal', 'apex', 'permanent-teeth', 
                                'primary-teeth', 'cej', 'decay-enamel-pulp-coronal', 'parl-rct-pulp-root'],
                        help='Subset of classes to train on')
    
    # nnU-Net Specific Parameters
    parser.add_argument('--dataset-id', type=int, default=701,
                        help='nnU-Net dataset ID (must be unique)')
    parser.add_argument('--fold', type=int, default=0,
                        help='nnU-Net fold for training (0-4)')
    parser.add_argument('--output-dir', type=str, default='/tmp/nnunet_data',
                        help='Temporary directory for nnU-Net data conversion')
    
    # Job Configuration
    parser.add_argument('--job-name', type=str, default=None,
                        help='SageMaker job name (auto-generated if not provided)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='S3 path for training outputs')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for training job to complete')

    args = parser.parse_args()

    # Setup AWS session
    if args.profile:
        boto_sess = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=boto_sess)
    else:
        sagemaker_session = sagemaker.Session()

    default_bucket = get_default_bucket(sagemaker_session)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Generate job name
    job_suffix = f"{args.class_group}-nnunet-ds{args.dataset_id}-fold{args.fold}"
    job_name = args.job_name or f'nnunet-dental-{job_suffix}-{timestamp}'
    output_path = args.output_path or f's3://{default_bucket}/nnunet-dental-output'

    # Build hyperparameters
    hyperparameters = {
        # Data parameters
        'train-annotations': args.train_annotations,
        'val-annotations': args.val_annotations,
        'class-group': args.class_group,
        'dataset-id': args.dataset_id,
        'fold': args.fold,
        'output-dir': args.output_dir,
    }

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Create SageMaker Estimator with custom container
    estimator = sagemaker.estimator.Estimator(
        image_uri=args.image_uri,
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        max_run=args.max_run,
        hyperparameters=hyperparameters,
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        use_spot_instances=False,  # Disable spot for reliability
        environment={
            'PYTHONPATH': '/opt/ml/code',
            'nnUNet_raw': '/tmp/nnunet_data/nnUNet_raw',
            'nnUNet_preprocessed': '/tmp/nnunet_data/nnUNet_preprocessed',
            'nnUNet_results': '/tmp/nnunet_data/nnUNet_results',
        }
    )

    # Set input channels
    inputs = {
        'train': args.train_data,
        'validation': args.validation_data
    }

    # Launch job with comprehensive logging
    print(f"🚀 === nnU-Net Dental Segmentation Training Job Submission ===")
    print(f"📋 Job name: {job_name}")
    print(f"🐳 Docker image: {args.image_uri}")
    print(f"📂 Training data S3 path: {args.train_data}")
    print(f"📂 Validation data S3 path: {args.validation_data}")
    print(f"📄 Training annotations: {args.train_annotations}")
    print(f"📄 Validation annotations: {args.val_annotations}")
    print(f"🎯 Class group: {args.class_group}")
    print(f"🔧 Task type: semantic segmentation")
    print(f"🏥 nnU-Net specific config:")
    print(f"   - Dataset ID: {args.dataset_id}")
    print(f"   - Fold: {args.fold}")
    print(f"🖥️ Instance: {args.instance_type}")
    print(f"💾 Volume size: {args.volume_size} GB")
    print(f"📈 Training config:")
    print(f"   - Max runtime: {args.max_run} seconds")
    print(f"📦 Output: {output_path}")
    print(f"⚙️ All hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    # nnU-Net specific tips
    print(f"💡 nnU-Net Training Tips:")
    print(f"   - Auto-configures all hyperparameters based on your data")
    print(f"   - Uses 5-fold cross-validation by default")
    print(f"   - Automatically handles preprocessing and augmentation")
    print(f"   - Recommended to train all folds for best results")
    print(f"   - Training time: ~2-4 hours per fold on GPU")
    
    if args.instance_type.startswith('ml.p') or args.instance_type.startswith('ml.g'):
        print(f"✅ GPU instance selected - required for nnU-Net training")
    else:
        print(f"❌ ERROR: nnU-Net requires GPU instance for training!")

    # Medical segmentation tips
    print("🏥 Medical Segmentation Tips:")
    print("   - nnU-Net excels at medical image segmentation")
    print("   - Automatic configuration based on dataset properties")
    print("   - Consider training all 5 folds and ensembling")
    print("   - Results will be in nnU-Net format (.nii.gz)")

    # Container-specific notes
    print("\n🐳 Container Notes:")
    print("   - Using pre-built Docker image with nnU-Net installed")
    print("   - All dependencies are included in the container")
    print("   - No need to install packages during runtime")

    estimator.fit(inputs=inputs, job_name=job_name, wait=args.wait)

    if not args.wait:
        print(f"✅ Training job '{job_name}' submitted successfully!")
        print("💡 Monitor progress in the SageMaker console:")
        print(f"🔗 https://{args.region}.console.aws.amazon.com/sagemaker/jobs/{job_name}")
        print(f"📊 Or use AWS CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
        
        # nnU-Net specific monitoring tips
        print("\n📊 nnU-Net Training Stages:")
        print("   1. Dataset conversion (COCO → nnU-Net format)")
        print("   2. Preprocessing and planning")
        print("   3. Training (1000 epochs by default)")
        print("   4. Model export")
    else: 
        print(f"🎉 Training job '{job_name}' completed!")
        print("📦 To use the trained model:")
        print("   1. Download from S3 output path")
        print("   2. Use nnU-Net inference pipeline")
        print("   3. Consider ensemble if multiple folds trained")

    return estimator

if __name__ == '__main__':
    estimator = main()
