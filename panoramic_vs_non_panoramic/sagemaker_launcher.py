#!/usr/bin/env python3
"""
SageMaker launcher for panoramic vs non-panoramic dental X-ray classification
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import argparse
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker training job for panoramic vs non-panoramic classification')
    
    # Data arguments - now with hardcoded defaults
    parser.add_argument('--train-json', type=str, 
                        default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/train.json',
                        help='S3 path to train.json file')
    parser.add_argument('--val-json', type=str, 
                        default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/val.json',
                        help='S3 path to val.json file')
    parser.add_argument('--image-dir', type=str, 
                        default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/images',
                        help='S3 root path (images use s3_key from JSON)')
    
    # Training arguments
    parser.add_argument('--instance-type', type=str, default='ml.g6.12xlarge',
                        help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--model-name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3_large'],
                        help='Model architecture')
    
    # SageMaker arguments
    parser.add_argument('--role', type=str, 
                       default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                       help='SageMaker execution role ARN')
    parser.add_argument('--output-path', type=str, default=None,
                        help='S3 output path for model artifacts')
    parser.add_argument('--job-name', type=str, default=None,
                        help='Training job name')
    
    args = parser.parse_args()
    
    # Setup session (matching working unet-trans pattern)
    if getattr(args, 'profile', None):
        session = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=session)
    else:
        sagemaker_session = sagemaker.Session()
    
    # Get execution role
    role = args.role
    print(f"Using SageMaker execution role: {role}")
    
    # Set default output path
    if args.output_path:
        output_path = args.output_path
    else:
        bucket = sagemaker_session.default_bucket()
        output_path = f's3://{bucket}/panoramic-vs-non-panoramic/models'
    
    print(f"Model artifacts will be saved to: {output_path}")
    
    # Set job name
    if args.job_name:
        job_name = args.job_name
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        job_name = f'panoramic-vs-non-panoramic-{timestamp}'
    
    print(f"Training job name: {job_name}")
    
    # Training inputs - match what train.py expects
    inputs = {
        'train': os.path.dirname(args.train_json),
        'val': os.path.dirname(args.val_json),
        'images': args.image_dir
    }
    
    # Create PyTorch estimator (matching working unet-trans pattern)
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version='1.12.0',
        py_version='py38',
        output_path=output_path,
        sagemaker_session=sagemaker_session,  # Add explicit session
        hyperparameters={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'model_name': args.model_name,
            'num_classes': 2,  # Binary classification
            'image_size': 224,
            'dropout_rate': 0.5,
            'weight_decay': 1e-3,
            'patience': 10,
            'remove_corner_text': True
        },
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': r'\[(\d+)\] train_loss: ([0-9\.]+)'},
            {'Name': 'train:accuracy', 'Regex': r'\[(\d+)\] train_accuracy: ([0-9\.]+)'},
            {'Name': 'validation:loss', 'Regex': r'\[(\d+)\] val_loss: ([0-9\.]+)'},
            {'Name': 'validation:accuracy', 'Regex': r'\[(\d+)\] val_accuracy: ([0-9\.]+)'},
            {'Name': 'final:accuracy', 'Regex': r'final_accuracy: ([0-9\.]+)'},
            {'Name': 'final:train_accuracy', 'Regex': r'final_train_accuracy: ([0-9\.]+)'},
            {'Name': 'final:accuracy_gap', 'Regex': r'accuracy_gap: ([0-9\.]+)'},
            {'Name': 'overfitting:detected', 'Regex': r'overfitting_detected: (True|False)'}
        ],
        enable_sagemaker_metrics=True,
        max_run=3600 * 4,  # 4 hours max
        job_name=job_name
    )
    
    print("Starting SageMaker training job...")
    print(f"Instance type: {args.instance_type}")
    print(f"Instance count: {args.instance_count}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Model: {args.model_name}")
    print(f"Train JSON: {args.train_json}")
    print(f"Val JSON: {args.val_json}")
    print(f"Images: {args.image_dir}")
    
    # Start training
    estimator.fit(
        inputs=inputs,
        job_name=job_name
    )
    
    print("Training job completed!")
    print(f"Model artifacts location: {estimator.model_data}")
    
    # Print job details
    training_job_name = estimator.latest_training_job.name
    print(f"Training job name: {training_job_name}")
    
    # Get final metrics
    try:
        sm_client = boto3.client('sagemaker')
        response = sm_client.describe_training_job(TrainingJobName=training_job_name)
        
        if 'FinalMetricDataList' in response:
            print("\nFinal Metrics:")
            for metric in response['FinalMetricDataList']:
                print(f"  {metric['MetricName']}: {metric['Value']}")
    except Exception as e:
        print(f"Could not retrieve final metrics: {e}")
    
    print(f"\nTo deploy this model, use:")
    print(f"predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')")

if __name__ == '__main__':
    main()
