#!/usr/bin/env python3
"""
SageMaker launcher for dental X-ray classification training using train.json and val.json
Uses JSON annotation files for training data
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch

def get_default_bucket():
    """Get the default SageMaker bucket"""
    session = sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch dental X-ray classification training using JSON files')
    
    # AWS Configuration
    parser.add_argument('--role', type=str, 
                       default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                       help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2')
    parser.add_argument('--profile', type=str, default=None)
    
    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--volume-size', type=int, default=50)
    parser.add_argument('--max-run', type=int, default=10800,
                       help='3 hours max run time')
    
    # Data Configuration - Updated for JSON files
    parser.add_argument('--train-json', type=str, 
                       default='s3://codentist-general/datasets/bw_pa_pans/train.json',
                       help='S3 path to train.json file')
    parser.add_argument('--val-json', type=str, 
                       default='s3://codentist-general/datasets/bw_pa_pans/val.json',
                       help='S3 path to val.json file')
    parser.add_argument('--image-dir', type=str,
                       default='s3://codentist-general/datasets/bw_pa_pans/',
                       help='S3 path to directory containing image files (bitewing/, panoramic/, periapical/ folders)')
    
    # Training Parameters
    parser.add_argument('--remove-corner-text', action='store_true', default=True,
                       help='Remove corner text labels to prevent data leakage')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of classes')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    
    # Model Configuration
    parser.add_argument('--model-name', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3_large'],
                       help='Model architecture')
    parser.add_argument('--pretrained', type=bool, default=True,
                       help='Use pretrained weights')
    
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
    
    # Create job name
    job_name = args.job_name or f'dental-json-{args.model_name}-{timestamp}'
    
    output_path = args.output_path or f's3://{default_bucket}/dental-json-models'
    code_location = args.code_location or f's3://{default_bucket}/dental-json-code'
    
    # Hyperparameters - Updated for JSON-based training
    hyperparameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'patience': args.patience,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'remove_corner_text': args.remove_corner_text,
        'train_json': '/opt/ml/input/data/train/train.json',  # Local path in container
        'val_json': '/opt/ml/input/data/val/val.json',        # Local path in container
        'image_dir': '/opt/ml/input/data/images',             # Local path in container
    }
    
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
        use_spot_instances=False,
        environment={
            'PYTHONPATH': '/opt/ml/code/src'
        }
    )
    
    # Training inputs - Updated for JSON files
    inputs = {
        'train': args.train_json,
        'val': args.val_json,
        'images': args.image_dir
    }
    
    # Print job configuration
    print("=" * 80)
    print("🦷 DENTAL X-RAY CLASSIFICATION TRAINING (JSON-based)")
    print("=" * 80)
    print(f"Job name: {job_name}")
    print(f"Model: {args.model_name}")
    print(f"Train JSON: {args.train_json}")
    print(f"Val JSON: {args.val_json}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output path: {output_path}")
    print(f"Instance type: {args.instance_type}")
    
    print(f"\n🎯 JSON-BASED TRAINING CONFIGURATION:")
    print("📋 Using pre-split train.json and val.json files")
    print("   • No need for runtime data splitting")
    print("   • Consistent train/val splits across runs")
    print("   • Direct access to image metadata")
    
    print(f"\n🔒 DATA LEAKAGE PREVENTION:")
    if args.remove_corner_text:
        print("✅ AGGRESSIVE center cropping: ENABLED")
        print("   • Removes 25% from all sides (keeps center 75%)")  
        print("   • Eliminates equipment markers, text labels, corner artifacts")
        print("   • Crucial for production deployment")
    else:
        print("⚠️  Center cropping: DISABLED")
        print("   • HIGH RISK: Model may learn from artifacts")
        print("   • Strongly recommend enabling this feature")
    
    print(f"\n📊 EXPECTED PERFORMANCE:")
    print("   • Task: panoramic vs bitewing vs periapical classification")
    print("   • Expected validation accuracy: 60-85%")
    print("   • Random baseline: 33% (3 classes)")
    print("   • High accuracy may be normal for this task")
    
    print(f"\n🎛️ TRAINING PARAMETERS:")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Weight decay: {args.weight_decay}")
    print(f"   • Dropout rate: {args.dropout_rate}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Early stopping patience: {args.patience}")
    print(f"   • Pretrained weights: {args.pretrained}")
    
    print("-" * 40)
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Start training
    estimator.fit(
        inputs=inputs,
        job_name=job_name,
        wait=args.wait
    )
    
    if not args.wait:
        print(f"\n✅ Training job '{job_name}' started successfully!")
        
        print(f"\n🔍 MONITORING:")
        print(f"   • AWS Console: https://console.aws.amazon.com/sagemaker/")
        print(f"   • CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
        
        print(f"\n📈 WHAT TO EXPECT:")
        print("   • Training will show overfitting warnings if detected")
        print("   • Progress bars will show step-by-step training")
        print("   • Detailed per-class metrics every 5 epochs")
        print("   • Early stopping if no improvement")
        
        print(f"\n💾 OUTPUTS:")
        print(f"   • Model: {output_path}")
        print(f"   • Training metrics: Detailed JSON with overfitting analysis")
        print(f"   • Class distribution from JSON files")
        
        print(f"\n🚨 RED FLAGS TO WATCH FOR:")
        print("   • Validation accuracy > 90% (investigate data leakage)")
        print("   • Large train/val accuracy gap (>15%)")
        print("   • Perfect scores (impossible in real data)")
        print("   • Missing image file errors")
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print("   • Validation accuracy: 60-85%")
        print("   • Balanced per-class performance")
        print("   • Minimal overfitting warnings")
        print("   • Stable training convergence")
    
    return estimator

if __name__ == '__main__':
    estimator = main()
