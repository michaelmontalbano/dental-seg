#!/usr/bin/env python3
"""
SageMaker launcher for 2-Stage YOLOv8 dental landmark segmentation
Stage 1: Use existing T01-T32 teeth detection model to extract crops
Stage 2: Train landmark segmentation model on those crops
"""

import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch
import os
import sys

def get_default_bucket(session=None):
    """Get the default SageMaker bucket"""
    session = session or sagemaker.Session()
    return session.default_bucket()

def main():
    parser = argparse.ArgumentParser(description='Launch 2-Stage YOLOv8 landmark segmentation training')

    # AWS Configuration
    parser.add_argument('--role', type=str, 
                        default='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
                        help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile to use')

    # Instance Configuration
    parser.add_argument('--instance-type', type=str, default='ml.g6.8xlarge',
                        help='SageMaker instance type')
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

    # Stage 1: Existing model configuration
    parser.add_argument('--stage1-model-s3', '--teeth-model-s3', type=str, 
                        default='s3://codentist-general/models/adult-teeth/model.tar.gz',
                        help='S3 path to Stage 1 model (teeth detection for boundary detection)')
    parser.add_argument('--stage1-model-type', '--teeth-model-type', type=str, default='teeth',
                        choices=['enamel', 'teeth', 'decay'],
                        help='Type of Stage 1 model: teeth (recommended for boundary detection), enamel, or decay')
    
    # Stage 2: Training target configuration  
    parser.add_argument('--stage2-target', type=str, default='decay',
                        choices=['decay', 'enamel', 'bone-loss', 'cej', 'ac', 'apex'],
                        help='What to train the Stage 2 model to detect within teeth boundaries')
    parser.add_argument('--stage1-conf-threshold', '--teeth-conf-threshold', type=float, default=0.3,
                        help='Confidence threshold for Stage 1 model detection')
    parser.add_argument('--crop-expand-factor', type=float, default=0.2,
                        help='How much to expand detected region crops for context')

    # Stage 2: Landmark segmentation training parameters
    parser.add_argument('--class-group', type=str, default='bone-loss',
                        choices=['bone-loss', 'teeth', 'conditions', 'surfaces', 'all', 
                                'mesial', 'distal', 'cej', 'apex', 'permanent-teeth', 
                                'primary-teeth', 'bridge', 'margin', 'enamel', 
                                'tooth-aspects', 'pulp', 'filling', 'crown', 'impaction',
                                'tooth-anatomy', 'decay', 'rct', 'parl', 'missing', 
                                'implant', 'calculus', 'distal-surface', 'mesial-surface',
                                'occlusal-surface', 'all-conditions', 'aspects-conditions',
                                'dental-work', 'decay-enamel-pulp-crown', 'parl-rct-pulp-root'],
                        help='Landmark classes to train on')
    parser.add_argument('--xray-type', type=str, default='bitewing,periapical',
                        help='Filter dataset by X-ray type')

    # ViT Model Training Parameters
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate for ViT')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size for ViT (224 recommended)')
    parser.add_argument('--model-size', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='ViT model size (base recommended)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-inferred from class_group if not specified)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Optional subset of training data for quick tests')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained ViT weights')
    
    # Monitoring parameters (from ViT approach)
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--save-best-only', type=str, default='true',
                        help='Only save the model with best validation performance')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')

    # Random seed parameter
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for training')

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
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for training job to complete (default: wait and tail logs)')

    # Framework Configuration
    parser.add_argument('--pytorch-version', type=str, default='2.3.0',
                        help='PyTorch framework version')
    parser.add_argument('--py-version', type=str, default='py311',
                        help='Python version')

    args = parser.parse_args()

    # Backward compatibility: If teeth model is explicitly specified, set model type to 'teeth'
    if '--teeth-model-s3' in sys.argv and args.stage1_model_type == 'enamel':
        args.stage1_model_type = 'teeth'

    # Setup AWS session
    if args.profile:
        boto_sess = boto3.Session(profile_name=args.profile)
        sagemaker_session = sagemaker.Session(boto_session=boto_sess)
    else:
        sagemaker_session = sagemaker.Session()

    default_bucket = get_default_bucket(sagemaker_session)
    timestamp = datetime.now().strftime('%m%d-%H%M%S')  # MMDD-HHMMSS format

    # Generate job name for 2-stage approach
    # Abbreviate class groups to keep job name short
    class_abbrev = {
        'bone-loss': 'bl',
        'conditions': 'cond',
        'surfaces': 'surf',
        'teeth': 'teeth',
        'all': 'all',
        'mesial': 'mes',
        'distal': 'dist',
        'apex': 'apex',
        'permanent-teeth': 'perm',
        'primary-teeth': 'prim',
        'bridge': 'brdg',
        'margin': 'mrgn',
        'enamel': 'enam',
        'tooth-aspects': 'asp',
        'pulp': 'pulp',
        'filling': 'fill',
        'crown': 'crwn',
        'impaction': 'imp',
        'tooth-anatomy': 'anat',
        'decay': 'decy',
        'rct': 'rct',
        'parl': 'parl',
        'missing': 'miss',
        'implant': 'impl',
        'calculus': 'calc',
        'distal-surface': 'dsrf',
        'mesial-surface': 'msrf',
        'occlusal-surface': 'osrf',
        'all-conditions': 'acnd',
        'aspects-conditions': 'aspc',
        'dental-work': 'work',
        'decay-enamel-pulp-crown': 'depc',
        'parl-rct-pulp-root': 'prpr',
        'cej': 'cej',
        'ac': 'ac'
    }
    
    # X-ray type abbreviations
    xray_abbrev_map = {
        'bitewing,periapical': 'bw-pa',
        'periapical,bitewing': 'bw-pa',
        'bitewing': 'bw',
        'periapical': 'pa',
        'panoramic': 'pan',
        'cbct': 'cbct'
    }
    
    # Build shortened job suffix
    class_short = class_abbrev.get(args.class_group, args.class_group[:4])
    job_suffix = f"2s-{class_short}"
    
    # Add abbreviated xray type if present
    if args.xray_type:
        # Check if we have a predefined abbreviation
        xray_abbrev = xray_abbrev_map.get(args.xray_type.lower().replace(' ', ''))
        if not xray_abbrev:
            # Fallback to simple abbreviation
            xray_parts = args.xray_type.split(',')
            if len(xray_parts) > 1:
                # Multiple types - use first 2 letters of each
                xray_abbrev = '-'.join([x.strip()[:2] for x in xray_parts])
            else:
                # Single type - use first 3 letters
                xray_abbrev = xray_parts[0].strip()[:3]
        job_suffix += f"-{xray_abbrev}"
    
    # Add subset if present (shortened)
    if args.subset_size:
        job_suffix += f"-n{args.subset_size}"
    
    # Create job name ensuring it's under 63 chars and follows pattern
    # Don't include seed in job name to keep it short
    job_name = args.job_name or f'yolo-{job_suffix}-{timestamp}'
    
    # Ensure job name is valid (max 63 chars, alphanumeric with hyphens)
    job_name = job_name.replace('_', '-')  # Replace underscores with hyphens
    if len(job_name) > 63:
        # Truncate if needed
        job_name = job_name[:63]
    
    output_path = args.output_path or f's3://{default_bucket}/yolo-2stage-dental-output'
    code_location = args.code_location or f's3://{default_bucket}/yolo-2stage-dental-code'

    # Build hyperparameters for 2-stage training
    hyperparameters = {
        # Core training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'model_size': args.model_size,
        'workers': args.workers,
        'seed': args.seed,
        
        # Dataset configuration
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'class_group': args.class_group,
        
        # Stage 1: Model configuration (boundary detection)
        'stage1_model_s3': args.stage1_model_s3,
        'stage1_model_type': args.stage1_model_type,
        'stage1_conf_threshold': args.stage1_conf_threshold,
        'crop_expand_factor': args.crop_expand_factor,
        
        # Stage 2: Training target (what to detect within teeth)
        'stage2_target': args.stage2_target,
        
        # Stage 2: Segmentation-specific
        'mask_ratio': args.mask_ratio,
        
        # Device configuration
        'device': 'auto',
        
        # Monitoring parameters
        'validate_every': args.validate_every,
        'save_best_only': args.save_best_only,
        'warmup_epochs': args.warmup_epochs,
        
        # Project name
        'project': '2stage_vit_dental',
    }
    
    # Add num_classes only if specified (otherwise auto-inferred)
    if args.num_classes is not None:
        hyperparameters['num_classes'] = args.num_classes

    # Add optional parameters
    if args.xray_type:
        hyperparameters['xray_type'] = args.xray_type
    if args.subset_size:
        hyperparameters['subset_size'] = args.subset_size
    if args.resume:
        hyperparameters['resume'] = args.resume
    if args.pretrained:
        hyperparameters['pretrained'] = args.pretrained
    if args.overlap_mask:
        hyperparameters['overlap_mask'] = True
    if args.wandb:
        hyperparameters['wandb'] = True

    # Clean up None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Define PyTorch Estimator for 2-Stage YOLOv8 Training
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
            'PYTHONPATH': '/opt/ml/code',
        }
    )

    # Set input channels
    inputs = {
        'train': args.train_data,
        'validation': args.validation_data
    }

    # Launch job with comprehensive logging
    print(f"🚀 === 2-Stage ViT Multi-Task Dental Training ===")
    print(f"📋 Job name: {job_name}")
    print(f"📂 Training data: {args.train_data}")
    print(f"📂 Validation data: {args.validation_data}")
    print(f"📄 Train annotations: {args.train_annotations}")
    print(f"📄 Val annotations: {args.val_annotations}")
    print(f"🧬 Stage 1 - {args.stage1_model_type.title()} model: {args.stage1_model_s3}")
    print(f"🎯 Stage 2 - Training target: {args.stage2_target}")
    print(f"📋 Target classes: {args.class_group}")
    if args.xray_type:
        print(f"📷 X-ray type filter: {args.xray_type}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"🖥️ Instance: {args.instance_type}")
    print(f"🔧 Model: ViT-{args.model_size} with multi-task heads")
    print(f"📈 Training config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Image size: {args.image_size} (ViT input)")
    print(f"   - Classes: {args.num_classes if args.num_classes else 'auto-inferred from class_group'}")
    print(f"   - Validate every: {args.validate_every} epochs")
    print(f"   - Save best only: {args.save_best_only}")
    print(f"   - Warmup epochs: {args.warmup_epochs}")
    print(f"   - Crop expand factor: {args.crop_expand_factor}")
    print(f"   - Stage 1 conf threshold: {args.stage1_conf_threshold}")
    print(f"📦 Output: {output_path}")
    print(f"⚙️ All hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    # 2-Stage specific tips
    print(f"\n💡 2-Stage Training Approach:")
    print(f"   🦷 Stage 1: Use {args.stage1_model_type} model to detect tooth boundaries")
    print(f"   🎯 Stage 2: Train ViT model to detect {args.stage2_target} within tooth boundaries")
    print(f"   🧠 Using Vision Transformer (ViT) with multi-task learning:")
    print(f"      • Classification head for presence detection")
    print(f"      • Segmentation head for pixel-level masks")
    print(f"      • Bounding box head for object localization")
    print(f"      • Keypoint head for landmark detection")
    if args.stage2_target == 'decay':
        print(f"   ✨ Clinically Important: Train decay detection within specific teeth for targeted treatment!")
    print(f"   📏 Crop expansion: {args.crop_expand_factor} for context")
    print(f"   🎭 Output: ViT model that performs multiple tasks within tooth boundaries")
    
    if args.instance_type.startswith('ml.p') or args.instance_type.startswith('ml.g'):
        print(f"✅ GPU instance selected - good for segmentation training")
    else:
        print(f"⚠️ Warning: CPU instance may be slow for segmentation training")

    print(f"\n🔄 Workflow:")
    print(f"   1. Download Stage 1 ({args.stage1_model_type}) model from S3")
    print(f"   2. Use Stage 1 model to detect tooth boundaries in training images")
    print(f"   3. Extract crops around each detected tooth")
    print(f"   4. Find {args.stage2_target} annotations within tooth crops")
    print(f"   5. Train ViT multi-task model to detect {args.stage2_target} within tooth boundaries")
    print(f"   6. Result: Trained model that detects {args.stage2_target} within specific teeth")

    if args.stage2_target == 'decay':
        print(f"\n💡 Decay Training Approach provides clinically actionable results!")
        print("   Example: Trained model will output 'Decay detected in tooth T06 at mesial surface coordinates...'")
        print("   🎯 Perfect for treatment planning and patient communication!")
    else:
        print(f"\n💡 Targeted training within tooth boundaries!")
        print(f"   Example: Trained model will detect '{args.stage2_target} detected in tooth T06 at coordinates...'")

    # Determine whether to wait (default is to wait and tail logs)
    wait_for_completion = not args.no_wait
    
    print(f"\n🚀 Starting training job...")
    if wait_for_completion:
        print("📊 Tailing logs (use --no-wait to submit without tailing)...")
    
    estimator.fit(inputs=inputs, job_name=job_name, wait=wait_for_completion)

    if not wait_for_completion:
        print(f"\n✅ 2-Stage training job '{job_name}' submitted successfully!")
        print("💡 Monitor progress in the SageMaker console:")
        print(f"🔗 https://{args.region}.console.aws.amazon.com/sagemaker/jobs/{job_name}")
        print(f"📊 Or use AWS CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
        print(f"\n📋 To tail logs, run:")
        print(f"   aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern '{job_name}'")
    else: 
        print(f"\n🎉 2-Stage training job '{job_name}' completed!")
        print(f"📁 Model artifacts saved to: {estimator.output_path}")

    return estimator

if __name__ == '__main__':
    estimator = main()
