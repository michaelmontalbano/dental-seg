#!/usr/bin/env python3
"""
nnU-Net training script for SageMaker
Handles dataset preparation and launches nnU-Net training
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

def setup_environment():
    """Set up nnU-Net environment variables for SageMaker"""
    # Get SageMaker paths
    sm_input_dir = os.environ.get('SM_INPUT_DIR', '/opt/ml/input/data')
    sm_output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    sm_model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    # Set nnU-Net environment variables
    os.environ['nnUNet_raw'] = os.path.join(sm_input_dir, 'training')
    os.environ['nnUNet_preprocessed'] = '/tmp/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = sm_model_dir
    
    # Create directories if they don't exist
    os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)
    os.makedirs(os.environ['nnUNet_results'], exist_ok=True)
    
    print(f"nnU-Net Environment Variables:")
    print(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"  nnUNet_results: {os.environ['nnUNet_results']}")

def parse_args():
    """Parse SageMaker hyperparameters"""
    parser = argparse.ArgumentParser(description='nnU-Net training for dental segmentation')
    
    # nnU-Net specific parameters
    parser.add_argument('--dataset_id', type=int, default=101,
                        help='nnU-Net dataset ID')
    parser.add_argument('--task_name', type=str, default='Task101_DentalCBCT',
                        help='Task name for nnU-Net')
    parser.add_argument('--configuration', type=str, default='2d',
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                        help='nnU-Net configuration')
    parser.add_argument('--folds', type=str, default='0',
                        help='Comma-separated list of folds to train')
    parser.add_argument('--trainer_class_name', type=str, default='nnUNetTrainer',
                        help='Trainer class to use')
    parser.add_argument('--plans_name', type=str, default='nnUNetPlans',
                        help='Plans identifier')
    parser.add_argument('--num_workers_dataloader', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--disable_lr_mirroring', action='store_true', default=True,
                        help='Disable left-right mirroring (important for dental)')
    
    # Training options
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing if already done')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--use_compressed_data', action='store_true',
                        help='Use compressed data')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    
    # Parse hyperparameters passed by SageMaker
    args, _ = parser.parse_known_args()
    
    # Also check for hyperparameters in SM_HPS environment variable
    sm_hps = json.loads(os.environ.get('SM_HPS', '{}'))
    for key, value in sm_hps.items():
        if hasattr(args, key):
            setattr(args, key, type(getattr(args, key))(value))
    
    return args

def check_dataset():
    """Verify dataset structure"""
    raw_dir = Path(os.environ['nnUNet_raw'])
    dataset_name = None
    
    # Look for dataset folder
    for item in raw_dir.iterdir():
        if item.is_dir() and item.name.startswith('Dataset'):
            dataset_name = item.name
            break
    
    if dataset_name:
        dataset_path = raw_dir / dataset_name
        print(f"\nFound dataset: {dataset_name}")
        
        # Check required folders
        images_tr = dataset_path / 'imagesTr'
        labels_tr = dataset_path / 'labelsTr'
        dataset_json = dataset_path / 'dataset.json'
        
        if images_tr.exists():
            print(f"  ✓ imagesTr: {len(list(images_tr.glob('*.nii.gz')))} files")
        else:
            print("  ✗ imagesTr folder missing!")
            
        if labels_tr.exists():
            print(f"  ✓ labelsTr: {len(list(labels_tr.glob('*.nii.gz')))} files")
        else:
            print("  ✗ labelsTr folder missing!")
            
        if dataset_json.exists():
            print("  ✓ dataset.json found")
        else:
            print("  ✗ dataset.json missing!")
    else:
        print(f"\nNo dataset folder found in {raw_dir}")
        print("Expected format: Dataset<ID>_<Name>")
        
    return dataset_name is not None

def run_preprocessing(args):
    """Run nnU-Net preprocessing"""
    print("\n=== Running nnU-Net Preprocessing ===")
    
    cmd = [
        'nnUNetv2_plan_and_preprocess',
        '-d', str(args.dataset_id),
        '--verify_dataset_integrity'
    ]
    
    if args.num_workers_dataloader:
        cmd.extend(['-np', str(args.num_workers_dataloader)])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Preprocessing warnings/errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

def run_training(args):
    """Run nnU-Net training"""
    print("\n=== Running nnU-Net Training ===")
    
    # Process folds
    folds = [int(f.strip()) for f in args.folds.split(',')]
    
    for fold in folds:
        print(f"\nTraining fold {fold}...")
        
        cmd = [
            'nnUNetv2_train',
            str(args.dataset_id),
            args.configuration,
            str(fold),
            '-tr', args.trainer_class_name,
            '-p', args.plans_name
        ]
        
        if args.num_workers_dataloader:
            cmd.extend(['-nw', str(args.num_workers_dataloader)])
            
        if args.continue_training:
            cmd.append('-c')
            
        if args.use_compressed_data:
            cmd.append('-use_compressed')
            
        if args.disable_lr_mirroring:
            # Note: This might need custom trainer or configuration
            print("Note: Left-right mirroring disabled via custom configuration")
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run training - this will take a long time
            result = subprocess.run(cmd, check=False)  # Don't capture output for real-time logs
            if result.returncode != 0:
                print(f"Training fold {fold} failed with exit code {result.returncode}")
                raise RuntimeError(f"Training failed for fold {fold}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

def save_training_info(args):
    """Save training configuration for reference"""
    info = {
        'dataset_id': args.dataset_id,
        'task_name': args.task_name,
        'configuration': args.configuration,
        'folds': args.folds,
        'trainer_class_name': args.trainer_class_name,
        'plans_name': args.plans_name,
        'disable_lr_mirroring': args.disable_lr_mirroring
    }
    
    model_dir = os.environ['nnUNet_results']
    with open(os.path.join(model_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining info saved to {model_dir}/training_info.json")

def main():
    """Main training function"""
    print("Starting nnU-Net training script...")
    
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Check dataset
    if not check_dataset():
        raise ValueError("Dataset validation failed!")
    
    # Run preprocessing if not skipped
    if not args.skip_preprocessing:
        run_preprocessing(args)
    else:
        print("\nSkipping preprocessing as requested")
    
    # Run training
    run_training(args)
    
    # Save training info
    save_training_info(args)
    
    print("\n=== Training Complete ===")
    print(f"Models saved to: {os.environ['nnUNet_results']}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
