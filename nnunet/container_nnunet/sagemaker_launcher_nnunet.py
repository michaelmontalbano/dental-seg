#!/usr/bin/env python3
"""
SageMaker launcher script for nnU-Net training
Bridges SageMaker hyperparameters with nnU-Net training script
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from sagemaker_training import environment

def parse_sagemaker_hyperparameters():
    """Parse hyperparameters from SageMaker environment"""
    env = environment.Environment()
    
    # Get hyperparameters
    hyperparameters = env.hyperparameters
    
    # Build command line arguments for train.py
    cmd = ['python', '/opt/ml/code/train.py']
    
    # Required arguments
    cmd.extend(['--task-name', hyperparameters.get('task_name', 'Task101_DentalCBCT')])
    cmd.extend(['--dataset-id', str(hyperparameters.get('dataset_id', 101))])
    
    # Configuration
    cmd.extend(['--configuration', hyperparameters.get('configuration', '3d_fullres')])
    
    # Folds
    folds = hyperparameters.get('folds', '0')
    if isinstance(folds, str):
        folds = [int(f) for f in folds.split(',')]
    cmd.extend(['--folds'] + [str(f) for f in folds])
    
    # Trainer configuration
    if 'trainer_class_name' in hyperparameters:
        cmd.extend(['--trainer-class-name', hyperparameters['trainer_class_name']])
    
    if 'plans_name' in hyperparameters:
        cmd.extend(['--plans-name', hyperparameters['plans_name']])
    
    # GPU configuration
    num_gpus = int(os.environ.get('SM_NUM_GPUS', 1))
    cmd.extend(['--num-gpus', str(num_gpus)])
    
    # Optional training arguments
    if hyperparameters.get('pretrained_weights'):
        cmd.extend(['--pretrained-weights', hyperparameters['pretrained_weights']])
    
    if hyperparameters.get('continue_training', False):
        cmd.append('--continue-training')
    
    if hyperparameters.get('use_compressed_data', False):
        cmd.append('--use-compressed-data')
        
    if hyperparameters.get('skip_preprocessing', False):
        cmd.append('--skip-preprocessing')
    
    # Dental specific
    if hyperparameters.get('disable_lr_mirroring', True):
        cmd.append('--disable-lr-mirroring')
    
    # Distributed training
    if num_gpus > 1 and hyperparameters.get('distributed', True):
        cmd.append('--distributed')
    
    # Data paths
    cmd.extend(['--data-dir', env.channel_input_dirs.get('training', '/opt/ml/input/data/training')])
    cmd.extend(['--model-dir', env.model_dir])
    
    # Worker configuration
    num_workers = hyperparameters.get('num_workers_dataloader', 8)
    cmd.extend(['--num-workers-dataloader', str(num_workers)])
    
    return cmd

def setup_environment():
    """Set up environment for nnU-Net"""
    # Ensure CUDA is available
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        num_gpus = int(os.environ.get('SM_NUM_GPUS', 1))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
    
    # Set up distributed training environment
    if int(os.environ.get('SM_NUM_GPUS', 1)) > 1:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        
def main():
    """Main launcher function"""
    print("Starting nnU-Net SageMaker launcher...")
    
    # Set up environment
    setup_environment()
    
    # Parse hyperparameters and build command
    cmd = parse_sagemaker_hyperparameters()
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # Execute training
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
