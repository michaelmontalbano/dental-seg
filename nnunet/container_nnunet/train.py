#!/usr/bin/env python3
"""
nnU-Net training script for dental segmentation
Self-contained with defaults - no external hyperparameters needed
"""

import os
import sys

# Set environment variables BEFORE importing nnU-Net modules
os.environ["nnUNet_raw_data_base"] = "/opt/ml/input/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/opt/ml/input/data/nnUNet_preprocessed" 
os.environ["nnUNet_results"] = "/opt/ml/model"

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from sagemaker_training import environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_directory(path, max_depth=3, current_depth=0):
    """Recursively explore directory structure"""
    if current_depth > max_depth:
        return
        
    try:
        path = Path(path)
        if not path.exists():
            print(f"{'  ' * current_depth}❌ {path} (does not exist)")
            return
            
        if path.is_file():
            size = path.stat().st_size
            print(f"{'  ' * current_depth}📄 {path.name} ({size:,} bytes)")
            return
            
        print(f"{'  ' * current_depth}📁 {path}/ {len(list(path.iterdir()))} items")
        
        # Sort items: directories first, then files
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        
        for item in items:
            explore_directory(item, max_depth, current_depth + 1)
            
    except PermissionError:
        print(f"{'  ' * current_depth}🔒 {path} (permission denied)")
    except Exception as e:
        print(f"{'  ' * current_depth}⚠️  {path} (error: {e})")

def print_filesystem_debug():
    """Print complete filesystem exploration"""
    print("=" * 60)
    print("🔍 FILESYSTEM EXPLORATION")
    print("=" * 60)
    
    # Key directories to explore
    directories_to_check = [
        "/opt/ml/input/data",
        "/opt/ml/model", 
        "/tmp",
        "/opt/ml/input",
    ]

    for directory in directories_to_check:
        print(f"\n📂 Exploring: {directory}")
        explore_directory(directory, max_depth=4)

    print(f"\n🌍 ENVIRONMENT VARIABLES:")
    for key, value in os.environ.items():
        if 'nnU' in key or 'SM_' in key:
            print(f"  {key}: {value}")

    print(f"\n📍 CURRENT WORKING DIRECTORY: {os.getcwd()}")
    explore_directory(os.getcwd(), max_depth=2)
    
    print("=" * 60)

class nnUNetSageMakerTrainer:
    """Wrapper for nnU-Net training in SageMaker environment"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env = environment.Environment()
        self.setup_paths()
        
    def setup_paths(self):
        """Configure nnU-Net paths for SageMaker environment"""
        # Ensure directories exist
        Path(os.environ['nnUNet_preprocessed']).mkdir(parents=True, exist_ok=True)
        Path(os.environ['nnUNet_results']).mkdir(parents=True, exist_ok=True)
        
        # Set additional environment variables
        os.environ['nnUNet_n_proc_DA'] = str(self.args.num_workers_dataloader)
        
        logger.info(f"nnUNet_raw_data_base: {os.environ['nnUNet_raw_data_base']}")
        logger.info(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
        logger.info(f"nnUNet_results: {os.environ['nnUNet_results']}")
        
    def prepare_dataset(self):
        """Prepare dataset structure for nnU-Net"""
        # Check if dataset already exists in correct structure
        dataset_dir = Path(os.environ['nnUNet_raw_data_base']) / self.args.task_name
        
        if dataset_dir.exists():
            logger.info(f"Dataset already exists at {dataset_dir}")
            return
            
        # If not, try to find it in the data directory
        source_dataset_dir = Path(self.args.data_dir) / self.args.task_name
        if source_dataset_dir.exists():
            logger.info(f"Found dataset at {source_dataset_dir}, copying to {dataset_dir}")
            shutil.copytree(source_dataset_dir, dataset_dir)
            return
            
        logger.error(f"Could not find dataset {self.args.task_name} in {self.args.data_dir}")
        raise FileNotFoundError(f"Dataset {self.args.task_name} not found")
                    
    def run_preprocessing(self):
        """Run nnU-Net preprocessing pipeline"""
        if self.args.skip_preprocessing:
            logger.info("Skipping preprocessing as requested")
            return
            
        logger.info("Running nnU-Net preprocessing...")
        
        # Use nnU-Net CLI command directly
        import subprocess
        cmd = [
            "nnUNetv2_plan_and_preprocess", 
            "-d", str(self.args.dataset_id),
            "--verify_dataset_integrity"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed: {result.stderr}")
            raise RuntimeError(f"Preprocessing failed: {result.stderr}")
        else:
            logger.info("Preprocessing completed successfully")
        
    def train(self):
        """Run nnU-Net training using CLI command (following the guide)"""
        logger.info(f"Starting nnU-Net training for {self.args.task_name}")
        
        # Extract dataset ID from task name
        dataset_id = int(self.args.task_name.split('_')[0].replace('Dataset', ''))
        
        # Use nnU-Net CLI command as per the guide
        import subprocess
        cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            self.args.configuration,
            "all",  # Use all data (no cross-validation) as per guide
            "-tr", "nnUNetTrainer"  # Using default trainer
        ]
        
        logger.info(f"Running training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        else:
            logger.info("Training completed successfully")
            
    def save_model_artifacts(self):
        """Save model artifacts for SageMaker model registry"""
        logger.info("Saving model artifacts...")
        
        # Copy best checkpoints to model directory
        results_dir = Path(os.environ['nnUNet_results']) / self.args.task_name
        trainer_dir = results_dir / f"{self.args.trainer_class_name}__{self.args.plans_name}__{self.args.configuration}"
        
        for fold in self.args.folds:
            fold_dir = trainer_dir / f"fold_{fold}"
            if fold_dir.exists():
                # Copy checkpoint files
                for checkpoint in ['checkpoint_best.pth', 'checkpoint_final.pth']:
                    src = fold_dir / checkpoint
                    if src.exists():
                        dst = Path(self.args.model_dir) / f"fold_{fold}_{checkpoint}"
                        shutil.copy2(src, dst)
                        
        # Copy plans and dataset fingerprint
        plans_file = results_dir / f"{self.args.plans_name}.json"
        if plans_file.exists():
            shutil.copy2(plans_file, Path(self.args.model_dir) / "plans.json")
            
        # Save metadata
        metadata = {
            'task_name': self.args.task_name,
            'configuration': self.args.configuration,
            'trainer_class_name': self.args.trainer_class_name,
            'plans_name': self.args.plans_name,
            'folds': self.args.folds,
            'nnunet_version': '2.2'
        }
        
        with open(Path(self.args.model_dir) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description='nnU-Net training for SageMaker')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, 
                        default=os.environ.get('SM_CHANNEL_NNUNET_RAW', '/opt/ml/input/data/nnUNet_raw'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    # nnU-Net specific arguments - WITH DEFAULTS
    parser.add_argument('--task-name', type=str, default='Dataset300_BoneLoss',
                        help='Task name (e.g., Task101_DentalCBCT)')
    parser.add_argument('--dataset-id', type=int, default=300,
                        help='Dataset ID number')
    parser.add_argument('--configuration', type=str, default='3d_fullres',
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'])
    parser.add_argument('--folds', type=int, nargs='+', default=[0],
                        help='Which folds to train')
    parser.add_argument('--trainer-class-name', type=str, default='nnUNetTrainer',
                        help='Trainer class to use')
    parser.add_argument('--plans-name', type=str, default='nnUNetPlans',
                        help='Plans identifier')
    parser.add_argument('--preprocessor-name', type=str, default='DefaultPreprocessor',
                        help='Preprocessor class name')
    
    # Training configuration
    parser.add_argument('--pretrained-weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu-memory-target', type=int, default=8,
                        help='GPU memory target in GB for preprocessing')
    parser.add_argument('--num-workers-dataloader', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Training options
    parser.add_argument('--use-compressed-data', action='store_true',
                        help='Use compressed data for training')
    parser.add_argument('--export-validation-probabilities', action='store_true',
                        help='Export validation probabilities')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--only-run-validation', action='store_true',
                        help='Only run validation')
    parser.add_argument('--disable-checkpointing', action='store_true',
                        help='Disable checkpoint saving')
    parser.add_argument('--val-with-best', action='store_true',
                        help='Validate with best checkpoint')
    parser.add_argument('--skip-preprocessing', action='store_true', default=False,
                        help='Skip preprocessing step')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    
    # Dental segmentation specific
    parser.add_argument('--disable-lr-mirroring', action='store_true',
                        help='Disable left-right mirroring for dental data')
    
    return parser.parse_args()

def main():
    # FIRST THING: Print filesystem structure
    print_filesystem_debug()
    
    args = parse_args()
    print("Received CLI args:", sys.argv)
    print("Parsed args:", args)
    
    logger.info(f"Starting nnU-Net training with args: {args}")
    
    # Initialize trainer
    trainer = nnUNetSageMakerTrainer(args)
    
    # Prepare dataset
    trainer.prepare_dataset()
    
    # Run preprocessing
    trainer.run_preprocessing()
    
    # Run training
    trainer.train()
    
    # Save artifacts
    trainer.save_model_artifacts()
    
    logger.info("Training completed successfully")

if __name__ == '__main__':
    main()