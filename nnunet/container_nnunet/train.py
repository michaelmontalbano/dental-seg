#!/usr/bin/env python3
"""
nnU-Net training script for dental segmentation
Self-contained with defaults - no external hyperparameters needed
"""

import os
import sys
import subprocess  # ← ADD THIS LINE!


# Set environment variables BEFORE importing nnU-Net modules
# CRITICAL: These must be absolute paths
os.environ["nnUNet_raw"] = "/opt/ml/input/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/opt/ml/input/data/nnUNet_preprocessed" 
os.environ["nnUNet_results"] = "/opt/ml/model"

# IMPORTANT: Also set the base directory environment variable
os.environ["nnUNet_raw_data_base"] = "/opt/ml/input/data/nnUNet_raw"

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

def explore_directory(path, max_depth=3, current_depth=0, max_items_to_show=10):
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
            
        items_list = list(path.iterdir())
        num_items = len(items_list)
        print(f"{'  ' * current_depth}📁 {path}/ {num_items} items")
        
        # If there are too many items, show a summary instead
        if num_items > max_items_to_show:
            # Count files and directories
            files = [item for item in items_list if item.is_file()]
            dirs = [item for item in items_list if item.is_dir()]
            
            print(f"{'  ' * (current_depth + 1)}📊 Summary: {len(files)} files, {len(dirs)} directories")
            
            # Show first few files as examples
            if files:
                print(f"{'  ' * (current_depth + 1)}📄 Sample files:")
                for file in files[:3]:
                    size = file.stat().st_size
                    print(f"{'  ' * (current_depth + 2)}• {file.name} ({size:,} bytes)")
                if len(files) > 3:
                    print(f"{'  ' * (current_depth + 2)}... and {len(files) - 3} more files")
            
            # Explore subdirectories
            for dir_item in dirs:
                explore_directory(dir_item, max_depth, current_depth + 1, max_items_to_show)
        else:
            # Sort items: directories first, then files
            items = sorted(items_list, key=lambda x: (x.is_file(), x.name))
            
            for item in items:
                explore_directory(item, max_depth, current_depth + 1, max_items_to_show)
            
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
        
        # Verify that the dataset directory exists in the expected location
        expected_dataset_path = Path(os.environ['nnUNet_raw']) / self.args.task_name
        if not expected_dataset_path.exists():
            logger.error(f"Dataset not found at expected path: {expected_dataset_path}")
            # Try to find it in the input data directory
            alt_path = Path(self.args.data_dir) / self.args.task_name
            if alt_path.exists():
                logger.info(f"Found dataset at alternative path: {alt_path}")
                # Create symlink to expected location
                expected_dataset_path.symlink_to(alt_path)
                logger.info(f"Created symlink: {expected_dataset_path} -> {alt_path}")
            else:
                raise FileNotFoundError(f"Dataset {self.args.task_name} not found in either {expected_dataset_path} or {alt_path}")
        
        logger.info(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
        logger.info(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")  
        logger.info(f"nnUNet_results: {os.environ['nnUNet_results']}")
        logger.info(f"Dataset path verified: {expected_dataset_path}")
        
    def prepare_dataset(self):
        """Prepare dataset structure for nnU-Net"""
        # The dataset should already be in the correct location from setup_paths()
        dataset_dir = Path(os.environ['nnUNet_raw']) / self.args.task_name
        
        if dataset_dir.exists():
            logger.info(f"Dataset confirmed at {dataset_dir}")
            
            # Verify required subdirectories exist
            required_dirs = ['imagesTr', 'labelsTr']
            for req_dir in required_dirs:
                dir_path = dataset_dir / req_dir
                if not dir_path.exists():
                    logger.warning(f"Required directory missing: {dir_path}")
                else:
                    num_files = len(list(dir_path.glob('*')))
                    logger.info(f"Found {num_files} files in {req_dir}")
            
            # Check for dataset.json
            dataset_json = dataset_dir / 'dataset.json'
            if dataset_json.exists():
                logger.info(f"Found dataset.json: {dataset_json}")
            else:
                logger.warning(f"dataset.json not found at {dataset_json}")
            
            return
            
        logger.error(f"Dataset preparation failed - could not locate {self.args.task_name}")
        raise FileNotFoundError(f"Dataset {self.args.task_name} not found")
                    
    def run_preprocessing(self):
        """Run nnU-Net preprocessing pipeline"""
        if self.args.skip_preprocessing:
            logger.info("Skipping preprocessing as requested")
            return
            
        logger.info("Running nnU-Net preprocessing...")
        
        # Double-check environment variables before preprocessing
        logger.info("Environment check before preprocessing:")
        for key in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
            logger.info(f"  {key}: {os.environ.get(key, 'NOT SET')}")
        
        # Use nnU-Net CLI command directly
        import subprocess
        cmd = [
            "nnUNetv2_plan_and_preprocess", 
            "-d", str(self.args.dataset_id),
            # "--verify_dataset_integrity"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        #Run with environment variables explicitly passed
        env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            raise RuntimeError(f"Preprocessing failed: {result.stderr}")
        else:
            logger.info("Preprocessing completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
        
    def train(self):
        """Run nnU-Net training using CLI command"""
        logger.info(f"Starting nnU-Net training for {self.args.task_name}")
        
        # Use nnU-Net CLI command as per the guide
        import subprocess
        cmd = [
            "nnUNetv2_train",
            str(self.args.dataset_id),
            self.args.configuration,
            "all",  # Use all data (no cross-validation) as per guide
            "-tr", "nnUNetTrainer"  # Using default trainer
        ]
        
        logger.info(f"Running training command: {' '.join(cmd)}")
        
        # Run with environment variables explicitly passed
        env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        else:
            logger.info("Training completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
            
    def save_model_artifacts(self):
        """Save model artifacts for SageMaker model registry"""
        logger.info("Saving model artifacts...")
        
        # The trained models should be in nnUNet_results
        results_dir = Path(os.environ['nnUNet_results']) / "nnUNet" / self.args.configuration / self.args.task_name / f"{self.args.trainer_class_name}__{self.args.plans_name}"
        
        if not results_dir.exists():
            # Try alternative path structure
            alt_results_dir = Path(os.environ['nnUNet_results']) 
            logger.warning(f"Standard results dir not found: {results_dir}")
            logger.info(f"Looking for results in: {alt_results_dir}")
            
            # Find any .pth files
            pth_files = list(alt_results_dir.rglob("*.pth"))
            if pth_files:
                logger.info(f"Found model files: {[str(f) for f in pth_files]}")
                for pth_file in pth_files:
                    dst = Path(self.args.model_dir) / pth_file.name
                    shutil.copy2(pth_file, dst)
                    logger.info(f"Copied {pth_file} -> {dst}")
            
        else:
            # Copy from standard location
            for fold in self.args.folds:
                fold_dir = results_dir / f"fold_{fold}"
                if fold_dir.exists():
                    # Copy checkpoint files
                    for checkpoint in ['checkpoint_best.pth', 'checkpoint_final.pth']:
                        src = fold_dir / checkpoint
                        if src.exists():
                            dst = Path(self.args.model_dir) / f"fold_{fold}_{checkpoint}"
                            shutil.copy2(src, dst)
                            logger.info(f"Copied {src} -> {dst}")
                            
        # Copy any plans files
        plans_files = list(Path(os.environ['nnUNet_results']).rglob("*Plans*.json"))
        for plans_file in plans_files:
            dst = Path(self.args.model_dir) / plans_file.name
            shutil.copy2(plans_file, dst)
            logger.info(f"Copied plans: {plans_file} -> {dst}")
            
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
        
        logger.info("Model artifacts saved successfully")

def parse_args():
    parser = argparse.ArgumentParser(description='nnU-Net training for SageMaker')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, 
                        default=os.environ.get('SM_CHANNEL_NNUNET_RAW', '/opt/ml/input/data/nnUNet_raw'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
        
    parser.add_argument('--task-name', type=str, default='Dataset305_AdultTeeth',
                        help='Task name (e.g., Task101_DentalCBCT)')
    parser.add_argument('--dataset-id', type=int, default=305,  # Changed from 300
                        help='Dataset ID number')
    parser.add_argument('--configuration', type=str, default='2d',
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