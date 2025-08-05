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
        """Run nnU-Net training"""
        logger.info(f"Starting nnU-Net training for {self.args.task_name}")
        
        # Configure distributed training if multiple GPUs
        if self.args.distributed and torch.cuda.device_count() > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
        # Run training for specified fold(s)
        for fold in self.args.folds:
            logger.info(f"Training fold {fold}")
            
            run_training(
                dataset_name_or_id=self.args.task_name,
                configuration=self.args.configuration,
                fold=fold,
                trainer_class_name=self.args.trainer_class_name,
                plans_identifier=self.args.plans_name,
                pretrained_weights=self.args.pretrained_weights,
                num_gpus=self.args.num_gpus,
                use_compressed_data=self.args.use_compressed_data,
                export_validation_probabilities=self.args.export_validation_probabilities,
                continue_training=self.args.continue_training,
                only_run_validation=self.args.only_run_validation,
                disable_checkpointing=self.args.disable_checkpointing,
                val_with_best=self.args.val_with_best,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
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