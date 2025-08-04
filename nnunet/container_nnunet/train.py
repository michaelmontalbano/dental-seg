#!/usr/bin/env python3
"""
nnU-Net training script for dental segmentation
Handles SageMaker integration and nnU-Net API
"""

import os
import sys
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

import os

os.environ["nnUNet_raw_data_base"] = os.getenv("nnUNet_raw_data_base", "/opt/ml/input/data/nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.getenv("nnUNet_preprocessed", "/opt/ml/input/data/nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = os.getenv("RESULTS_FOLDER", "/opt/ml/model")


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
        # Update environment variables for nnU-Net
        os.environ['nnUNet_raw_data_base'] = self.args.data_dir
        os.environ['nnUNet_preprocessed'] = '/tmp/nnUNet_preprocessed'
        os.environ['nnUNet_results'] = self.args.model_dir
        os.environ['nnUNet_n_proc_DA'] = str(self.args.num_workers_dataloader)
        
        # Create directories if they don't exist
        Path(os.environ['nnUNet_preprocessed']).mkdir(parents=True, exist_ok=True)
        Path(os.environ['nnUNet_results']).mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self):
        """Prepare dataset structure for nnU-Net"""
        # Copy data from S3 to expected nnU-Net structure
        source_dir = Path(self.args.data_dir)
        target_dir = Path(os.environ['nnUNet_raw_data_base']) / self.args.task_name

        
        if not target_dir.exists():
            logger.info(f"Setting up dataset at {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy dataset.json
            dataset_json = source_dir / 'dataset.json'
            if dataset_json.exists():
                shutil.copy2(dataset_json, target_dir / 'dataset.json')
            
            # Copy images and labels
            for subdir in ['imagesTr', 'labelsTr', 'imagesTs']:
                src = source_dir / subdir
                dst = target_dir / subdir
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    
    def run_preprocessing(self):
        """Run nnU-Net preprocessing pipeline"""
        if self.args.skip_preprocessing:
            logger.info("Skipping preprocessing as requested")
            return
            
        logger.info("Running nnU-Net preprocessing...")
        from nnunetv2.experiment_planning.plan_and_preprocess import plan_and_preprocess_entry
        
        plan_and_preprocess_entry(
            dataset_id=self.args.dataset_id,
            preprocessor_name=self.args.preprocessor_name,
            plans_name=self.args.plans_name,
            gpu_memory_target=self.args.gpu_memory_target,
            preprocessing_api_kwargs=None,
            overwrite_target_spacing=None,
            overwrite_plans_name=None
        )
        
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
                        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    # nnU-Net specific arguments
    parser.add_argument('--task-name', type=str, required=True,
                        help='Task name (e.g., Task101_DentalCBCT)')
    parser.add_argument('--dataset-id', type=int, default=101,
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
    parser.add_argument('--skip-preprocessing', action='store_true',
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
