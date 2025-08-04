#!/usr/bin/env python3
"""
nnU-Net training script for dental segmentation
Converts COCO format dental annotations to nnU-Net format and trains
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
import cv2
from PIL import Image
import shutil
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def install_nnunet():
    """Install nnU-Net and dependencies"""
    logger.info("🔧 Installing nnU-Net...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nnunetv2'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'SimpleITK'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'batchgenerators'])
        logger.info("✅ nnU-Net installed successfully")
    except Exception as e:
        logger.error(f"❌ nnU-Net installation failed: {e}")
        raise

# Install if in SageMaker environment
if os.getenv('SM_TRAINING_ENV') or os.getenv('SM_MODEL_DIR'):
    install_nnunet()

# Class groups mapping (same as YOLOv8)
CLASS_GROUPS = {
    'bone-loss': ['cej mesial', 'cej distal', 'ac distal', 'ac mesial', 'apex'],
    'teeth': [f"T{str(i).zfill(2)}" for i in range(1, 33)] + [f"P{str(i).zfill(2)}" for i in range(1, 21)],
    'conditions': ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                   'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'],
    'surfaces': ['distal surface', 'occlusal surface', 'mesial surface'],
    'dental-work': ['bridge', 'filling', 'crown', 'implant'],
    'mesial': ['cej mesial', 'ac mesial'],
    'tooth-anatomy': ['enamel', 'pulp'],
    'distal': ['ac distal', 'cej distal'],
    'apex': ['apex'],
    'permanent-teeth': [f"T{str(i).zfill(2)}" for i in range(1, 33)],
    'primary-teeth': [f"P{str(i).zfill(2)}" for i in range(1, 21)],
    'cej': ['cej mesial', 'cej distal', 'apex'],
    'decay-enamel-pulp-coronal': ['decay', 'enamel', 'pulp', 'coronal aspect'],
    'parl-rct-pulp-root': ['parl', 'rct', 'pulp', 'root aspect'],
}

class DentalNNUNetConverter:
    """Convert dental COCO annotations to nnU-Net format"""
    
    def __init__(self, class_group='bone-loss', dataset_id=701):
        self.class_group = class_group
        self.dataset_id = dataset_id
        self.dataset_name = f"Dataset{dataset_id:03d}_Dental{class_group.title()}"
        
        # Get class names
        if class_group == 'all':
            self.class_names = []
            for group_classes in CLASS_GROUPS.values():
                self.class_names.extend(group_classes)
            # Remove duplicates while preserving order
            seen = set()
            self.class_names = [x for x in self.class_names if not (x in seen or seen.add(x))]
        else:
            self.class_names = CLASS_GROUPS.get(class_group, ['cej mesial', 'cej distal', 'ac mesial', 'ac distal', 'apex'])
        
        self.class_to_id = {name: idx + 1 for idx, name in enumerate(self.class_names)}  # nnU-Net uses 1-indexed
        
        logger.info(f"🦷 nnU-Net Dental Converter initialized")
        logger.info(f"🎯 Class group: {class_group}")
        logger.info(f"📋 Classes: {self.class_names}")
        logger.info(f"🗂️ Class mapping: {self.class_to_id}")
        
    def create_mask_from_coco_segmentation(self, segmentation, img_width, img_height):
        """Convert COCO segmentation to binary mask"""
        if not segmentation:
            return None
            
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        if isinstance(segmentation[0], list):
            # Polygon format
            for polygon in segmentation:
                if len(polygon) >= 6:  # At least 3 points
                    polygon = np.array(polygon).reshape(-1, 2)
                    polygon = polygon.astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 1)
        else:
            # RLE format - skip for now, use bbox fallback
            return None
            
        return mask
    
    def create_mask_from_bbox(self, bbox, img_width, img_height):
        """Create mask from bounding box"""
        if len(bbox) != 4:
            return None
            
        x, y, w, h = bbox
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def convert_coco_to_nnunet(self, train_json_path, val_json_path, images_dir, output_dir):
        """Convert COCO format to nnU-Net format"""
        logger.info("🔄 Converting COCO to nnU-Net format...")
        
        # Create nnU-Net directory structure
        nnunet_raw = Path(output_dir) / "nnUNet_raw" / self.dataset_name
        nnunet_raw.mkdir(parents=True, exist_ok=True)
        
        imagesTr = nnunet_raw / "imagesTr"
        labelsTr = nnunet_raw / "labelsTr" 
        imagesTs = nnunet_raw / "imagesTs"
        labelsTs = nnunet_raw / "labelsTs"
        
        for folder in [imagesTr, labelsTr, imagesTs, labelsTs]:
            folder.mkdir(exist_ok=True)
        
        # Load annotations
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)
        
        def process_split(data, split_name, images_folder, labels_folder):
            """Process train or validation split"""
            logger.info(f"📝 Processing {split_name} split...")
            
            # Get categories mapping
            categories = {cat['id']: cat['name'] for cat in data['categories']}
            target_classes = set(self.class_names)
            
            # Group annotations by image
            image_annotations = {}
            for ann in data['annotations']:
                image_id = ann['image_id']
                category_name = categories.get(ann['category_id'], 'unknown')
                
                if category_name in target_classes:
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)
            
            processed_count = 0
            
            for img_info in data['images']:
                img_id = img_info['id']
                
                if img_id not in image_annotations:
                    continue
                
                # Load image
                img_filename = img_info['file_name']
                img_path = Path(images_dir) / img_filename
                
                if not img_path.exists():
                    logger.warning(f"⚠️ Image not found: {img_path}")
                    continue
                
                # Read image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                img_height, img_width = image.shape
                
                # Create combined mask for all classes
                combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # Process each annotation
                for ann in image_annotations[img_id]:
                    category_name = categories[ann['category_id']]
                    if category_name not in self.class_to_id:
                        continue
                    
                    class_id = self.class_to_id[category_name]
                    
                    # Try segmentation first, fallback to bbox
                    mask = None
                    if ann.get('segmentation'):
                        mask = self.create_mask_from_coco_segmentation(
                            ann['segmentation'], img_width, img_height
                        )
                    
                    if mask is None and ann.get('bbox'):
                        mask = self.create_mask_from_bbox(
                            ann['bbox'], img_width, img_height
                        )
                    
                    if mask is not None:
                        # Add to combined mask with class ID
                        combined_mask[mask > 0] = class_id
                
                # Save image and mask in nnU-Net format
                case_name = f"dental_{processed_count:04d}"
                
                # Save image as .nii.gz (nnU-Net format)
                img_nii = nib.Nifti1Image(image[..., np.newaxis].astype(np.float32), np.eye(4))
                nib.save(img_nii, images_folder / f"{case_name}_0000.nii.gz")
                
                # Save mask as .nii.gz
                mask_nii = nib.Nifti1Image(combined_mask[..., np.newaxis].astype(np.uint8), np.eye(4))
                nib.save(mask_nii, labels_folder / f"{case_name}.nii.gz")
                
                processed_count += 1
                
                if processed_count <= 5:
                    logger.info(f"✅ Processed {case_name}: {img_filename}")
            
            logger.info(f"✅ {split_name}: {processed_count} cases processed")
            return processed_count
        
        # Process train and validation splits
        train_count = process_split(train_data, "TRAIN", imagesTr, labelsTr)
        val_count = process_split(val_data, "VAL", imagesTs, labelsTs)
        
        # Create dataset.json
        dataset_json = {
            "channel_names": {
                "0": "grayscale"
            },
            "labels": {
                "background": 0,
                **{str(idx + 1): name for idx, name in enumerate(self.class_names)}
            },
            "numTraining": train_count,
            "numTest": val_count,
            "file_ending": ".nii.gz",
            "dataset_name": self.dataset_name,
            "reference": "Dental segmentation dataset",
            "licence": "Custom",
            "tensorImageSize": "2D"
        }
        
        with open(nnunet_raw / "dataset.json", 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        logger.info(f"✅ nnU-Net conversion completed!")
        logger.info(f"📁 Dataset created at: {nnunet_raw}")
        logger.info(f"📊 Train cases: {train_count}, Val cases: {val_count}")
        
        return str(nnunet_raw)

def train_nnunet(dataset_path, dataset_id, fold=0):
    """Train nnU-Net model"""
    logger.info("🚀 Starting nnU-Net training...")
    
    # Set nnU-Net environment variables
    os.environ['nnUNet_raw'] = str(Path(dataset_path).parent)
    os.environ['nnUNet_preprocessed'] = str(Path(dataset_path).parent / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(Path(dataset_path).parent / "nnUNet_results")
    
    try:
        # Import nnU-Net after setting environment
        from nnunetv2.experiment_planning.plan_and_preprocess import main as plan_and_preprocess
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        
        logger.info("📋 Planning and preprocessing...")
        
        # Plan and preprocess
        sys.argv = ['plan_and_preprocess', '-d', str(dataset_id), '--verify_dataset_integrity']
        plan_and_preprocess()
        
        logger.info("🎯 Starting training...")
        
        # Train model
        trainer = nnUNetTrainer(
            plans_file=f"{os.environ['nnUNet_preprocessed']}/Dataset{dataset_id:03d}_DentalBoneLoss/nnUNetPlans.json",
            fold=fold,
            output_folder=os.environ['nnUNet_results'],
            dataset_name_or_id=f"Dataset{dataset_id:03d}_DentalBoneLoss",
            configuration="2d"
        )
        
        trainer.initialize()
        trainer.run_training()
        
        logger.info("✅ nnU-Net training completed!")
        
        return os.environ['nnUNet_results']
        
    except Exception as e:
        logger.error(f"❌ nnU-Net training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Train nnU-Net for dental segmentation')
    
    # Data parameters
    parser.add_argument('--train-annotations', type=str, default='train.json')
    parser.add_argument('--val-annotations', type=str, default='val.json')
    parser.add_argument('--images-dir', type=str, default='/opt/ml/input/data/train/images')
    parser.add_argument('--output-dir', type=str, default='/tmp/nnunet_data')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './nnunet_models'))
    
    # Training parameters
    parser.add_argument('--dataset-id', type=int, default=701)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--class-group', type=str, default='bone-loss',
                        choices=['bone-loss', 'teeth', 'conditions', 'surfaces', 'dental-work', 
                                'mesial', 'tooth-anatomy', 'distal', 'apex', 'permanent-teeth', 
                                'primary-teeth', 'cej', 'decay-enamel-pulp-coronal', 'parl-rct-pulp-root'])
    
    args = parser.parse_args()
    
    logger.info("🦷 === nnU-Net Dental Segmentation Training ===")
    logger.info(f"📊 Arguments: {vars(args)}")
    
    try:
        # Setup paths
        input_path = Path('/opt/ml/input/data/train')
        train_json = input_path / args.train_annotations
        val_json = input_path / args.val_annotations
        
        # Convert to nnU-Net format
        converter = DentalNNUNetConverter(
            class_group=args.class_group,
            dataset_id=args.dataset_id
        )
        
        dataset_path = converter.convert_coco_to_nnunet(
            train_json, val_json, args.images_dir, args.output_dir
        )
        
        # Train model
        results_path = train_nnunet(dataset_path, args.dataset_id, args.fold)
        
        # Copy results to model directory for SageMaker
        if args.model_dir and results_path:
            os.makedirs(args.model_dir, exist_ok=True)
            shutil.copytree(results_path, args.model_dir, dirs_exist_ok=True)
            logger.info(f"📦 Results copied to {args.model_dir}")
        
        logger.info("🎉 nnU-Net training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
