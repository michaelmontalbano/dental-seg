#!/usr/bin/env python3
"""
Training script for YOLOv8 segmentation dental landmark detection
CEJ, Alveolar Crest, and Apex segmentation using instance segmentation
"""

import subprocess
import sys
import os
import argparse
import torch
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install YOLOv8 and required packages in SageMaker environment"""
    logger.info("🔧 Installing YOLOv8 and dependencies...")
    
    try:
        # Install basic requirements first
        requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if os.path.exists(requirements_file):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
            logger.info("✅ Basic requirements installed")
        
        # Install ultralytics (YOLOv8)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics>=8.0.0'])
        logger.info("✅ Ultralytics YOLOv8 installed")
        
        # Install additional segmentation dependencies
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools'])
        logger.info("✅ COCO tools installed")
            
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        logger.info("🔄 Continuing with available packages...")

# Install requirements if in SageMaker environment
if os.getenv('SM_TRAINING_ENV') or os.getenv('SM_MODEL_DIR'):
    install_requirements()

# Import YOLOv8 after installation
try:
    from ultralytics import YOLO
    logger.info("✅ Successfully imported YOLO from ultralytics")
except ImportError:
    logger.error("❌ Failed to import YOLO. Installing ultralytics...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    from ultralytics import YOLO

# Import wandb if available
try:
    import wandb
except ImportError:
    wandb = None

def convert_annotations_to_yolo_seg(train_annotations_file='train.json', val_annotations_file='val.json', class_group='bone-loss', xray_type=None):
    """Convert separate train.json and val.json to YOLO segmentation format"""
    try:
        logger.info("🔄 Converting train.json and val.json to YOLO segmentation format...")
        
        # Import dependencies
        import json
        import shutil
        from pathlib import Path
        from PIL import Image
        from collections import Counter
        import numpy as np
        
        # SageMaker paths
        input_path = Path('/opt/ml/input/data/train')
        train_annotations_path = input_path / train_annotations_file
        val_annotations_path = input_path / val_annotations_file
        images_dir = input_path / 'images'
        output_dir = Path('/tmp/yolo_seg_dataset')
        
        # Class configuration based on class_group
# Class configuration based on class_group
        if class_group == 'bone-loss':
            class_names = ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        elif class_group == 'teeth':
            class_names = [f"T{str(i).zfill(2)}" for i in range(1, 33)] + [f"P{str(i).zfill(2)}" for i in range(1, 21)]
        elif class_group == 'conditions':
            class_names = ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                          'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant']
        elif class_group == 'surfaces':
            class_names = ['distal surface', 'occlusal surface', 'mesial surface']
        elif class_group == 'all':
            class_names = (["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"] +
                          [f"T{str(i).zfill(2)}" for i in range(1, 33)] + 
                          [f"P{str(i).zfill(2)}" for i in range(1, 21)] +
                          ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                           'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'] +
                          ['distal surface', 'occlusal surface', 'mesial surface'])
        elif class_group == 'mesial':
            class_names = ["cej mesial", "ac mesial"]
        elif class_group == 'distal':
            class_names = ["ac distal", "cej distal"]
        elif class_group == 'cej':
            class_names = ['cej mesial', 'cej distal', 'apex']
        elif class_group == 'apex':
            class_names = ["apex"]
        elif class_group == 'permanent-teeth':
            class_names = [f"T{str(i).zfill(2)}" for i in range(1, 33)]
        elif class_group == 'primary-teeth':
            class_names = [f"P{str(i).zfill(2)}" for i in range(1, 21)]
        elif class_group == 'bridge':
            class_names = ['bridge']
        elif class_group == 'margin':
            class_names = ['margin']
        elif class_group == 'enamel':
            class_names = ['enamel']
        elif class_group == 'tooth-aspects':
            class_names = ['coronal aspect', 'root aspect']
        elif class_group == 'pulp':
            class_names = ['pulp']
        elif class_group == 'filling':
            class_names = ['filling']
        elif class_group == 'crown':
            class_names = ['crown']
        elif class_group == 'impaction':
            class_names = ['impaction']
        elif class_group == 'dental-work':
            class_names = ['bridge', 'filling', 'crown', 'implant']
        elif class_group == ['tooth-anatomy']:
            class_names = ['enamel', 'pulp']
        elif class_group == 'decay':
            class_names = ['decay']
        elif class_group == 'rct':
            class_names = ['rct']
        elif class_group == 'parl':
            class_names = ['parl']
        elif class_group == 'missing':
            class_names = ['missing']
        elif class_group == 'implant':
            class_names = ['implant']
        elif class_group == 'calculus':
            class_names = ['calculus']
        elif class_group == 'distal-surface':
            class_names = ['distal surface']
        elif class_group == 'mesial-surface':
            class_names = ['mesial surface']
        elif class_group == 'occlusal-surface':
            class_names = ['occlusal surface']
        elif class_group == 'decay-enamel-pulp-coronal':
            class_names = ['decay', 'enamel', 'pulp', 'coronal aspect']
        elif class_group == 'parl-rct-pulp-root':
            class_names = ['parl', 'rct', 'pulp', 'root aspect']
        else:
            class_names = ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        logger.info(f"📁 Input: {input_path}")
        logger.info(f"📄 Train annotations: {train_annotations_path}")
        logger.info(f"📄 Val annotations: {val_annotations_path}")
        logger.info(f"🖼️ Images: {images_dir}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"🎯 Class group: {class_group}")
        logger.info(f"🎯 Classes: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
        if xray_type:
            logger.info(f"📷 X-ray type filter: {xray_type}")
        
        # Validate paths
        if not train_annotations_path.exists():
            raise FileNotFoundError(f"Train annotations not found: {train_annotations_path}")
        if not val_annotations_path.exists():
            raise FileNotFoundError(f"Val annotations not found: {val_annotations_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images not found: {images_dir}")
        
        # Load both annotation files
        with open(train_annotations_path, 'r') as f:
            train_annotations = json.load(f)
        with open(val_annotations_path, 'r') as f:
            val_annotations = json.load(f)
        
        logger.info(f"✅ Loaded train annotations: {type(train_annotations)}")
        logger.info(f"✅ Loaded val annotations: {type(val_annotations)}")
        
        # Process both train and val annotations
        def process_annotations(annotations, split_name):
            """Process COCO format annotations for segmentation"""
            if isinstance(annotations, dict) and 'images' in annotations:
                categories = {cat['id']: cat['name'] for cat in annotations['categories']}
                target_classes = set(class_names)
                
                logger.info(f"📊 {split_name} - Found categories: {len(categories)}")
                logger.info(f"🎯 {split_name} - Target categories: {[k for k,v in categories.items() if v in target_classes]}")
                
                # Group annotations by image
                image_annotations = {}
                target_ann_count = 0
                
                for ann in annotations['annotations']:
                    image_id = ann['image_id']
                    category_name = categories.get(ann['category_id'], 'unknown')
                    
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    
                    # For segmentation, we need both bbox and segmentation data
                    custom_ann = {
                        'bbox': ann.get('bbox', []),
                        'segmentation': ann.get('segmentation', []),
                        'class_name': category_name,
                        'category_id': ann['category_id']
                    }
                    image_annotations[image_id].append(custom_ann)
                    
                    if category_name in target_classes:
                        target_ann_count += 1
                
                logger.info(f"✅ {split_name} - Found {target_ann_count} target annotations")
                
                # Create processed data with xray_type filtering
                processed_data = []
                
                for img in annotations['images']:
                    img_id = img['id']
                    
                    # Apply xray_type filtering
                    if xray_type:
                        img_xray_type = img.get('xray_type', '').lower()
                        allowed_types = [t.strip().lower() for t in xray_type.split(',')]
                        if img_xray_type not in allowed_types:
                            continue
                    
                    if img_id in image_annotations:
                        target_anns = [ann for ann in image_annotations[img_id] 
                                     if ann['class_name'] in target_classes]
                        
                        if target_anns:
                            processed_item = {
                                'file_name': img['file_name'],
                                'width': img['width'],
                                'height': img['height'],
                                'id': img_id,
                                'target_annotations': target_anns
                            }
                            processed_data.append(processed_item)
                
                logger.info(f"✅ {split_name} - Processed {len(processed_data)} images with targets")
                if xray_type:
                    logger.info(f"📷 {split_name} - After X-ray type filtering: {len(processed_data)} images")
                
                return processed_data
            else:
                raise ValueError(f"Unsupported annotation format for {split_name}: {type(annotations)}")
        
        # Process train and validation data separately
        train_processed_data = process_annotations(train_annotations, "TRAIN")
        val_processed_data = process_annotations(val_annotations, "VAL")
        
        # Create YOLO dataset structure
        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_processed_data,
            'val': val_processed_data
        }
        
        # Convert segmentation to YOLO format
        def segmentation_to_yolo(segmentation, img_width, img_height):
            """Convert COCO segmentation to YOLO format"""
            if not segmentation or len(segmentation) == 0:
                return None
            
            # Handle different segmentation formats
            if isinstance(segmentation[0], list):
                # Polygon format
                polygon = segmentation[0]
                if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
                    return None
                
                # Normalize coordinates
                normalized_coords = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / img_width
                    y = polygon[i + 1] / img_height
                    normalized_coords.extend([x, y])
                
                return normalized_coords
            else:
                # RLE format - convert to polygon (simplified)
                # For now, fall back to bbox if segmentation is complex
                return None
        
        def bbox_to_yolo_seg(bbox, img_width, img_height):
            """Convert bbox to YOLO segmentation format (as rectangle)"""
            x, y, w, h = bbox
            
            # Create rectangle polygon
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            
            # Normalize coordinates
            coords = [
                x1/img_width, y1/img_height,
                x2/img_width, y2/img_height,
                x3/img_width, y3/img_height,
                x4/img_width, y4/img_height
            ]
            
            return coords
        
        # Process splits
        total_images = 0
        total_annotations = 0
        
        for split_name, split_data in splits.items():
            logger.info(f"📝 Processing {split_name}: {len(split_data)} images")
            
            for item in split_data:
                file_name = item.get('file_name')
                if not file_name:
                    continue
                
                # Copy image
                src_path = images_dir / file_name
                if not src_path.exists():
                    logger.warning(f"⚠️ Image not found: {src_path}")
                    continue
                
                dst_path = output_dir / split_name / 'images' / file_name
                shutil.copy2(src_path, dst_path)
                
                # Create label file
                img_width = item.get('width', 0)
                img_height = item.get('height', 0)
                
                if img_width == 0 or img_height == 0:
                    with Image.open(src_path) as img:
                        img_width, img_height = img.size
                
                label_file = output_dir / split_name / 'labels' / f"{Path(file_name).stem}.txt"
                
                with open(label_file, 'w') as f:
                    for ann in item.get('target_annotations', []):
                        class_name = ann['class_name']
                        if class_name not in class_to_idx:
                            continue
                        
                        class_id = class_to_idx[class_name]
                        
                        # Try to get segmentation, fall back to bbox
                        segmentation = ann.get('segmentation', [])
                        coords = segmentation_to_yolo(segmentation, img_width, img_height)
                        
                        if coords is None:
                            # Fall back to bbox as rectangle
                            bbox = ann.get('bbox', [])
                            if len(bbox) == 4:
                                coords = bbox_to_yolo_seg(bbox, img_width, img_height)
                        
                        if coords and len(coords) >= 6:  # At least 3 points
                            # Write YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                            coord_str = ' '.join([f"{coord:.6f}" for coord in coords])
                            f.write(f"{class_id} {coord_str}\n")
                            total_annotations += 1
                
                total_images += 1
        
        # Create data.yaml for YOLOv8 segmentation
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images' if len(splits['val']) > 0 else 'train/images',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"✅ Segmentation conversion completed!")
        logger.info(f"📊 Total: {total_images} images, {total_annotations} segmentation annotations")
        logger.info(f"📄 YAML created: {yaml_path}")
        
        return str(yaml_path)
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise

class DentalYOLOv8SegTrainer:
    """YOLOv8 segmentation trainer for dental landmark detection"""
    def __init__(self, model_size='n', device='auto', class_group='bone-loss'):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.class_names = self.get_class_names(class_group)
        
        logger.info(f"🤖 YOLOv8 Segmentation Trainer initialized with classes: {self.class_names}")

    @staticmethod
    def get_class_names(class_group):
        """Returns class names for the given class_group"""
        if class_group == 'bone-loss':
            return ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        elif class_group == 'teeth':
            return [f"T{str(i).zfill(2)}" for i in range(1, 33)] + [f"P{str(i).zfill(2)}" for i in range(1, 21)]
        elif class_group == 'conditions':
            return ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                    'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant']
        elif class_group == 'surfaces':
            return ['distal surface', 'occlusal surface', 'mesial surface']
        elif class_group == 'all':
            return (["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"] +
                    [f"T{str(i).zfill(2)}" for i in range(1, 33)] +
                    [f"P{str(i).zfill(2)}" for i in range(1, 21)] +
                    ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                    'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'] +
                    ['distal surface', 'occlusal surface', 'mesial surface'])
        elif class_group == 'mesial':
            return ["cej mesial", "ac mesial"]
        elif class_group == 'tooth-anatomy':
            return ['enamel','pulp']
        elif class_group == 'distal':
            return ["ac distal", "cej distal"]
        elif class_group == 'cej':
            return ['cej mesial', 'cej distal', 'apex']
        elif class_group == 'apex':
            return ["apex"]
        elif class_group == 'permanent-teeth':
            return [f"T{str(i).zfill(2)}" for i in range(1, 33)]
        elif class_group == 'primary-teeth':
            return [f"P{str(i).zfill(2)}" for i in range(1, 21)]
        elif class_group in ['bridge', 'margin', 'enamel', 'pulp', 'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant', 'calculus']:
            return [class_group]
        elif class_group == 'tooth-aspects':
            return ['coronal aspect', 'root aspect']
        elif class_group == 'distal-surface':
            return ['distal surface']
        elif class_group == 'mesial-surface':
            return ['mesial surface']
        elif class_group == 'occlusal-surface':
            return ['occlusal surface']
        elif class_group == 'dental-work':
            return ['bridge', 'filling', 'crown', 'implant']
        elif class_group == 'decay-enamel-pulp-coronal':
            return ['decay', 'enamel', 'pulp', 'coronal aspect']
        elif class_group == 'parl-rct-pulp-root':
            return ['parl', 'rct', 'pulp', 'root aspect']
        else:
            return ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        
        
        
    def load_model(self, pretrained_path=None):
        """Load YOLOv8 segmentation model"""
        try:
            if pretrained_path and os.path.exists(pretrained_path):
                logger.info(f"Loading pretrained model from {pretrained_path}")
                self.model = YOLO(pretrained_path)
            else:
                # Load YOLOv8 segmentation model
                model_name = f'yolov8{self.model_size}-seg.pt'
                logger.info(f"Loading YOLOv8 segmentation model: {model_name}")
                self.model = YOLO(model_name)
                logger.info(f"✅ Successfully loaded: {model_name}")
        
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
        
        return self.model
    
    def setup_training_config(self, args):
        """Setup training configuration for segmentation"""
        
        config = {
            # Core parameters
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.image_size,
            'device': self.device,
            'workers': min(args.workers, 4),
            'project': args.project,
            'name': args.name,
            'lr0': args.learning_rate,
            
            # Segmentation-specific parameters
            'mask_ratio': int(args.mask_ratio),
            'overlap_mask': args.overlap_mask,
            
            # YOLOv8 specific parameters
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            
            # Augmentation for dental imaging
            'hsv_h': 0.025,
            'hsv_s': 0.35,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.075,
            'scale': 0.1,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Loss parameters
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Training behavior
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'seed': args.seed,  # Now configurable!
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': args.resume,
            'amp': True,
            'profile': False,
            'patience': 100,
            
            # Inference parameters
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'verbose': True,
        }
        
        # Handle subset size
        if args.subset_size:
            config['fraction'] = min(args.subset_size / 1000.0, 1.0)
            logger.info(f"🧪 Using dataset fraction: {config['fraction']}")
        
        return config

    def log_per_class_metrics(self, results):
        try:
            logger.info("📊 === Per-Class Validation Metrics ===")
            # ... rest of the method
            logger.info("📊 === Per-Class Validation Metrics ===")
            
            # Use self.class_names for the class names
            class_names = self.class_names
            
            # Get the trainer from the model
            if hasattr(self.model, 'trainer') and self.model.trainer:
                trainer = self.model.trainer
                
                # Try to get the validator and its metrics
                if hasattr(trainer, 'validator') and trainer.validator:
                    validator = trainer.validator
                    
                    # Get the latest validation results
                    if hasattr(validator, 'metrics') and validator.metrics:
                        metrics = validator.metrics
                        logger.info(f"✅ Found validation metrics")
                        
                        # Log overall metrics first
                        logger.info(f"📊 Overall Validation Results:")
                        
                        # Box detection metrics
                        if hasattr(metrics, 'box'):
                            box_metrics = metrics.box
                            logger.info(f"📦 Box Detection:")
                            logger.info(f"   mAP50: {getattr(box_metrics, 'map50', 0.0):.4f}")
                            logger.info(f"   mAP50-95: {getattr(box_metrics, 'map', 0.0):.4f}")
                            logger.info(f"   Precision: {getattr(box_metrics, 'mp', 0.0):.4f}")
                            logger.info(f"   Recall: {getattr(box_metrics, 'mr', 0.0):.4f}")
                            
                            # Log for SageMaker
                            print(f"[FINAL] box_map50: {getattr(box_metrics, 'map50', 0.0):.4f}")
                            print(f"[FINAL] box_map50_95: {getattr(box_metrics, 'map', 0.0):.4f}")
                            print(f"[FINAL] box_precision: {getattr(box_metrics, 'mp', 0.0):.4f}")
                            print(f"[FINAL] box_recall: {getattr(box_metrics, 'mr', 0.0):.4f}")
                        
                        # Segmentation metrics
                        if hasattr(metrics, 'seg'):
                            seg_metrics = metrics.seg
                            logger.info(f"🎭 Segmentation:")
                            logger.info(f"   mAP50: {getattr(seg_metrics, 'map50', 0.0):.4f}")
                            logger.info(f"   mAP50-95: {getattr(seg_metrics, 'map', 0.0):.4f}")
                            logger.info(f"   Precision: {getattr(seg_metrics, 'mp', 0.0):.4f}")
                            logger.info(f"   Recall: {getattr(seg_metrics, 'mr', 0.0):.4f}")
                            
                            # Log for SageMaker
                            print(f"[FINAL] seg_map50: {getattr(seg_metrics, 'map50', 0.0):.4f}")
                            print(f"[FINAL] seg_map50_95: {getattr(seg_metrics, 'map', 0.0):.4f}")
                            print(f"[FINAL] seg_precision: {getattr(seg_metrics, 'mp', 0.0):.4f}")
                            print(f"[FINAL] seg_recall: {getattr(seg_metrics, 'mr', 0.0):.4f}")
                            
                            # Try to get per-class metrics
                            logger.info(f"📋 Attempting to extract per-class metrics...")
                            
                            # Check for per-class AP values
                            if hasattr(seg_metrics, 'ap') and seg_metrics.ap is not None:
                                ap_values = seg_metrics.ap
                                logger.info(f"   Found AP values with shape: {ap_values.shape if hasattr(ap_values, 'shape') else 'unknown'}")
                                
                                # Get class indices if available
                                class_indices = getattr(seg_metrics, 'ap_class_index', None)
                                if class_indices is not None:
                                    logger.info(f"   Found class indices: {class_indices}")
                                else:
                                    # Create default indices
                                    class_indices = list(range(len(class_names)))
                                    logger.info(f"   Using default class indices: {class_indices}")
                                
                                # Extract per-class metrics
                                if hasattr(ap_values, 'shape') and len(ap_values.shape) >= 1:
                                    logger.info(f"📈 Per-Class Segmentation Metrics:")
                                    
                                    for i, class_idx in enumerate(class_indices):
                                        if i < len(class_names) and i < len(ap_values):
                                            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                                            
                                            # Get AP values for this class
                                            if len(ap_values.shape) == 2:
                                                # Shape: [num_classes, num_iou_thresholds]
                                                ap50 = ap_values[i, 0] if ap_values.shape[1] > 0 else 0.0
                                                ap50_95 = ap_values[i, :].mean() if ap_values.shape[1] > 1 else ap50
                                            else:
                                                # Shape: [num_classes] - single IoU threshold
                                                ap50 = ap_values[i]
                                                ap50_95 = ap50
                                            
                                            logger.info(f"   {class_name}: AP50={ap50:.4f}, AP50-95={ap50_95:.4f}")
                                            
                                            # Log for SageMaker monitoring
                                            safe_class_name = class_name.replace(' ', '_').replace('-', '_')
                                            print(f"[FINAL] {safe_class_name}_ap50: {ap50:.4f}")
                                            print(f"[FINAL] {safe_class_name}_ap50_95: {ap50_95:.4f}")
                            
                            # Try to get per-class precision and recall
                            if hasattr(seg_metrics, 'p') and hasattr(seg_metrics, 'r'):
                                precision_values = seg_metrics.p
                                recall_values = seg_metrics.r
                                
                                if precision_values is not None and recall_values is not None:
                                    logger.info(f"📊 Per-Class Precision & Recall:")
                                    
                                    for i, class_name in enumerate(class_names):
                                        if i < len(precision_values) and i < len(recall_values):
                                            precision = precision_values[i]
                                            recall = recall_values[i]
                                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                                            
                                            logger.info(f"   {class_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                                            
                                            # Log for SageMaker monitoring
                                            safe_class_name = class_name.replace(' ', '_').replace('-', '_')
                                            print(f"[FINAL] {safe_class_name}_precision: {precision:.4f}")
                                            print(f"[FINAL] {safe_class_name}_recall: {recall:.4f}")
                                            print(f"[FINAL] {safe_class_name}_f1: {f1:.4f}")
                    
                    # Also try to access results directly from trainer
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        logger.info(f"🔍 Also found trainer.metrics")
                        # Similar processing as above but from trainer.metrics
                
                # Try alternative approach - check if results object has metrics
                if results and hasattr(results, 'results_dict'):
                    logger.info(f"📊 Found results.results_dict")
                    results_dict = results.results_dict
                    for key, value in results_dict.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"   {key}: {value:.4f}")
                            print(f"[FINAL] {key}: {value:.4f}")
                
                logger.info("✅ Per-class metrics extraction completed")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not extract per-class metrics: {e}")
            logger.info("📊 YOLOv8 training completed but detailed metrics extraction failed")
            import traceback
            traceback.print_exc()   

    def train(self, data_yaml, args):
        """Train YOLOv8 segmentation model"""
        logger.info("🚀 Starting YOLOv8 segmentation training for dental landmarks")
        logger.info(f"🎲 Using seed: {args.seed}")
        
        if not self.model:
            self.load_model(args.pretrained)
        
        config = self.setup_training_config(args)
        
        if args.wandb and wandb:
            wandb.init(project=args.project, name=args.name, config=config)
        
        try:
            logger.info(f"🎯 Starting segmentation training with {len(self.class_names)} classes")
            logger.info(f"📊 Training config: {config}")
            
            results = self.model.train(data=data_yaml, **config)
            
            logger.info("✅ Training completed successfully!")
            
            # Extract and log per-class metrics
            self.log_per_class_metrics(results)
            
            # Get best weights path
            best_weights = None
            if hasattr(self.model, 'trainer') and hasattr(self.model.trainer, 'best'):
                best_weights = self.model.trainer.best
                logger.info(f"🏆 Best weights: {best_weights}")
            
            return results, best_weights
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 segmentation for dental landmark detection')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=4)
    parser.add_argument('--learning-rate', '--learning_rate', type=float, default=0.0005)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num-classes', '--num_classes', type=int, default=5)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--image-size', '--image_size', type=int, default=1024)
    parser.add_argument('--subset-size', '--subset_size', type=int, default=None)
    parser.add_argument('--model-size', '--model_size', type=str, default='n')
    parser.add_argument('--project', type=str, default='dental_segmentation')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './runs'))
    parser.add_argument('--pretrained', type=str, default=None)
    
    # NEW: Seed parameter with good default seeds
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training (good seeds to try: 42, 123, 3407, 777, 2023)')
    
    # Segmentation-specific parameters
    parser.add_argument('--mask-ratio', '--mask_ratio', type=float, default=4.0,
                        help='Mask loss ratio for segmentation')
    parser.add_argument('--overlap-mask', '--overlap_mask', action='store_true',
                        help='Allow overlapping masks')
    
    # Dataset filtering parameters
    parser.add_argument('--train-annotations', '--train_annotations', type=str, default='train.json')
    parser.add_argument('--val-annotations', '--val_annotations', type=str, default='val.json')
    parser.add_argument('--class-group', '--class_group', type=str, default='bone-loss',
                        choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth',
                                'mesial', 'distal', 'apex', 'permanent-teeth', 'primary-teeth',
                                'bridge', 'margin', 'enamel', 'tooth-aspects', 'pulp', 'filling',
                                'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant', 'calculus',
                                'distal-surface', 'mesial-surface', 'occlusal-surface', 'dental-work', 'cej', 'ac','decay-enamel-pulp-coronal', 'parl-rct-pulp-root'])
    parser.add_argument('--xray-type', '--xray_type', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("🎯 === YOLOv8 Segmentation Dental Landmark Training ===")
    logger.info(f"📊 Arguments: {vars(args)}")
    logger.info(f"🎯 Class group: {args.class_group}")
    logger.info(f"🎲 Random seed: {args.seed}")
    if args.xray_type:
        logger.info(f"📷 X-ray type: {args.xray_type}")
    
    # Create/convert data.yaml
    if not args.data:
        logger.info("📄 No data.yaml provided, creating/converting from train.json and val.json...")
        args.data = convert_annotations_to_yolo_seg(
            train_annotations_file=args.train_annotations,
            val_annotations_file=args.val_annotations,
            class_group=args.class_group,
            xray_type=args.xray_type
        )
        logger.info(f"✅ Using data.yaml: {args.data}")
    
    # Set experiment name
    if not args.name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_parts = [f'yolov8{args.model_size}-seg', args.class_group]
        if args.xray_type:
            name_parts.append(args.xray_type)
        name_parts.append(f"seed{args.seed}")  # Include seed in name
        name_parts.append(timestamp)
        if args.subset_size:
            name_parts.append(f"subset{args.subset_size}")
        args.name = '_'.join(name_parts)
    
    # Initialize trainer
    trainer = DentalYOLOv8SegTrainer(model_size=args.model_size, device=args.device, class_group=args.class_group)

    
    try:
        # Train model
        results, best_weights = trainer.train(args.data, args)
        
        # Copy weights to model directory for SageMaker
        if args.model_dir and best_weights and os.path.exists(best_weights):
            os.makedirs(args.model_dir, exist_ok=True)
            import shutil
            shutil.copy2(best_weights, os.path.join(args.model_dir, 'best.pt'))
            logger.info(f"📦 Best weights copied to {args.model_dir}/best.pt")
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available. Training will be slow on CPU.")
    else:
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    main()
