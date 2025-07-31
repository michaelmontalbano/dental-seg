#!/usr/bin/env python3
"""
2-Stage Training Script: YOLO Tooth Detection + ViT Landmark/Condition Detection
Stage 1: Use pretrained YOLO model to detect teeth (T01-T32)
Stage 2: Train ViT on tooth crops for segmentation, bounding boxes, and keypoints
"""

import subprocess
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import yaml
import json
import numpy as np
from datetime import datetime
from PIL import Image
import cv2
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union
import boto3
from tqdm import tqdm
import timm
from torchvision import transforms
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages in SageMaker environment"""
    logger.info("🔧 Installing requirements...")
       
    try:
        requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if os.path.exists(requirements_file):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
            logger.info("✅ Requirements installed")
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        logger.info("🔄 Continuing with available packages...")

# Install requirements if in SageMaker environment
if os.getenv('SM_TRAINING_ENV') or os.getenv('SM_MODEL_DIR'):
    install_requirements()

# Import after installation
try:
    from ultralytics import YOLO
    logger.info("✅ Successfully imported YOLO")
except ImportError:
    logger.error("❌ Failed to import YOLO. Installing ultralytics...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    from ultralytics import YOLO

try:
    import wandb
except ImportError:
    wandb = None

def get_class_names(class_group: str) -> List[str]:
    """Get class names for the specified class group"""
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
    elif class_group == 'tooth-anatomy':
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
    elif class_group == 'all-conditions':
        class_names = ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                      'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant',
                      'distal surface', 'occlusal surface', 'mesial surface']
    elif class_group == 'aspects-conditions':
        class_names = ['coronal aspect', 'root aspect', 'bridge', 'impaction']
    elif class_group == 'dental-work':
        class_names = ['bridge', 'filling', 'crown', 'implant']
    elif class_group == 'decay-enamel-pulp-crown':
        class_names = ['decay', 'enamel', 'pulp', 'crown']
    elif class_group == 'parl-rct-pulp-root':
        class_names = ['parl', 'rct', 'pulp', 'enamel']
    else:
        class_names = ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
    
    return class_names

# Define which classes need segmentation vs bounding box
SEGMENTATION_CLASSES = {
    # Bone-loss landmarks
    "cej mesial", "cej distal", "ac mesial", "ac distal", "apex",
    # Dental conditions
    "enamel", "pulp", "decay", "filling", "crown", "rct",
    "distal surface", "occlusal surface", "mesial surface",
    "margin", "calculus"
}

# Small classes that need special multi-scale handling
SMALL_CLASSES = {
    "margin", "calculus", "apex"  # Added apex for better detection
}

BBOX_CLASSES = {
     "cej mesial", "cej distal", "ac mesial", "ac distal", "apex",
    "bridge", "coronal aspect", "root aspect",
    "impaction", "parl", "missing", "implant"
}

# Classes that can be represented as points
POINT_CLASSES = {
    "cej mesial", "cej distal", "ac mesial", "ac distal", "apex"
}

class TwoStageDataset(Dataset):
    """Dataset for 2-stage training: tooth crops with landmarks/conditions"""
    
    def __init__(self, annotations_file: str, images_dir: str, stage1_model_path: str,
                 class_names: List[str], transform=None, split='train',
                 crop_expand_factor: float = 0.2, conf_threshold: float = 0.3,
                 xray_type: Optional[str] = None, subset_size: Optional[int] = None,
                 cache_crops: bool = True):
        
        self.images_dir = Path(images_dir)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.split = split
        self.crop_expand_factor = crop_expand_factor
        self.conf_threshold = conf_threshold
        self.cache_crops = cache_crops
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create category lookup
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        self.target_classes = set(class_names)
        
        # Load Stage 1 model (tooth detection)
        logger.info(f"🦷 Loading Stage 1 model from {stage1_model_path}")
        self.stage1_model = YOLO(stage1_model_path)
        
        # Process annotations and create crops
        self.crop_data = []
        self._process_annotations(xray_type, subset_size)
        
        logger.info(f"✅ Created {len(self.crop_data)} tooth crops for {split}")
    
    def _process_annotations(self, xray_type: Optional[str], subset_size: Optional[int]):
        """Process annotations and extract tooth crops"""
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in self.annotations['annotations']:
            image_annotations[ann['image_id']].append(ann)
        
        # Process each image
        processed_count = 0
        
        for img_info in tqdm(self.annotations['images'], desc=f"Processing {self.split} images"):
            
            # Apply xray_type filter
            if xray_type:
                img_xray_type = img_info.get('xray_type', '').lower()
                allowed_types = [t.strip().lower() for t in xray_type.split(',')]
                if img_xray_type not in allowed_types:
                    continue
            
            # Apply subset limit
            if subset_size and processed_count >= subset_size:
                break
            
            # Get image path
            img_path = self.images_dir / img_info['file_name']
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Run Stage 1 detection
            detections = self._detect_teeth(img_path, img_info)
            
            if not detections:
                continue
            
            # Extract crops and map annotations
            img_annotations = image_annotations.get(img_info['id'], [])
            crops = self._extract_crops_with_annotations(
                img_path, img_info, detections, img_annotations
            )
            
            self.crop_data.extend(crops)
            processed_count += 1
        
        logger.info(f"📊 Processed {processed_count} images, extracted {len(self.crop_data)} crops")
    
    def _detect_teeth(self, img_path: Path, img_info: Dict) -> List[Dict]:
        """Run Stage 1 model to detect teeth"""
        try:
            results = self.stage1_model(str(img_path), conf=self.conf_threshold, verbose=True)
            
            detections = []
            for r in results:
                if r.boxes is not None:
                    for i, box in enumerate(r.boxes):
                        # Get tooth class (T01-T32)
                        class_id = int(box.cls)
                        tooth_class = self.stage1_model.names[class_id]
                        
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf)
                        
                        detection = {
                            'tooth_class': tooth_class,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting teeth in {img_path}: {e}")
            return []
    
    def _extract_crops_with_annotations(self, img_path: Path, img_info: Dict,
                                      detections: List[Dict], annotations: List[Dict]) -> List[Dict]:
        """Extract tooth crops and map annotations to each crop"""
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return []
        
        img_h, img_w = image.shape[:2]
        crops = []
        
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Expand crop region
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Expand by factor
            new_w = w * (1 + self.crop_expand_factor)
            new_h = h * (1 + self.crop_expand_factor)
            
            # Calculate expanded bbox
            crop_x1 = max(0, int(cx - new_w / 2))
            crop_y1 = max(0, int(cy - new_h / 2))
            crop_x2 = min(img_w, int(cx + new_w / 2))
            crop_y2 = min(img_h, int(cy + new_h / 2))
            
            # Extract crop
            crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Find annotations within this crop
            crop_annotations = []
            for ann in annotations:
                cat_name = self.categories.get(ann['category_id'], 'unknown')
                if cat_name not in self.target_classes:
                    continue
                
                # Transform annotation to crop space
                transformed_ann = self._transform_annotation_to_crop(
                    ann, cat_name, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
                )
                
                if transformed_ann:
                    crop_annotations.append(transformed_ann)
            
            # Only keep crops with target annotations
            if crop_annotations:
                crop_data = {
                    'image_path': str(img_path),
                    'image_id': img_info['id'],
                    'tooth_class': detection['tooth_class'],
                    'tooth_bbox': detection['bbox'],
                    'crop_bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
                    'crop_idx': det_idx,
                    'annotations': crop_annotations,
                    'crop_width': crop_x2 - crop_x1,
                    'crop_height': crop_y2 - crop_y1
                }
                
                # Cache crop image if enabled
                if self.cache_crops:
                    crop_data['crop_image'] = crop_img
                
                crops.append(crop_data)
        
        return crops
    
    def _transform_annotation_to_crop(self, ann: Dict, cat_name: str,
                                    crop_x1: int, crop_y1: int, 
                                    crop_x2: int, crop_y2: int,
                                    img_w: int, img_h: int) -> Optional[Dict]:
        """Transform annotation from image space to crop space"""
        
        transformed = {
            'category_name': cat_name,
            'category_id': self.class_to_idx[cat_name]
        }
        
        # Handle bounding box
        if 'bbox' in ann and ann['bbox']:
            x, y, w, h = ann['bbox']
            
            # Check if bbox overlaps with crop
            if not (x + w > crop_x1 and x < crop_x2 and y + h > crop_y1 and y < crop_y2):
                return None
            
            # Transform to crop coordinates
            new_x = max(0, x - crop_x1)
            new_y = max(0, y - crop_y1)
            new_x2 = min(crop_x2 - crop_x1, x + w - crop_x1)
            new_y2 = min(crop_y2 - crop_y1, y + h - crop_y1)
            
            transformed['bbox'] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
        
        # Handle segmentation
        if 'segmentation' in ann and ann['segmentation']:
            transformed_seg = []
            
            for poly in ann['segmentation']:
                if len(poly) < 6:  # Need at least 3 points
                    continue
                
                # Transform polygon points
                new_poly = []
                inside_crop = False
                
                for i in range(0, len(poly), 2):
                    px, py = poly[i], poly[i + 1]
                    
                    # Check if point is in crop
                    if crop_x1 <= px <= crop_x2 and crop_y1 <= py <= crop_y2:
                        inside_crop = True
                    
                    # Transform to crop coordinates
                    new_px = px - crop_x1
                    new_py = py - crop_y1
                    
                    # Clamp to crop boundaries
                    new_px = max(0, min(crop_x2 - crop_x1, new_px))
                    new_py = max(0, min(crop_y2 - crop_y1, new_py))
                    
                    new_poly.extend([new_px, new_py])
                
                if inside_crop and len(new_poly) >= 6:
                    transformed_seg.append(new_poly)
            
            if transformed_seg:
                transformed['segmentation'] = transformed_seg
            else:
                return None
        
        # Handle keypoints (for landmark detection)
        if 'keypoints' in ann and ann['keypoints']:
            kpts = ann['keypoints']
            if len(kpts) >= 3:  # x, y, visibility
                kx, ky, kv = kpts[0], kpts[1], kpts[2]
                
                # Check if keypoint is in crop
                if crop_x1 <= kx <= crop_x2 and crop_y1 <= ky <= crop_y2:
                    # Transform to crop coordinates
                    new_kx = kx - crop_x1
                    new_ky = ky - crop_y1
                    transformed['keypoint'] = [new_kx, new_ky, kv]
        
        # Must have at least one type of annotation
        if not any(k in transformed for k in ['bbox', 'segmentation', 'keypoint']):
            return None
        
        return transformed
    
    def __len__(self):
        return len(self.crop_data)
    
    def __getitem__(self, idx):
        crop_info = self.crop_data[idx]
        
        # Load crop image
        if self.cache_crops and 'crop_image' in crop_info:
            crop_img = crop_info['crop_image']
        else:
            # Load and extract crop
            image = cv2.imread(crop_info['image_path'])
            x1, y1, x2, y2 = crop_info['crop_bbox']
            crop_img = image[y1:y2, x1:x2]
        
        # Convert to PIL
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_img)
        
        # Apply transforms
        if self.transform:
            crop_tensor = self.transform(crop_pil)
        else:
            crop_tensor = transforms.ToTensor()(crop_pil)
        
        # Prepare targets
        targets = self._prepare_targets(crop_info)
        
        return crop_tensor, targets, crop_info
    
    def _prepare_targets(self, crop_info: Dict) -> Dict:
        """Prepare targets with proper dimension tracking and scaling"""
        
        # Get ORIGINAL crop dimensions (variable size from YOLO detection)
        orig_h = crop_info.get('crop_height', 224)
        orig_w = crop_info.get('crop_width', 224)
        
        # Ensure valid dimensions with robust error handling
        try:
            orig_h = int(orig_h) if orig_h is not None and orig_h > 0 else 224
            orig_w = int(orig_w) if orig_w is not None and orig_w > 0 else 224
        except (ValueError, TypeError):
            logger.warning(f"Invalid crop dimensions, using defaults: h={orig_h}, w={orig_w}")
            orig_h, orig_w = 224, 224
        
        # TARGET dimensions (what the model expects - fixed at 224x224)
        target_h = target_w = 224
        
        # Calculate scaling factors: original crop → model input size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        logger.debug(f"🎯 _prepare_targets: Scaling from ({orig_h}, {orig_w}) → ({target_h}, {target_w})")
        logger.debug(f"   Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        logger.debug(f"   Image: {crop_info.get('image_path', 'unknown')}, Tooth: {crop_info.get('tooth_class', 'unknown')}")
        
        targets = {
            'labels': torch.zeros(len(self.class_names)),
            'boxes': torch.zeros((0, 4)),
            'masks': torch.zeros((len(self.class_names), target_h, target_w)),  # Always 224x224
            'keypoints': torch.zeros((len(self.class_names), 3)),
            'tooth_class': crop_info['tooth_class'],
            # SAVE THE SCALING INFO for debugging/reference
            'scale_info': {
                'orig_dims': (orig_h, orig_w),
                'target_dims': (target_h, target_w),
                'scale_factors': (scale_y, scale_x)
            }
        }
        
        logger.debug(f"   Created mask tensor with shape: {targets['masks'].shape}")
        
        # Process each annotation
        boxes = []
        for ann_idx, ann in enumerate(crop_info['annotations']):
            cat_idx = ann['category_id']
            cat_name = ann['category_name']
            
            logger.debug(f"   Processing annotation {ann_idx}: {cat_name} (idx={cat_idx})")
            
            # Set label
            targets['labels'][cat_idx] = 1
            
            # Add bounding box (scaled to standard size)
            if 'bbox' in ann:
                x, y, w_box, h_box = ann['bbox']
                # Scale bbox coordinates to standard size
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                w_scaled = w_box * scale_x
                h_scaled = h_box * scale_y
                boxes.append([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled])
                logger.debug(f"     Added scaled bbox: [{x_scaled:.1f}, {y_scaled:.1f}, {x_scaled + w_scaled:.1f}, {y_scaled + h_scaled:.1f}]")
            
            # Create segmentation mask
            if 'segmentation' in ann and ann['segmentation'] and cat_name in SEGMENTATION_CLASSES:
                logger.debug(f"     Creating segmentation mask for {cat_name}")
                logger.debug(f"     Segmentation type: {type(ann['segmentation'])}")
                if isinstance(ann['segmentation'], list):
                    logger.debug(f"     Segmentation list length: {len(ann['segmentation'])}")
                    if len(ann['segmentation']) > 0:
                        logger.debug(f"     First element type: {type(ann['segmentation'][0])}")
                
                # Ensure segmentation is not empty and is a list
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    try:
                        logger.debug(f"     Creating mask with proper scaling")
                        mask = self._create_scaled_mask(
                            ann['segmentation'], 
                            orig_w, orig_h,     # Original crop dimensions
                            target_h, target_w,  # Target model dimensions (224x224)
                            scale_x, scale_y     # Pre-calculated scale factors
                        )
                        logger.debug(f"     Returned mask shape: {mask.shape}, dtype: {mask.dtype}")
                        
                        # Verify mask dimensions before conversion
                        if mask.shape != (target_h, target_w):
                            logger.error(f"     ❌ MASK DIMENSION MISMATCH! Expected ({target_h}, {target_w}), got {mask.shape}")
                            logger.error(f"     This will cause tensor dimension errors!")
                            # Try to fix by creating a new mask with correct dimensions
                            correct_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                            # Copy what we can
                            min_h = min(target_h, mask.shape[0])
                            min_w = min(target_w, mask.shape[1])
                            correct_mask[:min_h, :min_w] = mask[:min_h, :min_w]
                            mask = correct_mask
                            logger.debug(f"     Fixed mask shape to: {mask.shape}")
                        
                        # Convert to tensor
                        mask_tensor = torch.from_numpy(mask).float()
                        logger.debug(f"     Tensor shape after conversion: {mask_tensor.shape}")
                        
                        # Final assignment
                        targets['masks'][cat_idx] = mask_tensor
                        logger.debug(f"     ✅ Successfully assigned mask for {cat_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to create mask for {cat_name}: {e}")
                        logger.warning(f"     Full exception: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.warning(f"     Traceback: {traceback.format_exc()}")
                        # Continue without mask for this annotation
                        pass
                else:
                    logger.debug(f"     Skipping mask - invalid segmentation format")
            
            # Add keypoint (scaled to standard size)
            if 'keypoint' in ann and cat_name in POINT_CLASSES:
                kx, ky, kv = ann['keypoint']
                # Scale keypoint coordinates
                kx_scaled = kx * scale_x
                ky_scaled = ky * scale_y
                targets['keypoints'][cat_idx] = torch.tensor([kx_scaled, ky_scaled, kv])
        
        if boxes:
            targets['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        
        return targets
    
    def _scale_segmentation(self, segmentation, scale_x: float, scale_y: float):
        """Scale segmentation coordinates to match standard size"""
        if isinstance(segmentation, list):
            scaled_segmentation = []
            for seg in segmentation:
                if isinstance(seg, list) and len(seg) >= 6:
                    # Scale polygon coordinates
                    scaled_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] * scale_x
                        y = seg[i + 1] * scale_y
                        scaled_seg.extend([x, y])
                    scaled_segmentation.append(scaled_seg)
                elif isinstance(seg, dict):
                    # Keep RLE or other dict formats as-is
                    scaled_segmentation.append(seg)
                else:
                    scaled_segmentation.append(seg)
            return scaled_segmentation
        else:
            # Return non-list formats as-is
            return segmentation
    
    def _create_scaled_mask(self, segmentation, orig_w: int, orig_h: int, 
                           target_w: int, target_h: int, scale_x: float, scale_y: float) -> np.ndarray:
        """Create mask directly at target size with proper coordinate scaling"""
        
        # Create mask at TARGET size (224x224)
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        if not segmentation:
            return mask
        
        try:
            # Handle RLE format
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                try:
                    import pycocotools.mask as mask_util
                    
                    # Ensure RLE has correct size
                    rle_data = segmentation.copy()
                    if 'size' not in rle_data:
                        rle_data['size'] = [orig_h, orig_w]
                    
                    # Decode at original size
                    decoded_mask = mask_util.decode(rle_data)
                    
                    # Resize to target size using OpenCV
                    if decoded_mask.shape != (target_h, target_w):
                        decoded_mask = cv2.resize(decoded_mask, (target_w, target_h), 
                                                interpolation=cv2.INTER_NEAREST)
                    
                    return decoded_mask.astype(np.uint8)
                    
                except ImportError:
                    logger.warning("pycocotools not available for RLE decoding")
                except Exception as e:
                    logger.debug(f"RLE decode failed: {e}")
                    
            # Handle polygon format - scale coordinates and draw directly
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if isinstance(polygon, list) and len(polygon) >= 6:
                        try:
                            # Convert to points array
                            points = np.array(polygon, dtype=np.float32)
                            if points.size % 2 != 0:
                                continue
                                
                            points = points.reshape(-1, 2)
                            
                            # Scale coordinates from original crop space to target space
                            points[:, 0] *= scale_x  # Scale X coordinates
                            points[:, 1] *= scale_y  # Scale Y coordinates
                            
                            # Clamp to target bounds
                            points[:, 0] = np.clip(points[:, 0], 0, target_w - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, target_h - 1)
                            
                            # Convert to integer coordinates
                            points = points.astype(np.int32)
                            
                            # Draw polygon directly on target-sized mask
                            cv2.fillPoly(mask, [points], 1)
                            
                        except Exception as e:
                            logger.debug(f"Polygon processing failed: {e}")
                            continue
                            
            # Handle single flat polygon
            elif isinstance(segmentation, list) and len(segmentation) >= 6:
                try:
                    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
                    points[:, 0] *= scale_x
                    points[:, 1] *= scale_y
                    points[:, 0] = np.clip(points[:, 0], 0, target_w - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, target_h - 1)
                    points = points.astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
                except Exception as e:
                    logger.debug(f"Single polygon failed: {e}")
                    
        except Exception as e:
            logger.debug(f"Mask creation failed: {e}")
        
        return mask
    
    def _create_mask_from_segmentation(self, segmentation, h: int, w: int) -> np.ndarray:
        """Advanced mask creation with proper dimensions and fallback"""
        logger.debug(f"🎨 _create_mask_from_segmentation called with target dimensions h={h}, w={w}")
        logger.debug(f"   Segmentation type: {type(segmentation)}")
        
        # Ensure h and w are integers
        h, w = int(h), int(w)
        
        # Always create mask with exact crop dimensions
        mask = np.zeros((h, w), dtype=np.uint8)
        logger.debug(f"   Created base mask with shape: {mask.shape}")
        
        if not segmentation:
            logger.debug("   No segmentation data, returning empty mask")
            return mask
        
        try:
            # 1. Try RLE format (COCO compressed mask)
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                logger.debug("   Attempting RLE format decoding...")
                try:
                    import pycocotools.mask as mask_util
                    
                    # Decode RLE mask
                    decoded_mask = mask_util.decode(segmentation)
                    mask_h, mask_w = decoded_mask.shape
                    logger.debug(f"   RLE decoded mask shape: ({mask_h}, {mask_w}), expected: ({h}, {w})")
                    
                    # If mask dimensions don't match crop, we need to handle it
                    if mask_h != h or mask_w != w:
                        logger.debug(f"   RLE mask dimension mismatch! Creating correctly sized mask...")
                        # Create a properly sized mask and copy the decoded mask into it
                        # This handles cases where the mask is smaller than the crop
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        
                        # Calculate how to position the mask within the crop
                        y_start = 0
                        x_start = 0
                        y_end = min(mask_h, h)
                        x_end = min(mask_w, w)
                        
                        logger.debug(f"   Copying region [{y_start}:{y_end}, {x_start}:{x_end}]")
                        # Copy the mask data
                        temp_mask[y_start:y_end, x_start:x_end] = decoded_mask[:y_end, :x_end]
                        logger.debug(f"   Final RLE mask shape: {temp_mask.shape}")
                        return temp_mask
                    else:
                        logger.debug(f"   RLE mask dimensions match, returning as-is")
                        return decoded_mask.astype(np.uint8)
                        
                except Exception as e:
                    logger.debug(f"   RLE decoding failed: {e}, trying polygon fallback")
                    # Fall through to polygon handling
            
            # 2. Try multiple polygons (can handle holes)
            if isinstance(segmentation, list) and len(segmentation) > 0:
                logger.debug(f"   Processing list of {len(segmentation)} items...")
                
                # Check if it's a list of RLE dicts
                if all(isinstance(seg, dict) and 'counts' in seg for seg in segmentation):
                    logger.debug("   Detected list of RLE masks, combining...")
                    # Multiple RLE masks - combine them
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for i, rle in enumerate(segmentation):
                        try:
                            import pycocotools.mask as mask_util
                            decoded = mask_util.decode(rle)
                            logger.debug(f"   RLE {i} decoded shape: {decoded.shape}")
                            
                            # Handle dimension mismatch
                            if decoded.shape[0] <= h and decoded.shape[1] <= w:
                                combined_mask[:decoded.shape[0], :decoded.shape[1]] |= decoded
                            else:
                                # Crop if larger
                                combined_mask |= decoded[:h, :w]
                        except Exception as e:
                            logger.debug(f"   Failed to decode RLE {i}: {e}")
                            continue
                    logger.debug(f"   Combined RLE mask shape: {combined_mask.shape}")
                    return combined_mask
                
                # Standard polygon format
                else:
                    logger.debug("   Processing polygon format...")
                    # Process all polygons (for complex shapes with multiple parts)
                    for poly_idx, polygon in enumerate(segmentation):
                        if isinstance(polygon, list) and len(polygon) >= 6:
                            logger.debug(f"   Processing polygon {poly_idx} with {len(polygon)} coordinates")
                            try:
                                # Convert to points array
                                pts_array = np.array(polygon, dtype=np.float32)
                                
                                # Validate array can be reshaped
                                if pts_array.size % 2 != 0:
                                    logger.debug(f"   Polygon {poly_idx} has odd number of coordinates ({pts_array.size}), skipping")
                                    continue
                                
                                pts = pts_array.reshape(-1, 2)
                                logger.debug(f"   Polygon {poly_idx} has {len(pts)} points")
                                
                                # Log bounds before clipping
                                x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                                y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                                logger.debug(f"   Original bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
                                
                                # Ensure points are within bounds
                                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                                pts = pts.astype(np.int32)
                                
                                # Fill polygon on the mask
                                cv2.fillPoly(mask, [pts], 1)
                                logger.debug(f"   Filled polygon {poly_idx} on mask")
                                
                            except Exception as e:
                                logger.debug(f"   Failed to process polygon {poly_idx}: {e}")
                                continue
                    
                    # Check mask after all polygons
                    mask_sum = mask.sum()
                    logger.debug(f"   Mask after polygons: shape={mask.shape}, non-zero pixels={mask_sum}")
                    
                    # Check if we successfully created any mask
                    if mask.any():
                        logger.debug(f"   ✅ Successfully created polygon mask with shape {mask.shape}")
                        return mask
                    else:
                        logger.debug("   ⚠️ No valid polygons resulted in mask pixels")
            
            # 3. Try bitmap/external mask reference
            elif isinstance(segmentation, dict) and 'mask_file' in segmentation:
                logger.debug("   Attempting to load external mask file...")
                try:
                    mask_path = segmentation['mask_file']
                    if os.path.exists(mask_path):
                        external_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if external_mask is not None:
                            logger.debug(f"   Loaded external mask with shape: {external_mask.shape}")
                            # Resize to match crop dimensions
                            if external_mask.shape != (h, w):
                                logger.debug(f"   Resizing external mask from {external_mask.shape} to ({h}, {w})")
                                external_mask = cv2.resize(external_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            return (external_mask > 0).astype(np.uint8)
                except Exception as e:
                    logger.debug(f"   Failed to load external mask: {e}")
            
            # 4. Final fallback - try simple polygon if segmentation is directly a list of coordinates
            elif isinstance(segmentation, list) and len(segmentation) >= 6 and all(isinstance(x, (int, float)) for x in segmentation):
                logger.debug(f"   Attempting simple polygon format with {len(segmentation)} coordinates...")
                try:
                    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                    logger.debug(f"   Simple polygon mask created with shape: {mask.shape}")
                    return mask
                except Exception as e:
                    logger.debug(f"   Simple polygon fallback failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to create mask with advanced formats: {e}, returning empty mask")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
        
        # Return empty mask if all methods fail
        logger.debug(f"   ❌ All mask creation methods failed, returning empty mask with shape {mask.shape}")
        return mask


class SegmentationHead(nn.Module):
    """Multi-scale segmentation head for ViT with special handling for small classes"""
    
    def __init__(self, in_dim: int, num_classes: int, img_size: int = 224, class_names: List[str] = None):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.class_names = class_names or []
        
        # Determine which classes need multi-scale
        self.small_class_indices = [i for i, name in enumerate(self.class_names) if name in SMALL_CLASSES]
        
        # Main segmentation path
        self.conv1 = nn.Conv2d(in_dim, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, 1)
        
        # Additional fine-grained path for small classes
        self.fine_conv1 = nn.Conv2d(in_dim, 256, 1)
        self.fine_bn1 = nn.BatchNorm2d(256)
        self.fine_conv2 = nn.Conv2d(256, 256, 3, padding=1, dilation=2)  # Dilated convolution
        self.fine_bn2 = nn.BatchNorm2d(256)
        self.fine_conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.fine_bn3 = nn.BatchNorm2d(128)
        self.fine_conv4 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    def forward(self, x, h, w):
        # Reshape from sequence to spatial
        x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        
        # Main segmentation path
        main = F.relu(self.bn1(self.conv1(x)))
        main = F.interpolate(main, scale_factor=2, mode='bilinear', align_corners=False)
        main = F.relu(self.bn2(self.conv2(main)))
        main = F.interpolate(main, scale_factor=2, mode='bilinear', align_corners=False)
        main = self.conv3(main)
        main = self.upsample(main)
        
        # Fine-grained path for small classes
        fine = F.relu(self.fine_bn1(self.fine_conv1(x)))
        fine = F.interpolate(fine, scale_factor=2, mode='bilinear', align_corners=False)
        fine = F.relu(self.fine_bn2(self.fine_conv2(fine)))
        fine = F.relu(self.fine_bn3(self.fine_conv3(fine)))
        fine = F.interpolate(fine, scale_factor=2, mode='bilinear', align_corners=False)
        fine = self.fine_conv4(fine)
        fine = self.upsample(fine)
        
        # Combine outputs: use fine-grained for small classes
        output = main.clone()
        if self.small_class_indices:
            output[:, self.small_class_indices] = fine[:, self.small_class_indices]
        
        return output, {'main': main, 'fine': fine}


class ViTMultiTask(nn.Module):
    """ViT model with multiple task heads for dental analysis"""
    
    def __init__(self, base_model_name: str, num_classes: int, img_size: int = 224):
        super().__init__()
        
        # Load pretrained ViT backbone
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        self.hidden_dim = self.backbone.num_features
        
        # Get patch embedding info
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_patches_per_side = img_size // self.patch_size
        
        # Task-specific heads
        self.classification_head = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_head = nn.Linear(self.hidden_dim, num_classes * 4)  # 4 coords per class
        self.keypoint_head = nn.Linear(self.hidden_dim, num_classes * 3)  # x, y, visibility
        
        # Get class names for segmentation head
        class_names = get_class_names('all')[:num_classes]  # Use first num_classes from all
        self.segmentation_head = SegmentationHead(
            self.hidden_dim, num_classes, img_size, class_names
        )
        
        logger.info(f"✅ Initialized ViT Multi-Task model")
        logger.info(f"   Backbone: {base_model_name}")
        logger.info(f"   Hidden dim: {self.hidden_dim}")
        logger.info(f"   Num classes: {num_classes}")
        logger.info(f"   Patch size: {self.patch_size}")
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.forward_features(x)
        
        # Pool features for classification/regression tasks
        if hasattr(self.backbone, 'global_pool'):
            # Check if it's a callable method or a string attribute
            if callable(self.backbone.global_pool):
                pooled_features = self.backbone.global_pool(features)
            elif isinstance(self.backbone.global_pool, str):
                # Handle string-based pooling (e.g., 'avg', 'token')
                if self.backbone.global_pool == 'avg':
                    # Average pooling over all tokens except CLS
                    pooled_features = features[:, 1:].mean(dim=1)
                elif self.backbone.global_pool == 'token':
                    # Use CLS token
                    pooled_features = features[:, 0]
                else:
                    # Default to CLS token
                    pooled_features = features[:, 0]
            else:
                # Default to CLS token
                pooled_features = features[:, 0]
        else:
            # Default to CLS token for ViT
            pooled_features = features[:, 0]
        
        # Get segmentation outputs (main output and auxiliary outputs)
        seg_output, seg_aux = self.segmentation_head(
            features[:, 1:],  # Skip CLS token
            self.num_patches_per_side,
            self.num_patches_per_side
        )
        
        # Task outputs
        outputs = {
            'classification': torch.sigmoid(self.classification_head(pooled_features)),
            'boxes': self.bbox_head(pooled_features).view(-1, self.classification_head.out_features, 4),
            'keypoints': self.keypoint_head(pooled_features).view(-1, self.classification_head.out_features, 3),
            'segmentation': seg_output,
            'segmentation_aux': seg_aux  # Auxiliary outputs for multi-scale loss
        }
        
        return outputs


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning with multi-scale support for small classes"""
    
    def __init__(self, num_classes: int, class_names: List[str],
                 classification_weight: float = 1.0,
                 bbox_weight: float = 1.0,
                 segmentation_weight: float = 1.0,
                 keypoint_weight: float = 1.0,
                 multiscale_weight: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Loss weights
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight
        self.segmentation_weight = segmentation_weight
        self.keypoint_weight = keypoint_weight
        self.multiscale_weight = multiscale_weight
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Determine which classes use which losses
        self.seg_classes = [i for i, name in enumerate(class_names) if name in SEGMENTATION_CLASSES]
        self.bbox_classes = [i for i, name in enumerate(class_names) if name in BBOX_CLASSES]
        self.point_classes = [i for i, name in enumerate(class_names) if name in POINT_CLASSES]
        self.small_classes = [i for i, name in enumerate(class_names) if name in SMALL_CLASSES]
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = outputs['classification'].shape[0]
        device = outputs['classification'].device
        
        losses = {
            'classification': torch.tensor(0.0, device=device),
            'bbox': torch.tensor(0.0, device=device),
            'segmentation': torch.tensor(0.0, device=device),
            'keypoint': torch.tensor(0.0, device=device),
            'multiscale': torch.tensor(0.0, device=device)
        }
        
        # Process each sample in batch
        for i in range(batch_size):
            target = targets[i]
            
            # Classification loss
            losses['classification'] += self.bce_loss(
                outputs['classification'][i],
                target['labels'].to(device)
            )
            
            # Bounding box loss (only for classes that use bboxes)
            if target['boxes'].shape[0] > 0:
                for class_idx in self.bbox_classes:
                    if target['labels'][class_idx] > 0:
                        # Find the corresponding box
                        pred_box = outputs['boxes'][i, class_idx]
                        # Simple assignment - in practice, you'd use Hungarian matching
                        if target['boxes'].shape[0] > 0:
                            gt_box = target['boxes'][0].to(device)  # Use first box
                            losses['bbox'] += self.smooth_l1_loss(pred_box, gt_box)
            
            # Segmentation loss
            for class_idx in self.seg_classes:
                if target['labels'][class_idx] > 0:
                    class_name = self.class_names[class_idx]
                    logger.debug(f"📊 Computing segmentation loss for {class_name} (idx={class_idx})")
                    
                    pred_mask = outputs['segmentation'][i, class_idx]
                    gt_mask = target['masks'][class_idx].to(device)
                    
                    logger.debug(f"   Pred mask shape: {pred_mask.shape}")
                    logger.debug(f"   GT mask shape before resize: {gt_mask.shape}")
                    
                    # Resize gt_mask to match pred_mask size
                    if gt_mask.shape != pred_mask.shape:
                        logger.debug(f"   ⚠️ Shape mismatch! Need to resize GT mask")
                        logger.debug(f"   Target size: {pred_mask.shape}")
                        
                        # Add dimensions for interpolation
                        gt_mask_expanded = gt_mask.unsqueeze(0).unsqueeze(0)
                        logger.debug(f"   GT mask expanded shape: {gt_mask_expanded.shape}")
                        
                        try:
                            # Check if dimensions are compatible
                            if len(gt_mask.shape) != 2:
                                logger.error(f"   ❌ GT mask has wrong number of dimensions: {len(gt_mask.shape)}, expected 2")
                                logger.error(f"   GT mask shape details: {gt_mask.shape}")
                                continue
                            
                            # Perform interpolation
                            gt_mask_resized = F.interpolate(
                                gt_mask_expanded,
                                size=pred_mask.shape,
                                mode='bilinear',
                                align_corners=False
                            )
                            logger.debug(f"   Resized shape before squeeze: {gt_mask_resized.shape}")
                            
                            gt_mask = gt_mask_resized.squeeze()
                            logger.debug(f"   Final GT mask shape after resize: {gt_mask.shape}")
                            
                        except RuntimeError as e:
                            logger.error(f"   ❌ Failed to resize mask for {class_name}: {e}")
                            logger.error(f"   This is the dimension mismatch error!")
                            logger.error(f"   Pred shape: {pred_mask.shape}, GT shape: {gt_mask.shape}")
                            logger.error(f"   Skipping this class for loss calculation")
                            continue
                    
                    # Binary cross entropy for segmentation
                    try:
                        loss_val = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
                        losses['segmentation'] += loss_val
                        logger.debug(f"   ✅ Successfully computed loss: {loss_val.item():.4f}")
                    except Exception as e:
                        logger.error(f"   ❌ Failed to compute BCE loss for {class_name}: {e}")
                        continue
            
            # Keypoint loss
            for class_idx in self.point_classes:
                if target['labels'][class_idx] > 0 and target['keypoints'][class_idx, 2] > 0:
                    pred_kpt = outputs['keypoints'][i, class_idx]
                    gt_kpt = target['keypoints'][class_idx].to(device)
                    losses['keypoint'] += self.mse_loss(pred_kpt[:2], gt_kpt[:2])
            
            # Multi-scale loss for small classes
            if 'segmentation_aux' in outputs:
                aux_outputs = outputs['segmentation_aux']
                if 'main' in aux_outputs and 'fine' in aux_outputs:
                    for class_idx in self.small_classes:
                        if target['labels'][class_idx] > 0:
                            gt_mask = target['masks'][class_idx].to(device)
                            
                            # Resize gt_mask if needed
                            main_mask = aux_outputs['main'][i, class_idx]
                            fine_mask = aux_outputs['fine'][i, class_idx]
                            
                            if gt_mask.shape != main_mask.shape:
                                gt_mask = F.interpolate(
                                    gt_mask.unsqueeze(0).unsqueeze(0),
                                    size=main_mask.shape,
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze()
                            
                            # Multi-scale loss: combine losses from both paths
                            main_loss = F.binary_cross_entropy_with_logits(main_mask, gt_mask)
                            fine_loss = F.binary_cross_entropy_with_logits(fine_mask, gt_mask)
                            
                            # Edge-aware loss for small structures
                            edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
                            
                            # Apply edge detection to ground truth
                            gt_edges = F.conv2d(gt_mask.unsqueeze(0).unsqueeze(0), edge_kernel, padding=1)
                            gt_edges = torch.abs(gt_edges).squeeze()
                            
                            # Apply edge detection to predictions
                            fine_edges = F.conv2d(torch.sigmoid(fine_mask).unsqueeze(0).unsqueeze(0), 
                                                edge_kernel, padding=1)
                            fine_edges = torch.abs(fine_edges).squeeze()
                            
                            # Edge loss
                            edge_loss = F.mse_loss(fine_edges, gt_edges)
                            
                            # Combined multi-scale loss
                            losses['multiscale'] += 0.3 * main_loss + 0.5 * fine_loss + 0.2 * edge_loss
        
        # Average losses
        for key in losses:
            if losses[key] > 0:
                losses[key] = losses[key] / batch_size
        
        # Weighted total loss
        total_loss = (
            self.classification_weight * losses['classification'] +
            self.bbox_weight * losses['bbox'] +
            self.segmentation_weight * losses['segmentation'] +
            self.keypoint_weight * losses['keypoint'] +
            self.multiscale_weight * losses['multiscale']
        )
        
        losses['total'] = total_loss
        return losses


class TwoStageTrainer:
    """Trainer for 2-stage dental landmark detection"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get class names
        self.class_names = get_class_names(args.class_group)
        self.num_classes = len(self.class_names)
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Class group: {args.class_group}")
        logger.info(f"   Num classes: {self.num_classes}")
        logger.info(f"   Classes: {self.class_names[:10]}{'...' if len(self.class_names) > 10 else ''}")
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # Initialize loss
        self.criterion = MultiTaskLoss(
            self.num_classes, self.class_names,
            classification_weight=1.0,
            bbox_weight=0.5,
            segmentation_weight=2.0,
            keypoint_weight=1.0
        )
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epoch = 0
        
        # WandB
        if args.wandb and wandb:
            wandb.init(
                project=args.project,
                name=args.name,
                config=vars(args)
            )
    
    def _init_model(self):
        """Initialize ViT model"""
        model_map = {
            'tiny': 'vit_tiny_patch16_224',
            'small': 'vit_small_patch16_224',
            'base': 'vit_base_patch16_224',
            'large': 'vit_large_patch16_224'
        }
        
        model_name = model_map.get(self.args.model_size, 'vit_base_patch16_224')
        model = ViTMultiTask(model_name, self.num_classes, self.args.image_size)
        
        # Load pretrained weights if provided
        if self.args.pretrained:
            logger.info(f"Loading pretrained weights from {self.args.pretrained}")
            checkpoint = torch.load(self.args.pretrained, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model.to(self.device)
    
    def _setup_data(self):
        """Setup data loaders"""
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Determine paths
        if self.args.train_data.startswith('s3://'):
            # SageMaker paths
            train_ann_path = '/opt/ml/input/data/train/' + self.args.train_annotations
            val_ann_path = '/opt/ml/input/data/validation/' + self.args.val_annotations
            train_img_dir = '/opt/ml/input/data/train/images'
            val_img_dir = '/opt/ml/input/data/validation/images'
        else:
            # Local paths
            train_ann_path = os.path.join(self.args.train_data, self.args.train_annotations)
            val_ann_path = os.path.join(self.args.validation_data, self.args.val_annotations)
            train_img_dir = os.path.join(self.args.train_data, 'images')
            val_img_dir = os.path.join(self.args.validation_data, 'images')
        
        # Stage 1 model path
        if self.args.stage1_model_s3 and self.args.stage1_model_s3.startswith('s3://'):
            # Download from S3
            stage1_model_path = self._download_stage1_model()
        else:
            # Local path
            stage1_model_path = self.args.stage1_model_s3 or 'src/best.pt'
        
        # Create datasets
        train_dataset = TwoStageDataset(
            train_ann_path, train_img_dir, stage1_model_path,
            self.class_names, train_transform, 'train',
            crop_expand_factor=self.args.crop_expand_factor,
            conf_threshold=self.args.stage1_conf_threshold,
            xray_type=self.args.xray_type,
            subset_size=self.args.subset_size,
            cache_crops=True
        )
        
        val_dataset = TwoStageDataset(
            val_ann_path, val_img_dir, stage1_model_path,
            self.class_names, val_transform, 'val',
            crop_expand_factor=self.args.crop_expand_factor,
            conf_threshold=self.args.stage1_conf_threshold,
            xray_type=self.args.xray_type,
            subset_size=self.args.subset_size if self.args.subset_size else None,
            cache_crops=True
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"📊 Dataset statistics:")
        logger.info(f"   Train crops: {len(train_dataset)}")
        logger.info(f"   Val crops: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _download_stage1_model(self):
        """Download Stage 1 model from S3"""
        logger.info(f"📥 Downloading Stage 1 model from {self.args.stage1_model_s3}")
        
        # Parse S3 path
        s3_parts = self.args.stage1_model_s3.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        # Determine file extension
        if key.endswith('.tar.gz'):
            local_tar_path = '/tmp/stage1_model.tar.gz'
            local_model_path = '/tmp/stage1_model.pt'
            
            # Download tar.gz file
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket, key, local_tar_path)
            logger.info(f"📦 Downloaded compressed model to {local_tar_path}")
            
            # Extract the model
            import tarfile
            logger.info("📂 Extracting model from tar.gz...")
            with tarfile.open(local_tar_path, 'r:gz') as tar:
                # List contents to find the .pt file
                for member in tar.getmembers():
                    if member.name.endswith('.pt'):
                        logger.info(f"📄 Found model file: {member.name}")
                        # Extract just this file
                        tar.extract(member, '/tmp/')
                        extracted_path = os.path.join('/tmp', member.name)
                        # Move to expected location
                        shutil.move(extracted_path, local_model_path)
                        break
                else:
                    # If no .pt file found, extract all and look for best.pt or model.pt
                    tar.extractall('/tmp/extracted/')
                    # Look for common model file names
                    for model_name in ['best.pt', 'model.pt', 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
                        possible_path = f'/tmp/extracted/{model_name}'
                        if os.path.exists(possible_path):
                            shutil.move(possible_path, local_model_path)
                            logger.info(f"📄 Found and moved model: {model_name}")
                            break
                    else:
                        # Last resort: find any .pt file
                        import glob
                        pt_files = glob.glob('/tmp/extracted/**/*.pt', recursive=True)
                        if pt_files:
                            shutil.move(pt_files[0], local_model_path)
                            logger.info(f"📄 Found and moved model: {os.path.basename(pt_files[0])}")
                        else:
                            raise ValueError("No .pt model file found in the tar.gz archive")
            
            # Clean up
            os.remove(local_tar_path)
            if os.path.exists('/tmp/extracted'):
                shutil.rmtree('/tmp/extracted')
                
        else:
            # Direct .pt file download
            local_model_path = '/tmp/stage1_model.pt'
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket, key, local_model_path)
            logger.info(f"📄 Downloaded model directly to {local_model_path}")
        
        logger.info(f"✅ Stage 1 model ready at {local_model_path}")
        return local_model_path
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        images = []
        targets = []
        infos = []
        
        for img, target, info in batch:
            images.append(img)
            targets.append(target)
            infos.append(info)
        
        images = torch.stack(images)
        return images, targets, infos
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_losses = defaultdict(float)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.args.epochs}")
        
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            for k, v in losses.items():
                running_losses[k] += v.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_losses['total'] / (batch_idx + 1),
                'cls': running_losses['classification'] / (batch_idx + 1),
                'seg': running_losses['segmentation'] / (batch_idx + 1)
            })
        
        # Average losses
        avg_losses = {k: v / len(self.train_loader) for k, v in running_losses.items()}
        
        return avg_losses
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_losses = defaultdict(float)
        
        # Per-class metrics
        class_metrics = {
            'tp': torch.zeros(self.num_classes),
            'fp': torch.zeros(self.num_classes),
            'fn': torch.zeros(self.num_classes),
            'iou': torch.zeros(self.num_classes),
            'count': torch.zeros(self.num_classes)
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for batch_idx, (images, targets, _) in enumerate(pbar):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate losses
                losses = self.criterion(outputs, targets)
                
                # Track losses
                for k, v in losses.items():
                    running_losses[k] += v.item()
                
                # Calculate metrics
                self._update_metrics(outputs, targets, class_metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_losses['total'] / (batch_idx + 1)
                })
        
        # Average losses
        avg_losses = {k: v / len(self.val_loader) for k, v in running_losses.items()}
        
        # Calculate per-class metrics
        metrics = self._calculate_metrics(class_metrics)
        
        return avg_losses, metrics
    
    def _update_metrics(self, outputs, targets, class_metrics):
        """Update running metrics"""
        batch_size = outputs['classification'].shape[0]
        device = outputs['classification'].device
        
        for i in range(batch_size):
            # Classification metrics
            pred_labels = outputs['classification'][i] > 0.5
            gt_labels = targets[i]['labels'].to(device) > 0.5
            
            tp = (pred_labels & gt_labels).float().cpu()
            fp = (pred_labels & ~gt_labels).float().cpu()
            fn = (~pred_labels & gt_labels).float().cpu()
            
            class_metrics['tp'] += tp
            class_metrics['fp'] += fp
            class_metrics['fn'] += fn
            
            # Segmentation IoU (for classes that use segmentation)
            for class_idx in range(self.num_classes):
                if self.class_names[class_idx] in SEGMENTATION_CLASSES and gt_labels[class_idx]:
                    pred_mask = torch.sigmoid(outputs['segmentation'][i, class_idx]) > 0.5
                    gt_mask = targets[i]['masks'][class_idx].to(device) > 0.5
                    
                    # Resize masks to same size
                    if gt_mask.shape != pred_mask.shape:
                        gt_mask = F.interpolate(
                            gt_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=pred_mask.shape,
                            mode='nearest'
                        ).squeeze() > 0.5
                    
                    # Calculate IoU
                    intersection = (pred_mask & gt_mask).float().sum()
                    union = (pred_mask | gt_mask).float().sum()
                    
                    if union > 0:
                        iou = intersection / union
                        class_metrics['iou'][class_idx] += iou.cpu()
                        class_metrics['count'][class_idx] += 1
    
    def _calculate_metrics(self, class_metrics):
        """Calculate final metrics"""
        metrics = {}
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            tp = class_metrics['tp'][i]
            fp = class_metrics['fp'][i]
            fn = class_metrics['fn'][i]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Handle both tensor and scalar cases
            metrics[f"{class_name}_precision"] = precision.item() if hasattr(precision, 'item') else precision
            metrics[f"{class_name}_recall"] = recall.item() if hasattr(recall, 'item') else recall
            metrics[f"{class_name}_f1"] = f1.item() if hasattr(f1, 'item') else f1
            
            # IoU for segmentation classes
            if class_name in SEGMENTATION_CLASSES and class_metrics['count'][i] > 0:
                avg_iou = class_metrics['iou'][i] / class_metrics['count'][i]
                metrics[f"{class_name}_iou"] = avg_iou.item()
        
        # Overall metrics
        overall_tp = class_metrics['tp'].sum()
        overall_fp = class_metrics['fp'].sum()
        overall_fn = class_metrics['fn'].sum()
        
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall_precision'] = overall_precision.item() if hasattr(overall_precision, 'item') else overall_precision
        metrics['overall_recall'] = overall_recall.item() if hasattr(overall_recall, 'item') else overall_recall
        metrics['overall_f1'] = overall_f1.item() if hasattr(overall_f1, 'item') else overall_f1
        
        # Average IoU
        seg_classes_mask = torch.tensor([name in SEGMENTATION_CLASSES for name in self.class_names])
        seg_count = class_metrics['count'][seg_classes_mask]
        if seg_count.sum() > 0:
            seg_iou = class_metrics['iou'][seg_classes_mask]
            avg_iou = (seg_iou / (seg_count + 1e-8)).mean()
            metrics['mean_iou'] = avg_iou.item()
        
        return metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args,
            'class_names': self.class_names
        }
        
        # Save latest
        os.makedirs(self.args.output_dir, exist_ok=True)
        latest_path = os.path.join(self.args.output_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best.pt')
            shutil.copy2(latest_path, best_path)
            logger.info(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Save to SageMaker model dir
        if self.args.model_dir != self.args.output_dir:
            os.makedirs(self.args.model_dir, exist_ok=True)
            if is_best:
                shutil.copy2(best_path, os.path.join(self.args.model_dir, 'best.pt'))
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting 2-stage training")
        
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Log results
            logger.info(f"\n📊 Epoch {epoch + 1}/{self.args.epochs} Results:")
            logger.info(f"   Train Loss: {train_losses['total']:.4f}")
            logger.info(f"   Val Loss: {val_losses['total']:.4f}")
            logger.info(f"   Overall F1: {val_metrics['overall_f1']:.4f}")
            if 'mean_iou' in val_metrics:
                logger.info(f"   Mean IoU: {val_metrics['mean_iou']:.4f}")
            
            # Log to WandB
            if self.args.wandb and wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'lr': self.scheduler.get_last_lr()[0]
                }
                
                # Add losses
                for k, v in train_losses.items():
                    log_dict[f'train/{k}'] = v
                for k, v in val_losses.items():
                    log_dict[f'val/{k}'] = v
                
                # Add metrics
                for k, v in val_metrics.items():
                    log_dict[f'val/{k}'] = v
                
                wandb.log(log_dict)
            
            # Log per-class metrics periodically
            if (epoch + 1) % 10 == 0:
                logger.info("\n📈 Per-Class Metrics:")
                for class_name in self.class_names[:10]:  # Show first 10 classes
                    f1 = val_metrics.get(f"{class_name}_f1", 0)
                    logger.info(f"   {class_name}: F1={f1:.4f}")
        
        logger.info("\n✅ Training completed!")
        logger.info(f"🏆 Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train 2-Stage Model: YOLO + ViT')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained Stage 2 weights')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--mask_ratio', type=float, default=4,
                        help='Mask ratio for training (not used in this version)')
    
    # Dataset parameters
    parser.add_argument('--train_data', type=str, default='s3://codentist-general/datasets/master')
    parser.add_argument('--validation_data', type=str, default='s3://codentist-general/datasets/master')
    parser.add_argument('--train_annotations', type=str, default='train.json')
    parser.add_argument('--val_annotations', type=str, default='val.json')
    parser.add_argument('--class_group', type=str, default='bone-loss',
                        help='Class group to train on')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-inferred from class_group if not specified)')
    parser.add_argument('--xray_type', type=str, default=None,
                        help='Filter by X-ray type (e.g., bitewing,periapical)')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Limit dataset size for testing')
    
    # Stage 1 parameters
    parser.add_argument('--stage1_model_s3', type=str, default=None,
                        help='S3 path or local path to Stage 1 YOLO model')
    parser.add_argument('--stage1_model_type', type=str, default='teeth',
                        help='Type of Stage 1 model (teeth detection)')
    parser.add_argument('--stage1_conf_threshold', type=float, default=0.3)
    parser.add_argument('--crop_expand_factor', type=float, default=0.2)
    
    # Stage 2 parameters
    parser.add_argument('--stage2_target', type=str, default='landmarks',
                        help='Stage 2 target task (landmarks, decay, etc.)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--project', type=str, default='dental_2stage')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    
    # Monitoring parameters (from ViT approach)
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--save_best_only', type=str, default='true',
                        help='Only save the model with best validation performance')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')
    
    args = parser.parse_args()
    
    # Auto-infer num_classes from class_group if not specified
    if args.num_classes is None:
        class_names = get_class_names(args.class_group)
        args.num_classes = len(class_names)
        logger.info(f"🎯 Auto-inferred num_classes={args.num_classes} from class_group='{args.class_group}'")
    
    # Set defaults
    if args.model_dir is None:
        args.model_dir = os.environ.get('SM_MODEL_DIR', args.output_dir)
    
    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'2stage_{args.class_group}_{timestamp}'
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Log configuration
    logger.info("🎯 === 2-Stage Training Configuration ===")
    logger.info(f"📋 Experiment: {args.name}")
    logger.info(f"🎯 Class group: {args.class_group}")
    logger.info(f"🦷 Stage 1 model: {args.stage1_model_s3 or 'src/best.pt'}")
    logger.info(f"🖼️ Image size: {args.image_size}")
    logger.info(f"🧠 Model size: {args.model_size}")
    logger.info(f"📊 Batch size: {args.batch_size}")
    logger.info(f"🔄 Epochs: {args.epochs}")
    logger.info(f"📈 Learning rate: {args.learning_rate}")
    logger.info(f"🎲 Random seed: {args.seed}")
    
    # Train
    trainer = TwoStageTrainer(args)
    trainer.train()
    
    logger.info("🎉 Training pipeline completed!")


if __name__ == '__main__':
    main()
