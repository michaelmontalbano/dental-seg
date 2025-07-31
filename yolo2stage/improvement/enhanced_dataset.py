"""
Enhanced TwoStageDataset with radiographic preprocessing
Integrates dental X-ray enhancement into the training pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.train import TwoStageDataset
from .radiographic_preprocessing import DentalRadiographicEnhancer, create_dental_preprocessor
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedTwoStageDataset(TwoStageDataset):
    """
    Enhanced version of TwoStageDataset with radiographic preprocessing.
    """
    
    def __init__(self, 
                 annotations_file: str,
                 images_dir: str,
                 stage1_model_path: str,
                 class_names: List[str],
                 transform=None,
                 split='train',
                 crop_expand_factor: float = 0.2,
                 conf_threshold: float = 0.3,
                 xray_type: Optional[str] = None,
                 subset_size: Optional[int] = None,
                 cache_crops: bool = True,
                 # New parameters for enhancement
                 apply_radiographic_enhancement: bool = True,
                 enhancement_mode: str = 'adaptive',
                 enhance_full_images: bool = True,
                 enhance_crops: bool = True):
        """
        Initialize enhanced dataset with radiographic preprocessing.
        
        Args:
            Same as TwoStageDataset, plus:
            apply_radiographic_enhancement: Whether to apply enhancement
            enhancement_mode: 'standard', 'adaptive', 'light', or 'aggressive'
            enhance_full_images: Whether to enhance full images before cropping
            enhance_crops: Whether to enhance individual tooth crops
        """
        # Store enhancement settings
        self.apply_radiographic_enhancement = apply_radiographic_enhancement
        self.enhancement_mode = enhancement_mode
        self.enhance_full_images = enhance_full_images
        self.enhance_crops = enhance_crops
        
        # Create enhancer
        if self.apply_radiographic_enhancement:
            self.enhancer = create_dental_preprocessor(enhancement_mode)
            logger.info(f"✨ Initialized radiographic enhancer in '{enhancement_mode}' mode")
        else:
            self.enhancer = None
        
        # Initialize parent class
        super().__init__(
            annotations_file=annotations_file,
            images_dir=images_dir,
            stage1_model_path=stage1_model_path,
            class_names=class_names,
            transform=transform,
            split=split,
            crop_expand_factor=crop_expand_factor,
            conf_threshold=conf_threshold,
            xray_type=xray_type,
            subset_size=subset_size,
            cache_crops=cache_crops
        )
    
    def _extract_crops_with_annotations(self, img_path, img_info, detections, annotations):
        """
        Override to add full image enhancement before cropping.
        """
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return []
        
        # Apply radiographic enhancement to full image if enabled
        if self.apply_radiographic_enhancement and self.enhance_full_images:
            # Convert to grayscale for enhancement
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance
            enhanced_gray = self.enhancer.enhance_radiograph(gray)
            
            # Convert back to BGR for consistency
            image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            logger.debug(f"🔬 Enhanced full radiograph: {img_path.name}")
        
        # Continue with parent method
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
            
            # Apply enhancement to individual crop if enabled
            if self.apply_radiographic_enhancement and self.enhance_crops and not self.enhance_full_images:
                # Only enhance crops if we didn't enhance the full image
                if len(crop_img.shape) == 3:
                    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_crop = crop_img
                
                enhanced_crop = self.enhancer.enhance_radiograph(gray_crop)
                crop_img = cv2.cvtColor(enhanced_crop, cv2.COLOR_GRAY2BGR)
            
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
    
    def __getitem__(self, idx):
        """
        Override to use custom preprocessing for enhanced images.
        """
        crop_info = self.crop_data[idx]
        
        # Load crop image
        if self.cache_crops and 'crop_image' in crop_info:
            crop_img = crop_info['crop_image']
        else:
            # Load and extract crop
            image = cv2.imread(crop_info['image_path'])
            x1, y1, x2, y2 = crop_info['crop_bbox']
            crop_img = image[y1:y2, x1:x2]
            
            # Apply enhancement if not cached
            if self.apply_radiographic_enhancement:
                if len(crop_img.shape) == 3:
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = crop_img
                
                enhanced = self.enhancer.enhance_radiograph(gray)
                crop_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Convert to PIL
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_img)
        
        # Apply transforms
        if self.transform:
            crop_tensor = self.transform(crop_pil)
        else:
            # Use custom radiographic normalization
            if self.apply_radiographic_enhancement:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],  # Centered for radiographs
                        std=[0.25, 0.25, 0.25]  # Less aggressive than ImageNet
                    )
                ])
            else:
                transform = transforms.ToTensor()
            crop_tensor = transform(crop_pil)
        
        # Prepare targets
        targets = self._prepare_targets(crop_info)
        
        return crop_tensor, targets, crop_info


def create_enhanced_data_loaders(args, stage1_model_path, class_names):
    """
    Create data loaders with radiographic enhancement.
    
    Args:
        args: Training arguments
        stage1_model_path: Path to Stage 1 model
        class_names: List of class names
        
    Returns:
        train_loader, val_loader
    """
    # Determine enhancement mode based on dataset characteristics
    enhancement_mode = getattr(args, 'enhancement_mode', 'adaptive')
    apply_enhancement = getattr(args, 'apply_radiographic_enhancement', True)
    
    # Training transforms with radiographic-aware augmentation
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Reduced color jitter for radiographs
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # Custom normalization for radiographs
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    # Determine paths
    if args.train_data.startswith('s3://'):
        train_ann_path = '/opt/ml/input/data/train/' + args.train_annotations
        val_ann_path = '/opt/ml/input/data/validation/' + args.val_annotations
        train_img_dir = '/opt/ml/input/data/train/images'
        val_img_dir = '/opt/ml/input/data/validation/images'
    else:
        train_ann_path = os.path.join(args.train_data, args.train_annotations)
        val_ann_path = os.path.join(args.validation_data, args.val_annotations)
        train_img_dir = os.path.join(args.train_data, 'images')
        val_img_dir = os.path.join(args.validation_data, 'images')
    
    # Create enhanced datasets
    train_dataset = EnhancedTwoStageDataset(
        train_ann_path, train_img_dir, stage1_model_path,
        class_names, train_transform, 'train',
        crop_expand_factor=args.crop_expand_factor,
        conf_threshold=args.stage1_conf_threshold,
        xray_type=args.xray_type,
        subset_size=args.subset_size,
        cache_crops=True,
        apply_radiographic_enhancement=apply_enhancement,
        enhancement_mode=enhancement_mode,
        enhance_full_images=True,
        enhance_crops=False  # Already enhanced full image
    )
    
    val_dataset = EnhancedTwoStageDataset(
        val_ann_path, val_img_dir, stage1_model_path,
        class_names, val_transform, 'val',
        crop_expand_factor=args.crop_expand_factor,
        conf_threshold=args.stage1_conf_threshold,
        xray_type=args.xray_type,
        subset_size=args.subset_size if args.subset_size else None,
        cache_crops=True,
        apply_radiographic_enhancement=apply_enhancement,
        enhancement_mode='light',  # Lighter enhancement for validation
        enhance_full_images=True,
        enhance_crops=False
    )
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset._collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=val_dataset._collate_fn
    )
    
    logger.info(f"📊 Enhanced dataset statistics:")
    logger.info(f"   Train crops: {len(train_dataset)} (with {enhancement_mode} enhancement)")
    logger.info(f"   Val crops: {len(val_dataset)} (with light enhancement)")
    
    return train_loader, val_loader


# Quality metrics tracking
class RadiographicQualityMonitor:
    """Track radiographic quality metrics during training."""
    
    def __init__(self):
        self.quality_metrics = []
    
    def add_batch_metrics(self, batch_images, enhanced_images):
        """Calculate and store quality improvement metrics."""
        batch_metrics = {
            'original_contrast': [],
            'enhanced_contrast': [],
            'noise_reduction': [],
            'edge_preservation': []
        }
        
        for orig, enh in zip(batch_images, enhanced_images):
            # Calculate contrast improvement
            orig_contrast = orig.std()
            enh_contrast = enh.std()
            batch_metrics['original_contrast'].append(orig_contrast.item())
            batch_metrics['enhanced_contrast'].append(enh_contrast.item())
            
            # Estimate noise reduction (simplified)
            noise_reduction = 1.0 - (enh.var() / (orig.var() + 1e-6))
            batch_metrics['noise_reduction'].append(noise_reduction.item())
            
            # Edge preservation (simplified)
            # In practice, would use actual edge detection
            edge_preservation = 0.9  # Placeholder
            batch_metrics['edge_preservation'].append(edge_preservation)
        
        self.quality_metrics.append(batch_metrics)
    
    def get_summary(self):
        """Get summary statistics of quality improvements."""
        if not self.quality_metrics:
            return {}
        
        # Aggregate metrics
        all_metrics = {
            'avg_contrast_improvement': [],
            'avg_noise_reduction': [],
            'avg_edge_preservation': []
        }
        
        for batch in self.quality_metrics:
            contrast_imp = np.mean([
                e/o for o, e in zip(batch['original_contrast'], batch['enhanced_contrast'])
                if o > 0
            ])
            all_metrics['avg_contrast_improvement'].append(contrast_imp)
            all_metrics['avg_noise_reduction'].append(np.mean(batch['noise_reduction']))
            all_metrics['avg_edge_preservation'].append(np.mean(batch['edge_preservation']))
        
        return {
            'contrast_improvement': np.mean(all_metrics['avg_contrast_improvement']),
            'noise_reduction': np.mean(all_metrics['avg_noise_reduction']),
            'edge_preservation': np.mean(all_metrics['avg_edge_preservation'])
        }
