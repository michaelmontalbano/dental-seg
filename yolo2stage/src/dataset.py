#!/usr/bin/env python3
"""
Dataset utilities for 2-Stage YOLOv8 dental landmark segmentation
Supports tooth crop extraction and YOLO format conversion
"""

import json
import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
from collections import Counter
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class TwoStageDatasetValidator:
    """
    Validator for 2-stage dental dataset preparation
    Ensures data quality for crop-based landmark segmentation
    """
    
    def __init__(self, annotations_file: str, images_dir: str, class_group: str = 'bone-loss'):
        self.annotations_file = annotations_file
        self.images_dir = Path(images_dir)
        self.class_group = class_group
        
        # Define landmark classes based on class_group
        self.landmark_classes = self._get_landmark_classes(class_group)
        
        logger.info(f"🔍 2-Stage Dataset Validator initialized")
        logger.info(f"📁 Annotations: {annotations_file}")
        logger.info(f"📁 Images: {images_dir}")
        logger.info(f"🎯 Target landmarks: {self.landmark_classes}")
        
        # Load and validate annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.validation_results = self._validate_dataset()
    
    def _get_landmark_classes(self, class_group: str) -> List[str]:
        """Get landmark classes based on class group"""
        class_mappings = {
            'bone-loss': ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"],
            'conditions': ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"],
            'surfaces': ["cej mesial", "ac mesial", "cej distal", "ac distal", "occlusal"],
            'cej': ['cej mesial', 'cej distal', 'apex'],
            'ac': ['ac mesial', 'ac distal', 'apex'],
            'apex': ["apex"],
            'mesial': ["cej mesial", "ac mesial"],
            'distal': ["ac distal", "cej distal"]
        }
        return class_mappings.get(class_group, class_mappings['bone-loss'])
    
    def _validate_dataset(self) -> Dict:
        """Validate dataset for 2-stage training compatibility"""
        logger.info("🔍 Validating dataset for 2-stage training...")
        
        results = {
            'total_images': 0,
            'images_with_landmarks': 0,
            'total_landmarks': 0,
            'landmark_counts': Counter(),
            'missing_images': [],
            'images_without_landmarks': [],
            'segmentation_quality': {'with_masks': 0, 'bbox_only': 0, 'invalid': 0},
            'xray_types': Counter(),
            'issues': []
        }
        
        if not isinstance(self.annotations, dict) or 'images' not in self.annotations:
            results['issues'].append("Invalid annotation format - not COCO format")
            return results
        
        # Create categories lookup
        categories = {cat['id']: cat['name'] for cat in self.annotations.get('categories', [])}
        target_landmark_set = set(self.landmark_classes)
        
        # Group annotations by image
        image_annotations = {}
        for ann in self.annotations.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Validate each image
        for img_info in self.annotations['images']:
            results['total_images'] += 1
            img_id = img_info['id']
            file_name = img_info['file_name']
            
            # Check if image file exists
            image_path = self.images_dir / file_name
            if not image_path.exists():
                results['missing_images'].append(file_name)
                continue
            
            # Count X-ray types
            xray_type = img_info.get('xray_type', 'unknown')
            results['xray_types'][xray_type] += 1
            
            # Check for landmark annotations
            landmark_anns = []
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    category_name = categories.get(ann['category_id'], 'unknown')
                    if category_name in target_landmark_set:
                        landmark_anns.append(ann)
                        results['landmark_counts'][category_name] += 1
                        
                        # Check segmentation quality
                        if ann.get('segmentation') and len(ann['segmentation'][0]) >= 6:
                            results['segmentation_quality']['with_masks'] += 1
                        elif ann.get('bbox') and len(ann['bbox']) == 4:
                            results['segmentation_quality']['bbox_only'] += 1
                        else:
                            results['segmentation_quality']['invalid'] += 1
            
            if landmark_anns:
                results['images_with_landmarks'] += 1
                results['total_landmarks'] += len(landmark_anns)
            else:
                results['images_without_landmarks'].append(file_name)
        
        # Generate summary
        logger.info(f"📊 === Dataset Validation Results ===")
        logger.info(f"📁 Total images: {results['total_images']}")
        logger.info(f"✅ Images with landmarks: {results['images_with_landmarks']}")
        logger.info(f"❌ Images without landmarks: {len(results['images_without_landmarks'])}")
        logger.info(f"🏷️ Total landmarks: {results['total_landmarks']}")
        logger.info(f"📷 X-ray types: {dict(results['xray_types'])}")
        
        logger.info(f"🎭 Segmentation data quality:")
        for quality_type, count in results['segmentation_quality'].items():
            logger.info(f"   {quality_type}: {count}")
        
        logger.info(f"📈 Landmark distribution:")
        for landmark, count in results['landmark_counts'].most_common():
            logger.info(f"   {landmark}: {count}")
        
        if results['missing_images']:
            logger.warning(f"⚠️ Missing images: {len(results['missing_images'])}")
            if len(results['missing_images']) <= 5:
                for img in results['missing_images']:
                    logger.warning(f"   Missing: {img}")
        
        return results
    
    def get_validation_summary(self) -> str:
        """Get formatted validation summary"""
        results = self.validation_results
        
        summary = f"""
2-Stage Dataset Validation Summary
================================
Total Images: {results['total_images']}
Images with Landmarks: {results['images_with_landmarks']}
Total Landmarks: {results['total_landmarks']}
Missing Images: {len(results['missing_images'])}

Landmark Distribution:
{chr(10).join([f"  {name}: {count}" for name, count in results['landmark_counts'].most_common()])}

Segmentation Quality:
  With Masks: {results['segmentation_quality']['with_masks']}
  Bbox Only: {results['segmentation_quality']['bbox_only']}
  Invalid: {results['segmentation_quality']['invalid']}

X-ray Types:
{chr(10).join([f"  {xtype}: {count}" for xtype, count in results['xray_types'].most_common()])}
"""
        return summary
    
    def is_dataset_ready(self) -> bool:
        """Check if dataset is ready for 2-stage training"""
        results = self.validation_results
        
        # Minimum requirements
        min_images_with_landmarks = 50
        min_landmarks_per_class = 10
        max_missing_images_ratio = 0.1
        
        # Check requirements
        if results['images_with_landmarks'] < min_images_with_landmarks:
            logger.error(f"❌ Insufficient images with landmarks: {results['images_with_landmarks']} < {min_images_with_landmarks}")
            return False
        
        if len(results['missing_images']) > results['total_images'] * max_missing_images_ratio:
            logger.error(f"❌ Too many missing images: {len(results['missing_images'])}")
            return False
        
        # Check landmark distribution
        for landmark in self.landmark_classes:
            count = results['landmark_counts'].get(landmark, 0)
            if count < min_landmarks_per_class:
                logger.error(f"❌ Insufficient {landmark} landmarks: {count} < {min_landmarks_per_class}")
                return False
        
        logger.info("✅ Dataset ready for 2-stage training!")
        return True

class CropDatasetAnalyzer:
    """
    Analyze extracted tooth crops for quality assessment
    """
    
    def __init__(self, crops_dataset_dir: str):
        self.crops_dir = Path(crops_dataset_dir)
        self.analysis_results = {}
        
        if self.crops_dir.exists():
            self.analysis_results = self._analyze_crops()
    
    def _analyze_crops(self) -> Dict:
        """Analyze crop dataset quality"""
        logger.info(f"📊 Analyzing crop dataset: {self.crops_dir}")
        
        results = {
            'splits': {},
            'total_crops': 0,
            'crop_sizes': [],
            'labels_per_crop': [],
            'class_distribution': Counter()
        }
        
        for split in ['train', 'val']:
            split_dir = self.crops_dir / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            split_results = {
                'images': len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0,
                'labels': len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0,
                'avg_crop_size': None,
                'avg_labels_per_crop': None
            }
            
            # Analyze crop sizes
            if images_dir.exists():
                crop_sizes = []
                for img_file in images_dir.glob('*.jpg'):
                    try:
                        img = Image.open(img_file)
                        crop_sizes.append(img.size[0] * img.size[1])  # width * height
                        results['crop_sizes'].append(img.size)
                    except Exception:
                        continue
                
                if crop_sizes:
                    split_results['avg_crop_size'] = sum(crop_sizes) / len(crop_sizes)
            
            # Analyze labels
            if labels_dir.exists():
                labels_counts = []
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            labels_counts.append(len(lines))
                            
                            # Count class distribution
                            for line in lines:
                                class_id = int(line.split()[0])
                                results['class_distribution'][class_id] += 1
                    except Exception:
                        continue
                
                if labels_counts:
                    split_results['avg_labels_per_crop'] = sum(labels_counts) / len(labels_counts)
                    results['labels_per_crop'].extend(labels_counts)
            
            results['splits'][split] = split_results
            results['total_crops'] += split_results['images']
        
        logger.info(f"📊 Crop Analysis Results:")
        logger.info(f"   Total crops: {results['total_crops']}")
        for split, split_data in results['splits'].items():
            logger.info(f"   {split.upper()}: {split_data['images']} images, {split_data['labels']} labels")
            if split_data['avg_labels_per_crop']:
                logger.info(f"      Avg labels per crop: {split_data['avg_labels_per_crop']:.1f}")
        
        logger.info(f"   Class distribution: {dict(results['class_distribution'])}")
        
        return results
    
    def get_crop_statistics(self) -> Dict:
        """Get detailed crop statistics"""
        if not self.analysis_results:
            return {"error": "No crop data analyzed"}
        
        stats = {
            'total_crops': self.analysis_results['total_crops'],
            'splits': self.analysis_results['splits'],
            'class_distribution': dict(self.analysis_results['class_distribution']),
        }
        
        # Calculate size statistics
        if self.analysis_results['crop_sizes']:
            areas = [w * h for w, h in self.analysis_results['crop_sizes']]
            stats['crop_size_stats'] = {
                'min_area': min(areas),
                'max_area': max(areas),
                'avg_area': sum(areas) / len(areas),
                'min_size': min(self.analysis_results['crop_sizes']),
                'max_size': max(self.analysis_results['crop_sizes'])
            }
        
        # Calculate label statistics
        if self.analysis_results['labels_per_crop']:
            labels_counts = self.analysis_results['labels_per_crop']
            stats['label_stats'] = {
                'min_labels': min(labels_counts),
                'max_labels': max(labels_counts),
                'avg_labels': sum(labels_counts) / len(labels_counts),
                'crops_with_labels': len([c for c in labels_counts if c > 0]),
                'crops_without_labels': len([c for c in labels_counts if c == 0])
            }
        
        return stats

def validate_2stage_dataset(annotations_file: str, images_dir: str, class_group: str = 'bone-loss') -> bool:
    """
    Convenience function to validate dataset for 2-stage training
    
    Args:
        annotations_file: Path to COCO annotations
        images_dir: Path to images directory
        class_group: Target landmark class group
    
    Returns:
        True if dataset is ready for training
    """
    validator = TwoStageDatasetValidator(annotations_file, images_dir, class_group)
    
    print(validator.get_validation_summary())
    
    return validator.is_dataset_ready()

def analyze_crop_dataset(crops_dataset_dir: str) -> Dict:
    """
    Convenience function to analyze extracted crop dataset
    
    Args:
        crops_dataset_dir: Path to crops dataset directory
    
    Returns:
        Analysis results dictionary
    """
    analyzer = CropDatasetAnalyzer(crops_dataset_dir)
    return analyzer.get_crop_statistics()

def main():
    """CLI interface for dataset validation and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='2-Stage Dataset Validation and Analysis')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset for 2-stage training')
    validate_parser.add_argument('--annotations', required=True, help='Path to annotations.json')
    validate_parser.add_argument('--images', required=True, help='Path to images directory')
    validate_parser.add_argument('--class-group', default='bone-loss', 
                                choices=['bone-loss', 'conditions', 'surfaces', 'cej', 'ac', 'apex', 'mesial', 'distal'],
                                help='Target landmark class group')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze extracted crop dataset')
    analyze_parser.add_argument('--crops-dir', required=True, help='Path to crops dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        print("🔍 === 2-Stage Dataset Validation ===")
        is_ready = validate_2stage_dataset(args.annotations, args.images, args.class_group)
        
        if is_ready:
            print("✅ Dataset is ready for 2-stage training!")
            exit(0)
        else:
            print("❌ Dataset has issues that need to be resolved")
            exit(1)
    
    elif args.command == 'analyze':
        print("📊 === Crop Dataset Analysis ===")
        stats = analyze_crop_dataset(args.crops_dir)
        
        import json
        print(json.dumps(stats, indent=2))
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
