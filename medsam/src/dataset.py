#!/usr/bin/env python3
"""
MedSAM dataset.py
Dataset class for MedSAM dental landmark segmentation
Interactive prompt-based segmentation using MedSAM format
"""

import json
import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
from sklearn.model_selection import train_test_split
from collections import Counter

logger = logging.getLogger(__name__)

# Define class mappings for different class groups
CLASS_GROUPS = {
    'conditions': ['decay'],
    'surfaces': ['distal surface', 'occlusal surface', 'mesial surface'],
    'bone-loss': ['apex', 'cej mesial', 'cej distal', 'ac distal', 'ac mesial'],
    'teeth': [
        'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10',
        'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20'
    ]
}

class DentalMedSAMDataset:
    """
    Dataset class for MedSAM dental landmark segmentation
    Converts annotations to MedSAM format with prompt generation
    """
    
    def __init__(self, annotations_file, images_dir, output_dir="medsam_dataset", 
                 class_group='bone-loss', xray_type=None, prompt_mode='auto'):
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        self.class_group = class_group
        self.xray_type = xray_type
        self.prompt_mode = prompt_mode
        
        # Validate class group
        if class_group != 'all' and class_group not in CLASS_GROUPS:
            raise ValueError(f"Invalid class_group: {class_group}. Must be 'all' or one of: {list(CLASS_GROUPS.keys())}")
        
        # Set class names based on class group
        if class_group == 'all':
            self.class_names = []
            for group_classes in CLASS_GROUPS.values():
                self.class_names.extend(group_classes)
        else:
            self.class_names = CLASS_GROUPS[class_group]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        logger.info(f"🔍 MedSAM Dataset Debugging Started")
        logger.info(f"📁 Annotations file: {annotations_file}")
        logger.info(f"📁 Images directory: {images_dir}")
        logger.info(f"🎯 Class group: '{self.class_group}'")
        logger.info(f"🎯 Prompt mode: '{self.prompt_mode}'")
        if self.xray_type:
            logger.info(f"📷 X-ray type filter: '{self.xray_type}'")
        logger.info(f"🎯 Target classes: {self.class_names}")
        logger.info(f"🗂️ Class mapping: {self.class_to_idx}")
        
        # Load annotations with debugging
        logger.info(f"📖 Loading annotations from {annotations_file}")
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        logger.info(f"✅ Loaded annotations, type: {type(self.annotations)}")
        
        if isinstance(self.annotations, dict):
            logger.info(f"📋 Annotation keys: {list(self.annotations.keys())}")
            if 'images' in self.annotations:
                logger.info(f"🖼️ Number of images in COCO: {len(self.annotations['images'])}")
            if 'annotations' in self.annotations:
                logger.info(f"🏷️ Number of annotations in COCO: {len(self.annotations['annotations'])}")
            if 'categories' in self.annotations:
                logger.info(f"📊 Number of categories in COCO: {len(self.annotations['categories'])}")
        
        # Process annotations to MedSAM format
        self.filtered_data = self._filter_annotations_with_debug()
        logger.info(f"✅ Final filtered data: {len(self.filtered_data)} items with target annotations")
    
    def _generate_prompts_for_annotation(self, annotation):
        """Generate prompts for a single annotation based on prompt mode"""
        prompts = []
        bbox = annotation.get('bbox', [])
        segmentation = annotation.get('segmentation', [])
        
        if len(bbox) != 4:
            return prompts
        
        x, y, w, h = bbox
        
        if self.prompt_mode in ['auto', 'mixed']:
            # Generate point prompts within the bbox
            num_points = 3  # Default number of positive points
            for _ in range(num_points):
                # Sample points within the bbox
                px = x + np.random.uniform(0.2 * w, 0.8 * w)
                py = y + np.random.uniform(0.2 * h, 0.8 * h)
                
                prompts.append({
                    'type': 'point',
                    'coordinates': [px, py],
                    'label': 1  # positive
                })
            
            # Add negative points outside the bbox
            for _ in range(1):  # One negative point
                # Sample point outside bbox
                if np.random.random() < 0.5:
                    # Left or right of bbox
                    px = x - np.random.uniform(20, 50) if np.random.random() < 0.5 else x + w + np.random.uniform(20, 50)
                    py = y + np.random.uniform(0, h)
                else:
                    # Above or below bbox
                    px = x + np.random.uniform(0, w)
                    py = y - np.random.uniform(20, 50) if np.random.random() < 0.5 else y + h + np.random.uniform(20, 50)
                
                prompts.append({
                    'type': 'point',
                    'coordinates': [px, py],
                    'label': 0  # negative
                })
        
        if self.prompt_mode in ['manual', 'mixed']:
            # Add bbox prompt
            prompts.append({
                'type': 'box',
                'coordinates': bbox,
                'label': 1
            })
        
        return prompts
    
    def _filter_annotations_with_debug(self):
        """Filter annotations with extensive debugging for MedSAM"""
        target_classes = set(self.class_names)
        
        logger.info(f"🎯 Filtering for target classes: {target_classes}")
        
        filtered_data = []
        all_class_names_found = set()
        class_counts = Counter()
        
        # Handle COCO format conversion with debugging
        if isinstance(self.annotations, dict):
            if 'images' in self.annotations and 'annotations' in self.annotations:
                logger.info("🔄 Converting COCO format to custom format...")
                self.annotations = self._convert_coco_to_custom_format_debug()
            else:
                logger.info("🔄 Converting dict to list format...")
                self.annotations = list(self.annotations.values())
        
        logger.info(f"📝 Processing {len(self.annotations)} annotation items...")
        
        for i, item in enumerate(self.annotations):
            if not isinstance(item, dict):
                continue
            
            # Apply X-ray type filtering
            if self.xray_type:
                item_xray_type = item.get('xray_type', '').lower()
                if item_xray_type != self.xray_type.lower():
                    continue
            
            # Process annotations for this item
            landmark_annotations = []
            annotations = item.get('annotations', [])
            
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                    
                class_name = ann.get('class_name', '')
                all_class_names_found.add(class_name)
                class_counts[class_name] += 1
                
                if class_name in target_classes:
                    # Keep original class name and add prompt information
                    ann['normalized_class'] = class_name
                    ann['prompts'] = self._generate_prompts_for_annotation(ann)
                    landmark_annotations.append(ann)
            
            # Only keep images with landmark annotations
            if landmark_annotations:
                item['landmark_annotations'] = landmark_annotations
                filtered_data.append(item)
        
        logger.info(f"✅ Final filtered data: {len(filtered_data)} items")
        return filtered_data
    
    def _convert_coco_to_custom_format_debug(self):
        """Convert COCO format with debugging for MedSAM"""
        logger.info("🔄 Converting COCO format to custom format for MedSAM...")
        
        # Get categories mapping with debugging
        categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        logger.info(f"📊 COCO Categories found: {len(categories)}")
        
        # Group annotations by image_id
        image_annotations = {}
        
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            category_name = categories.get(ann['category_id'], 'unknown')
            
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            # Convert COCO annotation to our format with MedSAM-specific data
            custom_ann = {
                'bbox': ann.get('bbox', []),
                'segmentation': ann.get('segmentation', []),
                'class_name': category_name,
                'category_id': ann['category_id']
            }
            image_annotations[image_id].append(custom_ann)
        
        # Create custom format data
        custom_data = []
        
        for img in self.annotations['images']:
            img_id = img['id']
            
            # Apply X-ray type filtering during COCO conversion
            if self.xray_type:
                img_xray_type = img.get('xray_type', '').lower()
                if img_xray_type != self.xray_type.lower():
                    continue
            
            if img_id in image_annotations:
                custom_item = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height'],
                    'id': img_id,
                    'annotations': image_annotations[img_id],
                    'xray_type': img.get('xray_type', 'unknown')
                }
                custom_data.append(custom_item)
        
        logger.info(f"✅ Converted {len(custom_data)} images from COCO format")
        return custom_data
    
    def apply_clahe_enhancement(self, image_path, output_path):
        """Apply CLAHE enhancement for better contrast"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"⚠️ Could not read image for CLAHE: {image_path}. Copying original.")
            shutil.copy2(image_path, output_path)
            return output_path
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        cv2.imwrite(output_path, enhanced)
        return output_path
    
    def create_medsam_dataset(self, train_split=0.8, val_split=0.1, test_split=0.1, 
                             apply_enhancement=True, subset_size=None):
        """Create MedSAM format dataset with train/val/test splits"""
        
        logger.info(f"🚀 Creating MedSAM dataset...")
        
        # Apply subset sampling if specified
        data_to_use = self.filtered_data
        if subset_size and subset_size < len(self.filtered_data):
            import random
            random.seed(42)  # For reproducibility
            data_to_use = random.sample(self.filtered_data, subset_size)
            logger.info(f"🧪 Using subset of {subset_size} samples from {len(self.filtered_data)} total")
        
        # Create directory structure
        dataset_dir = self.output_dir
        for split in ['train', 'val', 'test']:
            (dataset_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_data, temp_data = train_test_split(
            data_to_use, test_size=(1-train_split), random_state=42
        )
        val_size = val_split / (val_split + test_split)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1-val_size), random_state=42
        )
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        total_samples = 0
        
        for split_name, split_data in splits.items():
            logger.info(f"📝 Processing {split_name} split: {len(split_data)} images")
            
            split_samples = []
            for item in split_data:
                # Get image info
                file_name = item.get('file_name') or item.get('image_path') or item.get('filename')
                if not file_name:
                    continue
                
                img_width = item.get('width', 0)
                img_height = item.get('height', 0)
                
                # Handle image path
                image_path = os.path.join(self.images_dir, file_name)
                if not os.path.exists(image_path):
                    logger.warning(f"⚠️ Image not found: {image_path}")
                    continue
                
                # Load image to get dimensions if not provided
                if img_width == 0 or img_height == 0:
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                
                # Copy/enhance image
                img_output_path = dataset_dir / split_name / file_name
                if apply_enhancement:
                    self.apply_clahe_enhancement(image_path, str(img_output_path))
                else:
                    shutil.copy2(image_path, img_output_path)
                
                # Create MedSAM samples for each annotation
                annotations = item.get('landmark_annotations', [])
                for ann in annotations:
                    class_name = ann['normalized_class']
                    if class_name not in self.class_to_idx:
                        logger.warning(f"⚠️ Class '{class_name}' not in mapping: {self.class_to_idx}")
                        continue
                    
                    class_id = self.class_to_idx[class_name]
                    bbox = ann.get('bbox', [])
                    segmentation = ann.get('segmentation', [])
                    prompts = ann.get('prompts', [])
                    
                    if len(bbox) == 4:
                        # Create MedSAM sample
                        sample = {
                            'image_path': str(img_output_path),
                            'class_name': class_name,
                            'class_id': class_id,
                            'bbox': bbox,
                            'segmentation': segmentation,
                            'prompts': prompts,
                            'image_width': img_width,
                            'image_height': img_height
                        }
                        split_samples.append(sample)
                        total_samples += 1
            
            # Save split samples
            split_file = dataset_dir / f'{split_name}_samples.json'
            with open(split_file, 'w') as f:
                json.dump(split_samples, f, indent=2)
            
            logger.info(f"✅ {split_name}: {len(split_samples)} MedSAM samples created")
        
        logger.info(f"📊 Total MedSAM samples created: {total_samples}")
        
        # Create dataset config
        self.create_config_file(dataset_dir)
        
        logger.info(f"✅ MedSAM dataset created successfully at {dataset_dir}")
        return dataset_dir
    
    def create_config_file(self, dataset_dir):
        """Create config file for MedSAM training"""
        config = {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'num_classes': len(self.class_names),
            'image_size': 1024,  # MedSAM standard size
            'class_group': self.class_group,
            'prompt_mode': self.prompt_mode
        }
        
        config_path = dataset_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📄 Created config.json at {config_path}")
        logger.info(f"🎯 Classes in config: {config['class_names']}")
        logger.info(f"📊 Number of classes: {config['num_classes']}")
        logger.info(f"🎯 Prompt mode: {config['prompt_mode']}")
        
        return config_path

def main():
    """Example usage with debugging for MedSAM"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create MedSAM dataset for dental landmarks')
    parser.add_argument('--annotations', required=True, help='Path to annotations.json')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--output', default='medsam_dataset', help='Output directory')
    parser.add_argument('--enhance', action='store_true', help='Apply CLAHE enhancement')
    parser.add_argument('--subset', type=int, default=None, help='Use subset of N samples for testing')
    parser.add_argument('--class_group', type=str, default='bone-loss', 
                        help='Class group to create dataset for (e.g., teeth, bone-loss)')
    parser.add_argument('--xray_type', type=str, default=None, 
                        help='Filter dataset by xray_type (e.g., panoramic, bitewing, periapical)')
    parser.add_argument('--prompt_mode', type=str, default='auto',
                        choices=['auto', 'manual', 'mixed'],
                        help='Prompt generation mode')
    
    args = parser.parse_args()
    
    # Create MedSAM dataset with debugging
    dataset = DentalMedSAMDataset(
        args.annotations, 
        args.images, 
        args.output,
        class_group=args.class_group,
        xray_type=args.xray_type,
        prompt_mode=args.prompt_mode
    )
    dataset_dir = dataset.create_medsam_dataset(
        apply_enhancement=args.enhance,
        subset_size=args.subset
    )
    
    print(f"MedSAM dataset created at: {dataset_dir}")
    print(f"Classes: {dataset.class_names}")
    print(f"Prompt mode: {dataset.prompt_mode}")
    print(f"Total items processed: {len(dataset.filtered_data) if not args.subset else args.subset}")

if __name__ == '__main__':
    main()
