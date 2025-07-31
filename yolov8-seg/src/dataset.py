#!/usr/bin/env python3
"""
YOLOv8 segmentation dataset.py
Dataset class for YOLOv8 dental landmark segmentation
CEJ, Alveolar Crest, and Apex segmentation using instance segmentation format
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
    'bone-loss': ['cej mesial', 'cej distal', 'ac distal', 'ac mesial', 'apex'],
    'teeth': [f"T{str(i).zfill(2)}" for i in range(1, 33)] + [f"P{str(i).zfill(2)}" for i in range(1, 21)],
    'conditions': ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                   'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'],
    'surfaces': ['distal surface', 'occlusal surface', 'mesial surface'],
    'dental-work': ['bridge', 'filling', 'crown', 'implant'],
    # Individual class groups
    'mesial': ['cej mesial', 'ac mesial'],
    'tooth-anatomy': ['enamel', 'pulp'],
    'distal': ['ac distal', 'cej distal'],
    'apex': ['apex'],
    'permanent-teeth': [f"T{str(i).zfill(2)}" for i in range(1, 33)],
    'primary-teeth': [f"P{str(i).zfill(2)}" for i in range(1, 21)],
    'bridge': ['bridge'],
    'margin': ['margin'],
    'enamel': ['enamel'],
    'tooth-aspects': ['coronal aspect', 'root aspect'],
    'pulp': ['pulp'],
    'filling': ['filling'],
    'crown': ['crown'],
    'impaction': ['impaction'],
    'decay': ['decay'],
    'rct': ['rct'],
    'parl': ['parl'],
    'missing': ['missing'],
    'implant': ['implant'],
    'calculus': ['calculus'],
    'distal-surface': ['distal surface'],
    'mesial-surface': ['mesial surface'],
    'occlusal-surface': ['occlusal surface'],
    'cej': ['cej mesial', 'cej distal', 'apex'],
    'ac': ['ac mesial', 'ac distal', 'apex'],
    'decay-enamel-pulp-coronal': ['decay', 'enamel', 'pulp', 'coronal aspect'],
    'parl-rct-pulp-root': ['parl', 'rct', 'pulp', 'root aspect'],
}
class DentalYOLOv8SegDataset:
    """
    Dataset class for YOLOv8 segmentation dental landmark detection
    Converts segmentation annotations to YOLO segmentation format
    """
    
    def __init__(self, annotations_file, images_dir, output_dir="yolo_seg_dataset", 
                 class_group='bone-loss', xray_type=None):
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        self.class_group = class_group
        self.xray_type = xray_type
        
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
        
        logger.info(f"🔍 YOLOv8 Segmentation Dataset Debugging Started")
        logger.info(f"📁 Annotations file: {annotations_file}")
        logger.info(f"📁 Images directory: {images_dir}")
        logger.info(f"🎯 Class group: '{self.class_group}'")
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
        
        # Process annotations to YOLO segmentation format
        self.filtered_data = self._filter_annotations_with_debug()
        logger.info(f"✅ Final filtered data: {len(self.filtered_data)} items with target annotations")
    
    def _filter_annotations_with_debug(self):
        """Filter annotations with extensive debugging for segmentation"""
        target_classes = set(self.class_names)
        
        logger.info(f"🎯 Filtering for target classes: {target_classes}")
        
        filtered_data = []
        all_class_names_found = set()
        class_counts = Counter()
        xray_type_counts = Counter()
        segmentation_counts = {'with_segmentation': 0, 'bbox_only': 0}
        
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
            if i < 5:  # Debug first 5 items
                logger.info(f"🔍 Item {i}: type={type(item)}, keys={list(item.keys()) if isinstance(item, dict) else 'N/A'}")
            
            if not isinstance(item, dict):
                if i < 5:
                    logger.warning(f"⚠️ Skipping non-dict item at index {i}: {type(item)}")
                continue
            
            # Apply X-ray type filtering
            if self.xray_type:
                item_xray_type = item.get('xray_type', '').lower()
                xray_type_counts[item_xray_type] += 1
                if item_xray_type != self.xray_type.lower():
                    continue
            
            # Process annotations for this item
            landmark_annotations = []
            annotations = item.get('annotations', [])
            
            if i < 5:
                logger.info(f"🏷️ Item {i} has {len(annotations)} annotations")
            
            if not isinstance(annotations, list):
                if i < 5:
                    logger.warning(f"⚠️ Annotations not a list at index {i}: {type(annotations)}")
                continue
            
            for j, ann in enumerate(annotations):
                if not isinstance(ann, dict):
                    if i < 5:
                        logger.warning(f"⚠️ Skipping non-dict annotation {j} in item {i}: {type(ann)}")
                    continue
                    
                class_name = ann.get('class_name', '')
                all_class_names_found.add(class_name)
                class_counts[class_name] += 1
                
                # Check for segmentation data
                has_segmentation = bool(ann.get('segmentation', []))
                has_bbox = bool(ann.get('bbox', []))
                
                if has_segmentation:
                    segmentation_counts['with_segmentation'] += 1
                elif has_bbox:
                    segmentation_counts['bbox_only'] += 1
                
                if i < 3 and j < 3:  # Debug first few annotations
                    logger.info(f"🏷️ Item {i}, Ann {j}: class='{class_name}', "
                              f"bbox={bool(has_bbox)}, segmentation={bool(has_segmentation)}")
                
                if class_name in target_classes:
                    # Keep original class name and segmentation data
                    ann['normalized_class'] = class_name
                    landmark_annotations.append(ann)
                    
                    if i < 5:
                        logger.info(f"✅ Item {i}: Found target class '{class_name}' "
                                  f"(seg: {has_segmentation}, bbox: {has_bbox})")
            
            # Only keep images with landmark annotations
            if landmark_annotations:
                item['landmark_annotations'] = landmark_annotations
                filtered_data.append(item)
                
                if len(filtered_data) <= 5:
                    logger.info(f"✅ Added item {i} to filtered data (total landmarks: {len(landmark_annotations)})")
        
        # COMPREHENSIVE DEBUGGING OUTPUT
        logger.info(f"\n📊 === SEGMENTATION DATASET ANALYSIS ===")
        logger.info(f"🔍 All unique class names found: {sorted(all_class_names_found)}")
        logger.info(f"🎯 Target classes we want: {sorted(target_classes)}")
        logger.info(f"✅ Classes found AND wanted: {sorted(target_classes.intersection(all_class_names_found))}")
        logger.info(f"❌ Classes wanted but NOT found: {sorted(target_classes - all_class_names_found)}")
        logger.info(f"ℹ️ Classes found but NOT wanted: {sorted(all_class_names_found - target_classes)}")
        
        logger.info(f"\n🎭 Segmentation data availability:")
        logger.info(f"  ✅ With segmentation masks: {segmentation_counts['with_segmentation']}")
        logger.info(f"  📦 Bbox only (will convert to rectangles): {segmentation_counts['bbox_only']}")
        
        if self.xray_type:
            logger.info(f"\n📷 X-ray type distribution:")
            for xray_type, count in xray_type_counts.most_common():
                marker = "✅" if xray_type == self.xray_type.lower() else "❌"
                logger.info(f"  {marker} '{xray_type}': {count} images")
        
        logger.info(f"\n📈 Class frequency distribution:")
        for class_name in sorted(target_classes):
            count = class_counts.get(class_name, 0)
            logger.info(f"  '{class_name}': {count} annotations")
        
        logger.info(f"\n📝 Filtering results:")
        logger.info(f"  📥 Input items: {len(self.annotations)}")
        logger.info(f"  📤 Output items: {len(filtered_data)}")
        logger.info(f"  🏷️ Total target annotations: {sum(class_counts[c] for c in target_classes)}")
        
        if len(filtered_data) == 0:
            logger.error("🚨 CRITICAL: No items passed filtering! Check class names and data format.")
            logger.error(f"🚨 Available classes: {list(all_class_names_found)[:10]}")
            
        return filtered_data
    
    def _convert_coco_to_custom_format_debug(self):
        """Convert COCO format with debugging for segmentation"""
        logger.info("🔄 Converting COCO format to custom format for segmentation...")
        
        # Get categories mapping with debugging
        categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        logger.info(f"📊 COCO Categories found: {len(categories)}")
        
        # Log categories that match our targets
        target_classes = set(self.class_names)
        matching_categories = {id: name for id, name in categories.items() if name in target_classes}
        logger.info(f"🎯 Matching categories: {matching_categories}")
        
        # Group annotations by image_id with debugging
        image_annotations = {}
        total_anns = len(self.annotations['annotations'])
        target_ann_count = 0
        segmentation_ann_count = 0
        
        logger.info(f"🏷️ Processing {total_anns} COCO annotations...")
        
        for i, ann in enumerate(self.annotations['annotations']):
            image_id = ann['image_id']
            category_name = categories.get(ann['category_id'], 'unknown')
            
            if i < 5:  # Debug first 5 annotations
                logger.info(f"🏷️ COCO Ann {i}: image_id={image_id}, category_id={ann['category_id']}, "
                          f"name='{category_name}', has_segmentation={bool(ann.get('segmentation'))}")
            
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            # Convert COCO annotation to our format with segmentation data
            custom_ann = {
                'bbox': ann.get('bbox', []),
                'segmentation': ann.get('segmentation', []),
                'class_name': category_name,
                'category_id': ann['category_id']
            }
            image_annotations[image_id].append(custom_ann)
            
            if category_name in target_classes:
                target_ann_count += 1
            
            if ann.get('segmentation'):
                segmentation_ann_count += 1
        
        logger.info(f"✅ Found {target_ann_count} annotations with target classes out of {total_anns} total")
        logger.info(f"🎭 Found {segmentation_ann_count} annotations with segmentation data")
        
        # Create custom format data
        custom_data = []
        images_with_targets = 0
        
        for img in self.annotations['images']:
            img_id = img['id']
            
            # Apply X-ray type filtering during COCO conversion
            if self.xray_type:
                img_xray_type = img.get('xray_type', '').lower()
                if img_xray_type != self.xray_type.lower():
                    continue
            
            if img_id in image_annotations:
                # Check if this image has any target class annotations
                has_targets = any(ann['class_name'] in target_classes 
                                for ann in image_annotations[img_id])
                
                if has_targets:
                    images_with_targets += 1
                
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
        logger.info(f"🎯 {images_with_targets} images contain target classes")
        
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
    
    def segmentation_to_yolo_format(self, segmentation, img_width, img_height):
        """Convert COCO segmentation to YOLO segmentation format"""
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
            # RLE format or other complex formats
            # For now, return None to fall back to bbox
            return None
    
    def bbox_to_yolo_segmentation(self, bbox, img_width, img_height):
        """Convert bounding box to YOLO segmentation format (as rectangle)"""
        x, y, w, h = bbox
        
        # Create rectangle polygon (4 corners)
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
    
    def create_yolo_segmentation_dataset(self, train_split=0.8, val_split=0.1, test_split=0.1, 
                                       apply_enhancement=True, subset_size=None):
        """Create YOLO segmentation format dataset with train/val/test splits"""
        
        logger.info(f"🚀 Creating YOLO segmentation dataset...")
        
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
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
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
        
        total_yolo_annotations = 0
        segmentation_success_count = 0
        bbox_fallback_count = 0
        
        for split_name, split_data in splits.items():
            logger.info(f"📝 Processing {split_name} split: {len(split_data)} images")
            split_annotations = 0
            
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
                img_output_path = dataset_dir / split_name / 'images' / file_name
                if apply_enhancement:
                    self.apply_clahe_enhancement(image_path, str(img_output_path))
                else:
                    shutil.copy2(image_path, img_output_path)
                
                # Create YOLO segmentation format label file
                label_file = dataset_dir / split_name / 'labels' / f"{Path(file_name).stem}.txt"
                
                item_annotation_count = 0
                with open(label_file, 'w') as f:
                    annotations = item.get('landmark_annotations', [])
                    for ann in annotations:
                        class_name = ann['normalized_class']
                        if class_name not in self.class_to_idx:
                            logger.warning(f"⚠️ Class '{class_name}' not in mapping: {self.class_to_idx}")
                            continue
                        
                        class_id = self.class_to_idx[class_name]
                        
                        # Try to get segmentation first, fall back to bbox
                        segmentation = ann.get('segmentation', [])
                        coords = self.segmentation_to_yolo_format(segmentation, img_width, img_height)
                        
                        if coords is None:
                            # Fall back to bbox as rectangle
                            bbox = ann.get('bbox', [])
                            if len(bbox) == 4:
                                coords = self.bbox_to_yolo_segmentation(bbox, img_width, img_height)
                                bbox_fallback_count += 1
                            else:
                                continue
                        else:
                            segmentation_success_count += 1
                        
                        if coords and len(coords) >= 6:  # At least 3 points (6 coordinates)
                            # Write YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                            coord_str = ' '.join([f"{coord:.6f}" for coord in coords])
                            f.write(f"{class_id} {coord_str}\n")
                            item_annotation_count += 1
                            split_annotations += 1
                
                if split_name == 'train' and len(os.listdir(dataset_dir / split_name / 'labels')) <= 5:
                    logger.info(f"📄 Created {label_file} with {item_annotation_count} segmentation annotations")
            
            total_yolo_annotations += split_annotations
            logger.info(f"✅ {split_name}: {split_annotations} YOLO segmentation annotations created")
        
        logger.info(f"📊 Total YOLO segmentation annotations created: {total_yolo_annotations}")
        logger.info(f"🎭 Segmentation success: {segmentation_success_count}")
        logger.info(f"📦 Bbox fallback: {bbox_fallback_count}")
        
        # Create data.yaml file for YOLOv8 segmentation
        self.create_data_yaml(dataset_dir)
        
        logger.info(f"✅ YOLO segmentation dataset created successfully at {dataset_dir}")
        return dataset_dir
    
    def create_data_yaml(self, dataset_dir):
        """Create data.yaml file for YOLOv8 segmentation training"""
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(self.class_names)},
            'nc': len(self.class_names)
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"📄 Created data.yaml at {yaml_path}")
        logger.info(f"🎯 Classes in YAML: {data_yaml['names']}")
        logger.info(f"📊 Number of classes: {data_yaml['nc']}")
        
        return yaml_path

def main():
    """Example usage with debugging for segmentation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create YOLOv8 segmentation dataset for dental landmarks')
    parser.add_argument('--annotations', required=True, help='Path to annotations.json')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--output', default='yolo_seg_dataset', help='Output directory')
    parser.add_argument('--enhance', action='store_true', help='Apply CLAHE enhancement')
    parser.add_argument('--subset', type=int, default=None, help='Use subset of N samples for testing')
    parser.add_argument('--class_group', type=str, default='bone-loss', 
                        help='Class group to create dataset for (e.g., teeth, bone-loss)')
    parser.add_argument('--xray_type', type=str, default=None, 
                        help='Filter dataset by xray_type (e.g., panoramic, bitewing, periapical)')
    
    args = parser.parse_args()
    
    # Create segmentation dataset with debugging
    dataset = DentalYOLOv8SegDataset(
        args.annotations, 
        args.images, 
        args.output,
        class_group=args.class_group,
        xray_type=args.xray_type
    )
    dataset_dir = dataset.create_yolo_segmentation_dataset(
        apply_enhancement=args.enhance,
        subset_size=args.subset
    )
    
    print(f"YOLOv8 segmentation dataset created at: {dataset_dir}")
    print(f"Classes: {dataset.class_names}")
    print(f"Total items processed: {len(dataset.filtered_data) if not args.subset else args.subset}")

if __name__ == '__main__':
    main()
