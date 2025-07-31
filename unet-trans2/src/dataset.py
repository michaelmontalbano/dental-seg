#!/usr/bin/env python3
"""
Dataset class for CEJ/AC bounding box detection
TransUnet
src/dataset.py
FIXED: Re-enabled X-ray type filtering, supports comma-separated values
"""

import json
import os
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T
import logging
import os

# Configure logging level from environment variable
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Define class mappings for different class groups
CLASS_GROUPS = {
    'conditions': [
        'bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
        'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'
    ],
    'surfaces': ['distal surface', 'occlusal surface', 'mesial surface'],
    'bone-loss': ['apex', 'cej mesial', 'cej distal', 'ac distal', 'ac mesial'],
    'teeth': [  # You can add all Txx and Pxx labels here or filter by regex
        'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10',
        'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20',
        'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30',
        'T31', 'T32',
        'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10',
        'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20'
    ]
}


class BboxDataset(torch.utils.data.Dataset):
    """
    Dataset class for pure bounding box detection
    No masks - just bounding boxes for CEJ/AC landmarks
    """
    
    def __init__(self, annotations_file, images_dir, transforms=None, class_group='all', xray_type=None, img_size=512):
        self.images_dir = images_dir
        self.transforms = transforms
        self.class_group = class_group
        self.xray_type = xray_type
        self.img_size = img_size  # Make image size configurable

        if class_group != 'all' and class_group not in CLASS_GROUPS:
            raise ValueError(f"Invalid class_group: {class_group}. Must be 'all' or one of: {list(CLASS_GROUPS.keys())}")

        # Validate xray_type - NOW ENABLED for train.json/val.json compatibility
        if xray_type is not None:
            # Handle comma-separated xray types
            allowed_types = ['bitewing', 'periapical', 'panoramic']
            xray_types = [t.strip().lower() for t in xray_type.split(',')]
            for xtype in xray_types:
                if xtype not in allowed_types:
                    logger.warning(f"⚠️ Unknown X-ray type: {xtype}. Allowed: {allowed_types}")
            logger.info(f"✅ X-ray type filtering enabled: {xray_types}")

        # Set class names based on class group
        if class_group == 'all':
            # Use all classes from all groups
            self.class_names = []
            for group_classes in CLASS_GROUPS.values():
                self.class_names.extend(group_classes)
        else:
            # Use classes from specific group
            self.class_names = CLASS_GROUPS[class_group]
        
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(self.class_names)}  # +1 for background
            
        logger.info(f"Class group: {class_group}")
        if xray_type:
            logger.info(f"X-ray type filter: {xray_type}")
        
        # Load annotations_merged
        with open(annotations_file, 'r') as f:
            self.annotations_data = json.load(f)
        
        # Debug: Print annotation structure
        logger.info(f"Annotations type: {type(self.annotations_data)}")
        
        # Handle annotations_merged format
        self.annotations = self._process_annotations_merged()
        
        # Filter annotations to only include relevant classes for this class group
        self.filtered_data = self._filter_annotations()
        
        logger.info(f"Loaded {len(self.filtered_data)} images with {class_group} class annotations")
        
    def _process_annotations_merged(self):
        """Process annotations_merged format"""
        logger.info("Processing annotations_merged format")
        
        processed_data = []
        
        if isinstance(self.annotations_data, dict):
            # Check for COCO-like format first (presence of 'images' and 'categories')
            if 'images' in self.annotations_data and 'categories' in self.annotations_data:
                processed_data = self._convert_coco_to_custom_format()
            elif 'annotations' in self.annotations_data:
                # If it has 'annotations' key, use that directly (assuming it's already in the desired custom format)
                # This branch would be for a dataset that's already in the custom format, not COCO raw annotations
                processed_data = self.annotations_data['annotations']
            else:
                # Direct dictionary format (e.g., image_id: {image_info, annotations})
                processed_data = list(self.annotations_data.values())
        elif isinstance(self.annotations_data, list):
            # If it's a list, assume it's already in the custom format (list of image dicts)
            processed_data = self.annotations_data
        else:
            raise ValueError(f"Unexpected annotations_merged format: {type(self.annotations_data)}")
        
        logger.info(f"Processed {len(processed_data)} annotation entries")
        return processed_data
    

    def _convert_coco_to_custom_format(self):
        """Convert COCO format to our custom format - FIXED VERSION with X-ray filtering"""
        logger.info("Converting COCO format to custom format")
        
        # Get categories mapping
        categories = {cat['id']: cat['name'] for cat in self.annotations_data['categories']}
        logger.info(f"Available categories: {len(categories)} total")
        
        # Group annotations by image_id
        image_annotations = {}
        segmentation_count = 0  # Track how many have segmentation data
        
        for ann in self.annotations_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            category_id = ann.get('category_id')
            class_name = categories.get(category_id, 'unknown_category') 
            
            if class_name == 'unknown_category':
                logger.warning(f"Category ID {category_id} not found in categories mapping for annotation: {ann}")
            
            # ✅ FIXED: Copy ALL relevant fields including segmentation
            custom_ann = {
                'bbox': ann['bbox'],
                'class_name': class_name, 
                'category_id': category_id
            }
            
            # ✅ CRITICAL FIX: Preserve segmentation data
            if 'segmentation' in ann:
                custom_ann['segmentation'] = ann['segmentation']
                segmentation_count += 1
                
            # Also check for other possible mask fields
            for mask_field in ['mask', 'masks']:
                if mask_field in ann:
                    custom_ann[mask_field] = ann[mask_field]
            
            image_annotations[image_id].append(custom_ann)
        
        # ✅ Log segmentation statistics
        total_annotations = len(self.annotations_data['annotations'])
        logger.info(f"✅ Preserved segmentation data for {segmentation_count}/{total_annotations} annotations ({segmentation_count/total_annotations*100:.1f}%)")
        
        # Create custom format data with xray_type filtering
        custom_data = []
        total_images = len(self.annotations_data['images'])
        filtered_count = 0
        
        for img in self.annotations_data['images']:
            img_id = img['id']
            
            # Filter by xray_type if specified (handle comma-separated values)
            if self.xray_type is not None:
                img_xray_type = img.get('xray_type', '').lower()
                # Handle comma-separated xray types
                allowed_types = [t.strip().lower() for t in self.xray_type.split(',')]
                if img_xray_type not in allowed_types:
                    continue  # Skip this image
            
            if img_id in image_annotations:
                custom_item = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height'],
                    'id': img_id,
                    'annotations': image_annotations[img_id],
                    'xray_type': img.get('xray_type', 'unknown'),
                    'class_group': img.get('class_group', self.class_group)
                }
                custom_data.append(custom_item)
                filtered_count += 1
        
        if self.xray_type:
            logger.info(f"X-ray type filtering: {filtered_count}/{total_images} images match '{self.xray_type}'")
        else:
            logger.info(f"No X-ray type filtering: using all {filtered_count} images")
        
        logger.info(f"Converted {len(custom_data)} images from COCO format")
        return custom_data

    def _filter_annotations(self):
        """Filter annotations to only include relevant classes for the specified class group"""
        target_classes = set(self.class_names)
        
        filtered_data = []
        
        # Ensure we have a list to iterate over
        if not isinstance(self.annotations, list):
            logger.error(f"Expected list, got {type(self.annotations)}")
            return []
        
        # Debug: Log first few class names found
        class_names_found = set()
        
        for i, item in enumerate(self.annotations):
            # Debug: print what we're getting
            if i < 3:  # Only log first 3 items
                logger.info(f"Processing item {i}, type: {type(item)}")
            
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {i}: {type(item)}")
                continue
            
            # Check if this image matches our class group (if specified in annotations)
            item_class_group = item.get('class_group', self.class_group)
            if self.class_group != 'all' and item_class_group != self.class_group:
                continue
                
            # Filter annotations
            relevant_annotations = []
            annotations = item.get('annotations', [])
            
            # Handle case where annotations might also be strings
            if not isinstance(annotations, list):
                logger.warning(f"Annotations is not a list at index {i}: {type(annotations)}")
                continue
            
            for ann in annotations:
                if not isinstance(ann, dict):
                    logger.warning(f"Skipping non-dict annotation: {type(ann)}")
                    continue
                    
                class_name = ann.get('class_name', '')
                class_names_found.add(class_name)  # Collect all class names for debugging
                
                # Check if this class is relevant for our class group
                if class_name in target_classes:
                    # Keep original class name
                    ann['normalized_class'] = class_name
                    relevant_annotations.append(ann)
                else:
                    # Check if it's a composite class that should be included
                    # (e.g., "T14_CEJ" for bone-loss classes)
                    for target_class in target_classes:
                        if target_class in class_name or class_name in target_class:
                            ann['normalized_class'] = target_class
                            relevant_annotations.append(ann)
                            break
            
            # Only keep images with relevant annotations
            if relevant_annotations:
                item['relevant_annotations'] = relevant_annotations
                filtered_data.append(item)
        
        # DEBUGGING: Log all unique class names found in dataset
        logger.info(f"All unique class names found in dataset: {sorted(class_names_found)}")
        logger.info(f"Target classes we're looking for: {sorted(target_classes)}")
        logger.info(f"Filtered {len(filtered_data)} items with target classes for {self.class_group}")
        
        # Additional debugging: Check if we have any relevant annotations at all
        total_annotation_count = sum(len(item.get('relevant_annotations', [])) for item in filtered_data)
        logger.info(f"Total relevant annotations found: {total_annotation_count}")
        
        return filtered_data
    
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        # Load image
        item = self.filtered_data[idx]
        
        # Handle different possible file name fields
        file_name = None
        if isinstance(item, dict):
            file_name = item.get('file_name') or item.get('image_path') or item.get('filename')
        else:
            # If item is a string, it might be the filename itself
            file_name = str(item)
        
        if not file_name:
            raise ValueError(f"No valid file name found in item: {item}")
            
        image_path = os.path.join(self.images_dir, file_name)
        
        if not os.path.exists(image_path):
            # Try alternative path formats
            if isinstance(item, dict):
                alt_paths = [
                    item.get('image_path', ''),
                    item.get('file_name', ''),
                    os.path.basename(item.get('file_name', ''))
                ]
            else:
                alt_paths = [str(item), os.path.basename(str(item))]
            
            for alt_path in alt_paths:
                if alt_path:  # Skip empty strings
                    full_path = os.path.join(self.images_dir, alt_path)
                    if os.path.exists(full_path):
                        image_path = full_path
                        break
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Get bounding boxes and labels - NO MASKS
        boxes = []
        labels = []
        
        # Handle annotations
        annotations = []
        if isinstance(item, dict):
            annotations = item.get('relevant_annotations', [])
        
        for ann in annotations:
            # Get bounding box
            bbox = ann.get('bbox', ann.get('bounding_box', []))
            if len(bbox) == 4:
                x, y, w, h = bbox
                # Convert to [x1, y1, x2, y2] format
                boxes.append([x, y, x + w, y + h])
                
                # Get class label with error handling
                class_name = ann['normalized_class']
                if class_name not in self.class_to_idx:
                    logger.error(f"Class '{class_name}' not found in class_to_idx mapping: {self.class_to_idx}")
                    logger.error(f"Available classes: {list(self.class_to_idx.keys())}")
                    raise KeyError(f"Class '{class_name}' not in mapping. Check class name consistency.")
                labels.append(self.class_to_idx[class_name])
        
        # Convert to tensors
        if len(boxes) == 0:
            # Handle empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary - NO MASKS
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([])
        }
        
        # Apply transforms to image only (not target)
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Resize image and adjust bounding boxes accordingly
        original_size = image.shape[-2:]  # (H, W)
        if original_size != (self.img_size, self.img_size):
            # Resize image
            resize_transform = T.Resize((self.img_size, self.img_size))
            image = resize_transform(image)
            
            # Scale bounding boxes
            if len(boxes) > 0:
                h_scale = self.img_size / original_size[0]
                w_scale = self.img_size / original_size[1]
                
                # Scale boxes: [x1, y1, x2, y2] format
                boxes[:, 0] *= w_scale  # x1
                boxes[:, 1] *= h_scale  # y1
                boxes[:, 2] *= w_scale  # x2
                boxes[:, 3] *= h_scale  # y2
        
        # Update target with scaled boxes
        target["boxes"] = boxes
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([])
        
        return image, target

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))
