#!/usr/bin/env python3
"""
Dataset class for CEJ/AC bounding box detection
Fixed version that properly loads annotations before using them
"""

import json
import os
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T
import logging

logger = logging.getLogger(__name__)

class BboxDataset(torch.utils.data.Dataset):
    """
    Dataset class for pure bounding box detection
    No masks - just bounding boxes for CEJ/AC landmarks
    Uses only predefined 8 classes, ignores everything else
    """
    
    def __init__(self, annotations_file, images_dir, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Define your exact 8 target classes - NO AUTO-DISCOVERY
        self.class_names = ["cej_mesial", "cej_distal", "ac_mesial", "ac_distal", "apex", "root_aspect", "coronal_aspect", "calculus"]
        
        # Create proper mapping: background=0, your 8 classes=1,2,3,4,5,6,7,8
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(self.class_names)}
        
        print("🎭" + "="*60)
        print(f"🎯 USING PREDEFINED {len(self.class_names)} CLASSES (ignoring others):")
        for name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            print(f"   {idx}: {name}")
        print(f"🏷️  LABEL RANGE: 1 to {len(self.class_names)}")
        print(f"🚫 IGNORING: All other classes in COCO file")
        print("🎭" + "="*60)
        
        # STEP 1: Load the annotations file first
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # STEP 2: Now we can safely debug what we loaded
        logger.info(f"Annotations type: {type(self.annotations)}")
        
        if isinstance(self.annotations, dict):
            logger.info(f"Annotations is a dictionary with keys: {list(self.annotations.keys())[:5]}...")
            
            # Check if it's COCO format
            if 'images' in self.annotations and 'annotations' in self.annotations:
                logger.info("Detected COCO format")
                self.annotations = self._convert_coco_to_custom_format()
            else:
                # Convert dict to list format
                logger.info("Converting dict format to list format")
                self.annotations = list(self.annotations.values())
        
        elif isinstance(self.annotations, list):
            logger.info(f"Annotations is a list with {len(self.annotations)} items")
            if len(self.annotations) > 0:
                logger.info(f"First item type: {type(self.annotations[0])}")
        
        else:
            raise ValueError(f"Unexpected annotations format: {type(self.annotations)}")
        
        # Filter annotations to only include your 8 landmark classes
        self.filtered_data = self._filter_annotations()
        
        logger.info(f"Loaded {len(self.filtered_data)} images with landmark annotations")
        
    def _convert_coco_to_custom_format(self):
        """Convert COCO format to our custom format"""
        logger.info("Converting COCO format to custom format")
        
        # Get categories mapping
        categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        
        # Group annotations by image_id
        image_annotations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            # Convert COCO annotation to our format
            custom_ann = {
                'bbox': ann['bbox'],  # COCO format: [x, y, width, height]
                'class_name': categories.get(ann['category_id'], 'unknown'),
                'category_id': ann['category_id']
            }
            image_annotations[image_id].append(custom_ann)
        
        # Create custom format data
        custom_data = []
        for img in self.annotations['images']:
            img_id = img['id']
            if img_id in image_annotations:
                custom_item = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height'],
                    'id': img_id,
                    'annotations': image_annotations[img_id]
                }
                custom_data.append(custom_item)
        
        logger.info(f"Converted {len(custom_data)} images from COCO format")
        return custom_data

    def _filter_annotations(self):
        """Filter annotations to ONLY include your 8 target landmark classes"""
        # Convert class names to the format they appear in COCO (with spaces)
        target_classes = set([name.replace('_', ' ') for name in self.class_names])
        
        print(f"🎯 FILTERING FOR ONLY THESE CLASSES: {target_classes}")
        
        filtered_data = []
        total_annotations = 0
        kept_annotations = 0
        
        # Ensure we have a list to iterate over
        if not isinstance(self.annotations, list):
            logger.error(f"Expected list, got {type(self.annotations)}")
            return []
        
        for i, item in enumerate(self.annotations):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {i}: {type(item)}")
                continue
                
            # Filter annotations to ONLY your target classes
            landmark_annotations = []
            annotations = item.get('annotations', [])
            
            if not isinstance(annotations, list):
                logger.warning(f"Annotations is not a list at index {i}: {type(annotations)}")
                continue
            
            for ann in annotations:
                total_annotations += 1
                
                if not isinstance(ann, dict):
                    logger.warning(f"Skipping non-dict annotation: {type(ann)}")
                    continue
                    
                class_name = ann.get('class_name', '')
                
                # STRICT FILTERING: Only keep if it's one of your 8 target classes
                if class_name in target_classes:
                    normalized_name = class_name.replace(' ', '_')
                    ann['normalized_class'] = normalized_name
                    landmark_annotations.append(ann)
                    kept_annotations += 1
                # If not in target classes, IGNORE IT COMPLETELY
            
            # Only keep images that have at least one of your target landmark annotations
            if landmark_annotations:
                item['landmark_annotations'] = landmark_annotations
                filtered_data.append(item)
        
        print(f"📊 FILTERING RESULTS:")
        print(f"   📥 Total annotations in dataset: {total_annotations}")
        print(f"   ✅ Kept annotations (your 8 classes): {kept_annotations}")
        print(f"   🗑️  Ignored annotations (other classes): {total_annotations - kept_annotations}")
        print(f"   📸 Images with your target classes: {len(filtered_data)}")
        
        logger.info(f"Filtered {len(filtered_data)} items with target classes")
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
            annotations = item.get('landmark_annotations', [])
        
        for ann in annotations:
            # Get bounding box
            bbox = ann.get('bbox', ann.get('bounding_box', []))
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # 🛡️ VALIDATE BOUNDING BOX: Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue  # Skip boxes with zero/negative width or height
                
                # Convert to [x1, y1, x2, y2] format
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # 🛡️ ENSURE POSITIVE DIMENSIONS after conversion
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid boxes
                
                boxes.append([x1, y1, x2, y2])
                
                # Get class label
                class_name = ann['normalized_class']
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    labels.append(label)
                else:
                    # Skip unknown classes silently
                    continue
        
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
        if original_size != (512, 512):
            # Resize image
            resize_transform = T.Resize((512, 512))
            image = resize_transform(image)
            
            # Scale bounding boxes
            if len(boxes) > 0:
                h_scale = 512 / original_size[0]
                w_scale = 512 / original_size[1]
                
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