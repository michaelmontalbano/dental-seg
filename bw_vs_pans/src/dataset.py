#!/usr/bin/env python3
"""
Dataset class for dental X-ray classification using train.json and val.json
Classes: bitewing, periapical, panoramic
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import logging
import random
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class DentalDataset(Dataset):
    """
    Dataset for dental X-ray classification using JSON annotation files
    Expected structure:
    - train.json and val.json with format:
      {
        "images": [
          {
            "id": <image_id>,
            "file_name": "<filename>.jpg", 
            "width": <width>,
            "height": <height>,
            "xray_type": "<bitewing|panoramic|periapical>"
          },
          ...
        ]
      }
    - Images should be accessible via image_dir/<file_name>
    """
    
    def __init__(self, json_file, image_dir, transform=None, image_size=224, seed=42):
        """
        Args:
            json_file: Path to JSON annotation file (train.json or val.json)
            image_dir: Directory containing the image files
            transform: Image transformations
            image_size: Target image size
            seed: Random seed for reproducibility
        """
        self.json_file = json_file
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Class mapping for 3-way classification
        self.classes = ['bitewing', 'periapical', 'panoramic']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load samples from JSON
        self.samples, self.class_counts = self._load_samples()
        
        logger.info(f"Dataset loaded from {json_file}:")
        for cls, count in self.class_counts.items():
            logger.info(f"  {cls}: {count} samples")
        logger.info(f"Total samples: {len(self.samples)}")
    
    def _load_samples(self):
        """Load samples from JSON annotation file"""
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        if 'images' not in data:
            raise ValueError(f"Invalid JSON format: missing 'images' key in {self.json_file}")
        
        samples = []
        class_counts = Counter()
        
        for image_info in data['images']:
            file_name = image_info.get('file_name')
            xray_type = image_info.get('xray_type')
            
            if not file_name or not xray_type:
                logger.warning(f"Skipping invalid entry: {image_info}")
                continue
            
            if xray_type not in self.class_to_idx:
                logger.warning(f"Unknown xray_type '{xray_type}' for {file_name}")
                continue
            
            # Check if image file exists - try multiple locations
            # First try direct path
            image_path = os.path.join(self.image_dir, file_name)
            
            # If not found, try in xray_type subdirectory
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, xray_type, file_name)
            
            # If still not found, skip this image
            if not os.path.exists(image_path):
                logger.debug(f"Image file not found: {file_name} (tried both direct and {xray_type}/ subdirectory)")
                continue
            
            class_idx = self.class_to_idx[xray_type]
            samples.append((image_path, class_idx))
            class_counts[xray_type] += 1
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.json_file}")
        
        return samples, class_counts
    
    def _load_image(self, image_path):
        """Load image with error handling"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Validate image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                raise ValueError(f"Image too small: {image.size}")
                
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Handle corrupted images by returning a different random sample
        if image is None:
            new_idx = np.random.randint(0, len(self.samples))
            return self.__getitem__(new_idx)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(phase, image_size=224, center_crop_corners=True):
    """Get transforms for training and validation"""
    
    if phase == 'train':
        transforms_list = []
        
        # Center crop to remove corner artifacts (including text labels)
        if center_crop_corners:
            transforms_list.append(T.Lambda(lambda img: center_crop_to_remove_corners(img, crop_ratio=0.75)))
        
        transforms_list.extend([
            T.Resize((image_size, image_size)),
            # Conservative augmentation for medical images
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=8, fill=0),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms_list)
        
    else:  # val/test
        transforms_list = []
        
        if center_crop_corners:
            transforms_list.append(T.Lambda(lambda img: center_crop_to_remove_corners(img, crop_ratio=0.75)))
        
        transforms_list.extend([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms_list)

def center_crop_to_remove_corners(image, crop_ratio=0.75):
    """
    Center crop image to remove 25% from all sides (keep center 75%)
    This removes corner artifacts including text labels, equipment markers, etc.
    
    Args:
        image: PIL Image
        crop_ratio: Ratio of image to keep (0.75 = keep 75%, crop 25% from each side)
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate crop dimensions (center 75% of image)
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    # Calculate crop box (centered)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop to center region
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def create_dataloaders(train_json, val_json, image_dir, batch_size=32, 
                      image_size=224, num_workers=4, center_crop_corners=True):
    """
    Create data loaders from JSON annotation files
    
    Args:
        train_json: Path to train.json file
        val_json: Path to val.json file
        image_dir: Directory containing image files
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        center_crop_corners: Center crop to remove corner artifacts
    """
    dataloaders = {}
    datasets = {}
    
    logger.info("Creating dataloaders from JSON files...")
    logger.info(f"Train JSON: {train_json}")
    logger.info(f"Val JSON: {val_json}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Center crop corners: {center_crop_corners}")
    
    try:
        # Create training dataset
        if os.path.exists(train_json):
            train_dataset = DentalDataset(
                json_file=train_json,
                image_dir=image_dir,
                transform=get_transforms('train', image_size, center_crop_corners),
                image_size=image_size
            )
            datasets['train'] = train_dataset
            
            dataloaders['train'] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            logger.info(f"✅ Training dataset: {len(train_dataset)} samples")
        else:
            logger.warning(f"Training JSON file not found: {train_json}")
        
        # Create validation dataset
        if os.path.exists(val_json):
            val_dataset = DentalDataset(
                json_file=val_json,
                image_dir=image_dir,
                transform=get_transforms('val', image_size, center_crop_corners),
                image_size=image_size
            )
            datasets['val'] = val_dataset
            
            dataloaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            logger.info(f"✅ Validation dataset: {len(val_dataset)} samples")
        else:
            logger.warning(f"Validation JSON file not found: {val_json}")
        
        if center_crop_corners:
            logger.info("✂️ Center cropping enabled - removes 25% from all sides (keeps center 75%)")
            logger.info("   • Removes equipment markers, text labels, corner artifacts")
        
        return datasets, dataloaders
        
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        return {}, {}

def analyze_dataset_distribution(json_file):
    """Analyze class distribution in a JSON file"""
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'images' not in data:
        print(f"❌ Invalid JSON format in {json_file}")
        return
    
    class_counts = Counter()
    total_images = len(data['images'])
    
    for image_info in data['images']:
        xray_type = image_info.get('xray_type', 'unknown')
        class_counts[xray_type] += 1
    
    print(f"📊 Dataset: {os.path.basename(json_file)}")
    print(f"   Total images: {total_images}")
    for class_name, count in class_counts.most_common():
        percentage = (count / total_images) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    print()

if __name__ == "__main__":
    # Analyze dataset distributions
    train_json = "../model/train.json"  # Adjust path as needed
    val_json = "../model/val.json"      # Adjust path as needed
    
    print("🔍 Analyzing dataset distributions...")
    analyze_dataset_distribution(train_json)
    analyze_dataset_distribution(val_json)
