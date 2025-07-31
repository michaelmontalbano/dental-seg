#!/usr/bin/env python3
"""
Convert S3 bw_pa_pans dataset to train.json and val.json format for three-class classification
Source: s3://codentist-general/datasets/bw_pa_pans/
- bitewing/ (folder with images - only JPG files)
- panoramic/ (folder with images) 
- periapical/ (folder with images)

This script will:
1. List all images in the three S3 folders
2. Apply filename filtering rules (bitewing: only JPG, others: all image formats)
3. Create stratified train/val split for three classes
4. Generate train.json and val.json files

Key differences from create_train_val_split.py:
- Three classes instead of two
- Bitewing images must end in JPG (not PNG)
- Don't exclude images with duplicate xray type in filename
"""

import json
import boto3
import os
import argparse
from collections import Counter, defaultdict
import random
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_s3_objects(bucket, prefix):
    """List objects in S3 bucket with prefix"""
    s3 = boto3.client('s3')
    objects = []
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append(obj['Key'])
        
        logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
        return objects
    except Exception as e:
        logger.error(f"Failed to list s3://{bucket}/{prefix}: {e}")
        return []

def should_include_file(filename, xray_type):
    """
    Apply filename filtering rules to determine if file should be included
    
    Rules for three-class bw_pa_pans dataset:
    - For bitewing: Only keep files ending in .JPG/.JPEG (exclude .png files)
    - For periapical and panoramic: Keep all standard image formats
    - Don't exclude files where xray_type appears twice in filename (changed from original)
    """
    filename_lower = filename.lower()
    
    # Exclude non-image files
    if not any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
        return False, "Not an image file"
    
    # Exclude hidden/system files
    if filename_lower.startswith('._'):
        return False, "Hidden/system file"
    
    # Special rule for bitewing images: only keep .JPG/.JPEG files
    if xray_type.lower() == 'bitewing':
        if not (filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg')):
            return False, f"Bitewing image must end in .JPG (found: {filename})"
    
    # NOTE: Removed the duplicate xray_type check as per requirements
    # Files like "bitewing_1231_bitewing.jpg" should now be included
    
    # Include by default if all filters pass
    return True, "Passed all filters"

def create_image_inventory(bucket, base_prefix):
    """Create inventory of all images in the S3 folders"""
    inventory = {}
    
    for xray_type in ['bitewing', 'panoramic', 'periapical']:
        prefix = f"{base_prefix}/{xray_type}/"
        objects = list_s3_objects(bucket, prefix)
        
        # Filter to only image files and apply filename rules
        valid_images = []
        excluded_count = 0
        exclusion_reasons = Counter()
        
        for obj_key in objects:
            filename = os.path.basename(obj_key)
            should_include, reason = should_include_file(filename, xray_type)
            
            if should_include:
                valid_images.append({
                    'key': obj_key,
                    'filename': filename,
                    'xray_type': xray_type
                })
            else:
                excluded_count += 1
                exclusion_reasons[reason] += 1
        
        inventory[xray_type] = valid_images
        
        logger.info(f"{xray_type}: {len(valid_images)} valid images, {excluded_count} excluded")
        if exclusion_reasons:
            logger.info(f"  Exclusion reasons: {dict(exclusion_reasons)}")
    
    return inventory

def create_train_val_split(inventory, val_ratio=0.2, random_seed=42):
    """Create stratified train/validation split for three classes"""
    random.seed(random_seed)
    
    train_images = []
    val_images = []
    
    # Process each class separately to ensure stratification
    for xray_type, images in inventory.items():
        if not images:
            logger.warning(f"No images found for {xray_type}")
            continue
        
        # Shuffle images
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        
        # Split
        n_val = int(len(shuffled_images) * val_ratio)
        n_train = len(shuffled_images) - n_val
        
        train_split = shuffled_images[:n_train]
        val_split = shuffled_images[n_train:]
        
        train_images.extend(train_split)
        val_images.extend(val_split)
        
        logger.info(f"{xray_type}: {n_train} train, {n_val} val")
    
    # Shuffle the final splits
    random.shuffle(train_images)
    random.shuffle(val_images)
    
    logger.info(f"Total: {len(train_images)} train, {len(val_images)} val")
    
    return train_images, val_images

def create_json_annotation(images, output_path):
    """Create JSON annotation file in the expected format for three-class classification"""
    
    # Create class label mapping
    class_to_label = {
        'bitewing': 0,
        'periapical': 1,
        'panoramic': 2
    }
    
    # Convert to the expected format
    json_data = {
        "images": [],
        "class_mapping": class_to_label,
        "num_classes": 3
    }
    
    for i, img_info in enumerate(images):
        # Generate a unique ID
        image_id = i + 1
        
        json_data["images"].append({
            "id": image_id,
            "file_name": img_info['filename'],
            "width": 0,  # Will be filled when images are processed
            "height": 0,  # Will be filled when images are processed
            "xray_type": img_info['xray_type'],
            "label": class_to_label[img_info['xray_type']],  # Numeric label for training
            "s3_key": img_info['key']  # Keep S3 reference for downloading
        })
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Created {output_path} with {len(images)} images")
    
    # Print class distribution
    class_counts = Counter(img['xray_type'] for img in images)
    logger.info(f"Class distribution in {os.path.basename(output_path)}:")
    for class_name, count in class_counts.items():
        percentage = (count / len(images)) * 100
        label = class_to_label[class_name]
        logger.info(f"  {class_name} (label {label}): {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Create train/val split from bw_pa_pans dataset for three-class classification')
    
    parser.add_argument('--bucket', type=str, default='codentist-general',
                       help='S3 bucket name')
    parser.add_argument('--dataset-prefix', type=str, default='datasets/bw_pa_pans',
                       help='S3 prefix for dataset')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for train.json and val.json')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (0.2 = 20%)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create inventory of images from S3
    logger.info("Creating image inventory from S3...")
    inventory = create_image_inventory(args.bucket, args.dataset_prefix)
    
    # Print overall statistics
    total_images = sum(len(images) for images in inventory.values())
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total valid images: {total_images}")
    for xray_type, images in inventory.items():
        percentage = (len(images) / total_images) * 100 if total_images > 0 else 0
        logger.info(f"  {xray_type}: {len(images)} ({percentage:.1f}%)")
    
    if total_images == 0:
        logger.error("No valid images found!")
        return
    
    # Check for class imbalance
    class_counts = {xray_type: len(images) for xray_type, images in inventory.items()}
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    logger.info(f"\nClass balance analysis:")
    logger.info(f"  Min class size: {min_count}")
    logger.info(f"  Max class size: {max_count}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 3:
        logger.warning("Significant class imbalance detected! Consider using weighted loss or resampling.")
    
    # Create train/val split
    logger.info(f"\nCreating train/val split (val_ratio={args.val_ratio})...")
    train_images, val_images = create_train_val_split(
        inventory, 
        val_ratio=args.val_ratio, 
        random_seed=args.random_seed
    )
    
    # Create JSON files
    train_path = os.path.join(args.output_dir, 'train.json')
    val_path = os.path.join(args.output_dir, 'val.json')
    
    create_json_annotation(train_images, train_path)
    create_json_annotation(val_images, val_path)
    
    # Create summary
    summary = {
        'dataset_info': {
            'source_bucket': args.bucket,
            'source_prefix': args.dataset_prefix,
            'total_images': total_images,
            'num_classes': 3,
            'classes': ['bitewing', 'periapical', 'panoramic'],
            'class_labels': {'bitewing': 0, 'periapical': 1, 'panoramic': 2},
            'val_ratio': args.val_ratio,
            'random_seed': args.random_seed
        },
        'class_distribution': {
            xray_type: len(images) for xray_type, images in inventory.items()
        },
        'class_balance': {
            'min_class_size': min_count,
            'max_class_size': max_count,
            'imbalance_ratio': imbalance_ratio
        },
        'split_info': {
            'train_images': len(train_images),
            'val_images': len(val_images),
            'train_ratio': len(train_images) / total_images,
            'val_ratio': len(val_images) / total_images
        },
        'filtering_rules': {
            'bitewing': 'Only .JPG/.JPEG files accepted',
            'periapical': 'All image formats accepted',
            'panoramic': 'All image formats accepted',
            'duplicate_names': 'Files with duplicate xray type in name are INCLUDED'
        },
        'files_created': [
            train_path,
            val_path
        ]
    }
    
    summary_path = os.path.join(args.output_dir, 'three_class_split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nThree-class split completed successfully!")
    logger.info(f"Files created:")
    logger.info(f"  - {train_path}")
    logger.info(f"  - {val_path}")
    logger.info(f"  - {summary_path}")
    
    logger.info(f"\nKey changes from original script:")
    logger.info(f"  - Three classes: bitewing (0), periapical (1), panoramic (2)")
    logger.info(f"  - Bitewing: Only JPG files included")
    logger.info(f"  - Files with duplicate xray type names are now INCLUDED")
    logger.info(f"  - Added class imbalance analysis")
    
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review the generated files and class distribution")
    logger.info(f"2. Consider data augmentation if class imbalance is significant")
    logger.info(f"3. Update your training code to handle 3 classes")
    logger.info(f"4. Consider the architectural recommendations in the task description")

if __name__ == '__main__':
    main()
