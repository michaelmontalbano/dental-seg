#!/usr/bin/env python3
"""
Convert S3 bw_pa_pans dataset to train.json and val.json format
Source: s3://codentist-general/datasets/bw_pa_pans/
- annotations.json
- bitewing/ (folder with images)
- panoramic/ (folder with images) 
- periapical/ (folder with images)

This script will:
1. Download and examine annotations.json
2. Apply filename filtering rules
3. Create stratified train/val split
4. Generate train.json and val.json files
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

def download_s3_file(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
        return False

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
    
    Specific rules for bw_pa_pans dataset:
    - For bitewing: Only keep files ending in .JPG (exclude .png files)
    - For periapical and panoramic: Keep all standard image formats
    """
    filename_lower = filename.lower()
    
    # Exclude non-image files
    if not any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
        return False, "Not an image file"
    
    # Exclude hidden/system files
    if filename_lower.startswith('._'):
        return False, "Hidden/system file"
    
    # Special rule for bitewing images: only keep .JPG files
    if xray_type.lower() == 'bitewing':
        if not (filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg')):
            return False, f"Bitewing image must end in .JPG (found: {filename})"
    
    # For all types: exclude files where xray_type appears twice in filename
    # e.g., bitewing_0001_bitewing_2007.png should be excluded
    # but bitewing_0001_X12255_9.JPG should be kept
    xray_type_lower = xray_type.lower()
    
    # Count occurrences of xray_type in filename
    xray_type_count = filename_lower.count(xray_type_lower)
    
    if xray_type_count > 1:
        return False, f"Xray type '{xray_type}' appears {xray_type_count} times in filename (likely duplicate/test file)"
    
    # Include by default if all filters pass
    return True, "Passed all filters"

def analyze_annotations(annotations_path):
    """Analyze the annotations.json file structure"""
    if not os.path.exists(annotations_path):
        logger.error(f"Annotations file not found: {annotations_path}")
        return None
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    logger.info("Annotations file structure:")
    logger.info(f"Keys: {list(data.keys())}")
    
    if 'images' in data:
        logger.info(f"Number of images: {len(data['images'])}")
        if data['images']:
            logger.info(f"Sample image entry: {data['images'][0]}")
    
    return data

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
    """Create stratified train/validation split"""
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
    """Create JSON annotation file in the expected format"""
    
    # Convert to the expected format
    json_data = {
        "images": []
    }
    
    for i, img_info in enumerate(images):
        # Generate a unique ID (you might want to use actual image IDs if available)
        image_id = i + 1
        
        json_data["images"].append({
            "id": image_id,
            "file_name": img_info['filename'],
            "width": 0,  # Will be filled when images are processed
            "height": 0,  # Will be filled when images are processed
            "xray_type": img_info['xray_type'],
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
        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Create train/val split from bw_pa_pans dataset')
    
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
    parser.add_argument('--download-annotations', action='store_true',
                       help='Download annotations.json from S3')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download annotations.json if requested
    annotations_path = os.path.join(args.output_dir, 'annotations.json')
    if args.download_annotations:
        annotations_key = f"{args.dataset_prefix}/annotations.json"
        if not download_s3_file(args.bucket, annotations_key, annotations_path):
            logger.error("Failed to download annotations.json")
            return
    
    # Analyze annotations (if available)
    if os.path.exists(annotations_path):
        logger.info("Analyzing existing annotations.json...")
        annotations_data = analyze_annotations(annotations_path)
    else:
        logger.info("No annotations.json found, proceeding with S3 inventory...")
        annotations_data = None
    
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
            'val_ratio': args.val_ratio,
            'random_seed': args.random_seed
        },
        'class_distribution': {
            xray_type: len(images) for xray_type, images in inventory.items()
        },
        'split_info': {
            'train_images': len(train_images),
            'val_images': len(val_images),
            'train_ratio': len(train_images) / total_images,
            'val_ratio': len(val_images) / total_images
        },
        'files_created': [
            train_path,
            val_path
        ]
    }
    
    summary_path = os.path.join(args.output_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSplit completed successfully!")
    logger.info(f"Files created:")
    logger.info(f"  - {train_path}")
    logger.info(f"  - {val_path}")
    logger.info(f"  - {summary_path}")
    
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review the generated files")
    logger.info(f"2. Update image dimensions if needed")
    logger.info(f"3. Upload train.json and val.json to your training location")
    logger.info(f"4. Use the updated BW_VS_PANS training system")

if __name__ == '__main__':
    main()
