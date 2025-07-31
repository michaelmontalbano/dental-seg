#!/usr/bin/env python3
"""
Create panoramic vs non-panoramic train/val split from existing S3 datasets
Source: s3://codentist-general/datasets/
- panoramic_dataset/annotations/ -> train.json and val.json (panoramic images)
- merged_bw_pa/annotations/ -> train.json and val.json (non-panoramic images)

This script will:
1. Download existing train.json and val.json from both datasets
2. Combine them into unified binary classification splits
3. Map panoramic -> 'panoramic', others -> 'non_panoramic'
4. Generate new train.json and val.json files for binary classification
"""

import json
import boto3
import os
import argparse
from collections import Counter
import random
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

def should_include_file(filename, dataset_type):
    """
    Apply filename filtering rules to determine if file should be included
    
    Args:
        filename: Name of the image file
        dataset_type: 'panoramic' or 'non_panoramic'
    """
    filename_lower = filename.lower()
    
    # Exclude non-image files
    if not any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
        return False, "Not an image file"
    
    # Exclude hidden/system files
    if filename_lower.startswith('._'):
        return False, "Hidden/system file"
    
    # Include by default if all filters pass
    return True, "Passed all filters"

def load_dataset_splits_from_s3(bucket, dataset_prefix, dataset_type, binary_class):
    """Load train.json and val.json from S3 dataset and convert to binary classification"""
    
    train_key = f"{dataset_prefix}/annotations/train.json"
    val_key = f"{dataset_prefix}/annotations/val.json"
    
    local_train = f"./temp_train_{dataset_type}.json"
    local_val = f"./temp_val_{dataset_type}.json"
    
    # Download train.json
    if not download_s3_file(bucket, train_key, local_train):
        logger.error(f"Failed to download train.json for {dataset_type}")
        return [], []
    
    # Download val.json
    if not download_s3_file(bucket, val_key, local_val):
        logger.error(f"Failed to download val.json for {dataset_type}")
        return [], []
    
    # Process train split
    train_images = process_split_file(local_train, dataset_prefix, dataset_type, binary_class, 'train')
    
    # Process val split
    val_images = process_split_file(local_val, dataset_prefix, dataset_type, binary_class, 'val')
    
    # Clean up temp files
    for temp_file in [local_train, local_val]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return train_images, val_images

def process_split_file(json_file, dataset_prefix, dataset_type, binary_class, split_name):
    """Process a single train.json or val.json file"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'images' not in data:
        logger.error(f"Invalid JSON format in {json_file}")
        return []
    
    processed_images = []
    excluded_count = 0
    exclusion_reasons = Counter()
    
    for image_info in data['images']:
        filename = image_info.get('file_name', '')
        if not filename:
            continue
            
        should_include, reason = should_include_file(filename, dataset_type)
        
        if should_include:
            # Create image entry with binary classification
            image_entry = {
                'key': f"{dataset_prefix}/images/{filename}",
                'filename': filename,
                'xray_type': binary_class,  # 'panoramic' or 'non_panoramic'
                'original_type': image_info.get('xray_type', 'unknown'),  # Keep original for reference
                'width': image_info.get('width', 0),
                'height': image_info.get('height', 0),
                'id': image_info.get('id', 0),
                'source_dataset': dataset_type,
                'source_split': split_name
            }
            processed_images.append(image_entry)
        else:
            excluded_count += 1
            exclusion_reasons[reason] += 1
    
    logger.info(f"{dataset_type} {split_name}: {len(processed_images)} valid images, {excluded_count} excluded")
    if exclusion_reasons:
        logger.info(f"  Exclusion reasons: {dict(exclusion_reasons)}")
    
    return processed_images

def combine_and_resplit(panoramic_train, panoramic_val, non_panoramic_train, non_panoramic_val, 
                       val_ratio=0.2, random_seed=42):
    """Combine datasets and create new stratified train/val split"""
    
    random.seed(random_seed)
    
    # Combine all panoramic images
    all_panoramic = panoramic_train + panoramic_val
    logger.info(f"Total panoramic images: {len(all_panoramic)}")
    
    # Combine all non-panoramic images
    all_non_panoramic = non_panoramic_train + non_panoramic_val
    logger.info(f"Total non-panoramic images: {len(all_non_panoramic)}")
    
    # Shuffle each class
    random.shuffle(all_panoramic)
    random.shuffle(all_non_panoramic)
    
    # Split panoramic images
    pano_val_count = int(len(all_panoramic) * val_ratio)
    pano_train_count = len(all_panoramic) - pano_val_count
    
    pano_train_split = all_panoramic[:pano_train_count]
    pano_val_split = all_panoramic[pano_train_count:]
    
    # Split non-panoramic images
    non_pano_val_count = int(len(all_non_panoramic) * val_ratio)
    non_pano_train_count = len(all_non_panoramic) - non_pano_val_count
    
    non_pano_train_split = all_non_panoramic[:non_pano_train_count]
    non_pano_val_split = all_non_panoramic[non_pano_train_count:]
    
    # Combine final splits
    final_train = pano_train_split + non_pano_train_split
    final_val = pano_val_split + non_pano_val_split
    
    # Shuffle final splits
    random.shuffle(final_train)
    random.shuffle(final_val)
    
    logger.info(f"Final split:")
    logger.info(f"  Panoramic: {len(pano_train_split)} train, {len(pano_val_split)} val")
    logger.info(f"  Non-panoramic: {len(non_pano_train_split)} train, {len(non_pano_val_split)} val")
    logger.info(f"  Total: {len(final_train)} train, {len(final_val)} val")
    
    return final_train, final_val

def create_json_annotation(images, output_path):
    """Create JSON annotation file in the expected format"""
    
    # Convert to the expected format
    json_data = {
        "images": []
    }
    
    for i, img_info in enumerate(images):
        # Generate a unique ID
        image_id = img_info.get('id', i + 1)
        
        json_data["images"].append({
            "id": image_id,
            "file_name": img_info['filename'],
            "width": img_info.get('width', 0),
            "height": img_info.get('height', 0),
            "xray_type": img_info['xray_type'],  # 'panoramic' or 'non_panoramic'
            "original_type": img_info.get('original_type', 'unknown'),  # Original classification
            "s3_key": img_info['key'],  # Keep S3 reference for downloading
            "source_dataset": img_info.get('source_dataset', 'unknown'),
            "source_split": img_info.get('source_split', 'unknown')
        })
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Created {output_path} with {len(images)} images")
    
    # Print class distribution
    class_counts = Counter(img['xray_type'] for img in images)
    logger.info(f"Binary class distribution in {os.path.basename(output_path)}:")
    for class_name, count in class_counts.items():
        percentage = (count / len(images)) * 100
        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Print original type distribution for reference
    original_counts = Counter(img.get('original_type', 'unknown') for img in images)
    logger.info(f"Original type distribution in {os.path.basename(output_path)}:")
    for orig_type, count in original_counts.items():
        percentage = (count / len(images)) * 100
        logger.info(f"  {orig_type}: {count} ({percentage:.1f}%)")
    
    # Print source dataset distribution
    source_counts = Counter(img.get('source_dataset', 'unknown') for img in images)
    logger.info(f"Source dataset distribution in {os.path.basename(output_path)}:")
    for source, count in source_counts.items():
        percentage = (count / len(images)) * 100
        logger.info(f"  {source}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Create binary panoramic vs non-panoramic train/val split from existing S3 datasets')
    
    parser.add_argument('--bucket', type=str, default='codentist-general',
                       help='S3 bucket name')
    parser.add_argument('--panoramic-prefix', type=str, default='datasets/panoramic_dataset',
                       help='S3 prefix for panoramic dataset')
    parser.add_argument('--merged-bw-pa-prefix', type=str, default='datasets/merged_bw_pa',
                       help='S3 prefix for merged bitewing/periapical dataset')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for train.json and val.json')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (0.2 = 20 percent)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading existing train/val splits from S3 datasets...")
    logger.info(f"Panoramic dataset: s3://{args.bucket}/{args.panoramic_prefix}")
    logger.info(f"Merged BW/PA dataset: s3://{args.bucket}/{args.merged_bw_pa_prefix}")
    
    # Load panoramic dataset splits
    logger.info("Loading panoramic dataset...")
    pano_train, pano_val = load_dataset_splits_from_s3(
        args.bucket, args.panoramic_prefix, 'panoramic', 'panoramic'
    )
    
    # Load merged_bw_pa dataset splits
    logger.info("Loading merged_bw_pa dataset...")
    non_pano_train, non_pano_val = load_dataset_splits_from_s3(
        args.bucket, args.merged_bw_pa_prefix, 'non_panoramic', 'non_panoramic'
    )
    
    # Print dataset statistics
    total_images = len(pano_train) + len(pano_val) + len(non_pano_train) + len(non_pano_val)
    logger.info(f"\nOriginal Dataset Statistics:")
    logger.info(f"Panoramic: {len(pano_train)} train, {len(pano_val)} val (total: {len(pano_train) + len(pano_val)})")
    logger.info(f"Non-panoramic: {len(non_pano_train)} train, {len(non_pano_val)} val (total: {len(non_pano_train) + len(non_pano_val)})")
    logger.info(f"Total images: {total_images}")
    
    if total_images == 0:
        logger.error("No valid images found!")
        return
    
    # Combine and create new stratified split
    logger.info(f"\nCreating new stratified binary split (val_ratio={args.val_ratio})...")
    final_train, final_val = combine_and_resplit(
        pano_train, pano_val, non_pano_train, non_pano_val,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    
    # Create JSON files
    train_path = os.path.join(args.output_dir, 'train.json')
    val_path = os.path.join(args.output_dir, 'val.json')
    
    create_json_annotation(final_train, train_path)
    create_json_annotation(final_val, val_path)
    
    # Create summary
    summary = {
        'dataset_info': {
            'source_bucket': args.bucket,
            'panoramic_prefix': args.panoramic_prefix,
            'merged_bw_pa_prefix': args.merged_bw_pa_prefix,
            'total_images': total_images,
            'val_ratio': args.val_ratio,
            'random_seed': args.random_seed,
            'classification_type': 'binary',
            'classes': ['panoramic', 'non_panoramic']
        },
        'original_splits': {
            'panoramic_train': len(pano_train),
            'panoramic_val': len(pano_val),
            'non_panoramic_train': len(non_pano_train),
            'non_panoramic_val': len(non_pano_val)
        },
        'final_splits': {
            'train_images': len(final_train),
            'val_images': len(final_val),
            'train_ratio': len(final_train) / total_images,
            'val_ratio': len(final_val) / total_images
        },
        'class_distribution': {
            'train': dict(Counter(img['xray_type'] for img in final_train)),
            'val': dict(Counter(img['xray_type'] for img in final_val))
        },
        'files_created': [
            train_path,
            val_path
        ]
    }
    
    summary_path = os.path.join(args.output_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nBinary split completed successfully!")
    logger.info(f"Files created:")
    logger.info(f"  - {train_path}")
    logger.info(f"  - {val_path}")
    logger.info(f"  - {summary_path}")
    
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review the generated files")
    logger.info(f"2. Upload train.json and val.json to your training location")
    logger.info(f"3. Use the panoramic vs non-panoramic training system")
    
    # Print final class balance
    train_class_counts = Counter(img['xray_type'] for img in final_train)
    val_class_counts = Counter(img['xray_type'] for img in final_val)
    
    logger.info(f"\nFinal binary class balance:")
    logger.info(f"Training set:")
    for class_name, count in train_class_counts.items():
        percentage = (count / len(final_train)) * 100
        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    logger.info(f"Validation set:")
    for class_name, count in val_class_counts.items():
        percentage = (count / len(final_val)) * 100
        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
