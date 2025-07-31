# PANORAMIC_VS_NON_PANORAMIC - Binary Dental X-ray Classification

This directory contains a binary classification system for dental X-rays that distinguishes between **panoramic** and **non-panoramic** (bitewing + periapical) images. It's based on the `bw_vs_pans` system but adapted for binary classification using data from two S3 datasets.

## Overview

The system classifies dental X-rays into two categories:
- **Panoramic**: Wide-angle X-rays showing the entire mouth (from `panoramic_dataset`)
- **Non-panoramic**: Bitewing and periapical X-rays combined (from `merged_bw_pa`)

## Key Features

### 1. Binary Classification Architecture
- **Classes**: `panoramic` vs `non_panoramic` (2 classes instead of 3)
- **Data Sources**: Two separate S3 datasets with their own annotations
- **Balanced Approach**: Stratified train/val splits to ensure class balance

### 2. Dual Dataset Integration
- **Panoramic Dataset**: `s3://codentist-general/datasets/panoramic_dataset/`
- **Merged BW/PA Dataset**: `s3://codentist-general/datasets/merged_bw_pa/`
- **Unified Processing**: Combines both datasets into single train/val splits

### 3. Enhanced Training Features
- Comprehensive overfitting detection (adjusted for binary classification)
- Progress bars with real-time metrics
- Detailed per-class evaluation every 5 epochs
- Early stopping with configurable patience
- Center cropping to remove corner artifacts and text labels

## File Structure

```
panoramic_vs_non_panoramic/
├── src/
│   ├── dataset.py          # Binary classification dataset class
│   ├── train.py           # Training script with overfitting detection
│   └── requirements.txt   # Python dependencies
├── sagemaker_launcher.py  # SageMaker training launcher
├── create_train_val_split.py  # Dataset preparation script
├── data/                  # Generated train/val splits
├── model/                 # Saved models and metrics
└── README.md             # This file
```

## Data Format

### Source Data Structure
```
s3://codentist-general/datasets/
├── panoramic_dataset/
│   ├── images/           # Panoramic X-ray images
│   └── annotations/
│       ├── train.json    # Panoramic training annotations
│       └── val.json      # Panoramic validation annotations
└── merged_bw_pa/
    ├── images/           # Bitewing + periapical images
    └── annotations/
        ├── train.json    # BW/PA training annotations
        └── val.json      # BW/PA validation annotations
```

### Generated JSON Format
Both `train.json` and `val.json` follow this structure:

```json
{
  "images": [
    {
      "id": 404437371,
      "file_name": "404437371.jpg",
      "width": 1692,
      "height": 1324,
      "xray_type": "panoramic",
      "original_type": "panoramic",
      "s3_key": "datasets/panoramic_dataset/images/404437371.jpg"
    },
    {
      "id": 373067253,
      "file_name": "373067253.jpg", 
      "width": 844,
      "height": 660,
      "xray_type": "non_panoramic",
      "original_type": "bitewing",
      "s3_key": "datasets/merged_bw_pa/images/373067253.jpg"
    }
  ]
}
```

## Setup

### Step 1: Create Train/Val Split from Source Data

Convert the existing train/val splits from both S3 datasets into unified binary classification files:

```bash
cd cej-ac-detection/panoramic_vs_non_panoramic

# Create binary train/val split from existing S3 dataset splits
python create_train_val_split.py \
    --bucket codentist-general \
    --panoramic-prefix datasets/panoramic_dataset \
    --merged-bw-pa-prefix datasets/merged_bw_pa \
    --output-dir ./data \
    --val-ratio 0.2

# This will create:
# - ./data/train.json
# - ./data/val.json  
# - ./data/split_summary.json
```

**Key Features of the Split Script:**
- **Existing Split Processing**: Downloads existing train.json and val.json from both datasets
- **Binary Classification**: Maps panoramic → `panoramic`, others → `non_panoramic`
- **Stratified Re-splitting**: Combines and re-splits to ensure balanced class distribution
- **S3 Integration**: Directly downloads from S3 annotations/ directories
- **Reproducible**: Uses fixed random seed for consistent splits

### Step 2: Upload Generated Files to S3

```bash
# Upload the generated JSON files to S3 for SageMaker training
aws s3 cp ./data/train.json s3://codentist-general/datasets/panoramic-vs-non-panoramic/train.json
aws s3 cp ./data/val.json s3://codentist-general/datasets/panoramic-vs-non-panoramic/val.json
```

## Usage

### Local Training

**Note**: The training script now has hardcoded S3 paths for the JSON files:
- Train JSON: `s3://codentist-general/datasets/panoramic_vs_non_panoramic/train.json`
- Val JSON: `s3://codentist-general/datasets/panoramic_vs_non_panoramic/val.json`

```bash
cd cej-ac-detection/panoramic_vs_non_panoramic/src

# Basic training (uses hardcoded S3 paths for JSON files)
python train.py --image_dir /path/to/images

# Custom configuration (JSON paths are hardcoded, but can be overridden)
python train.py \
    --image_dir /path/to/images \
    --train_json /path/to/custom/train.json \
    --val_json /path/to/custom/val.json \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --model_name efficientnet_b0
```

### SageMaker Training

**Note**: The SageMaker launcher now has hardcoded S3 paths as defaults:
- Train JSON: `s3://codentist-general/datasets/panoramic_vs_non_panoramic/train.json`
- Val JSON: `s3://codentist-general/datasets/panoramic_vs_non_panoramic/val.json`
- Image directory: `s3://codentist-general/datasets/`

```bash
cd cej-ac-detection/panoramic_vs_non_panoramic

# Launch SageMaker training job with hardcoded defaults (simplest usage)
python sagemaker_launcher.py

# With custom job name and parameters
python sagemaker_launcher.py \
    --job-name my-panoramic-experiment-001 \
    --instance-type ml.g4dn.xlarge \
    --epochs 30 \
    --batch-size 16

# Override paths if needed
python sagemaker_launcher.py \
    --job-name custom-data-experiment \
    --train-json s3://your-bucket/custom/train.json \
    --val-json s3://your-bucket/custom/val.json \
    --image-dir s3://your-bucket/images/ \
    --instance-type ml.g4dn.xlarge \
    --epochs 30
```

## Key Differences from BW_VS_PANS

### 1. Binary vs Ternary Classification
- **Classes**: 2 instead of 3 (`panoramic` vs `non_panoramic`)
- **Thresholds**: Adjusted overfitting detection for binary classification
- **Metrics**: Updated evaluation criteria for 2-class problem

### 2. Dual Dataset Architecture
- **Data Sources**: Two separate S3 datasets instead of one
- **Processing**: Unified processing of heterogeneous data sources
- **Mapping**: Automatic classification mapping during data preparation

### 3. Enhanced Data Integration
- **Original Type Tracking**: Preserves original classification for analysis
- **S3 Key Preservation**: Maintains S3 references for data lineage
- **Flexible Image Loading**: Handles multiple directory structures

## Model Architecture

### Supported Models
- **ResNet-50** (default): Good balance of accuracy and speed
- **EfficientNet-B0**: Efficient architecture for mobile deployment
- **MobileNet-V3-Large**: Lightweight for resource-constrained environments

### Binary Classification Adjustments
- **Output Layer**: 2 neurons instead of 3
- **Class Weights**: Calculated for panoramic vs non-panoramic balance
- **Evaluation**: Binary-specific metrics and thresholds

## Expected Performance

### Realistic Expectations (Binary Classification)
- **Validation Accuracy**: 60-85% (task-dependent)
- **Random Baseline**: 50% (2 classes)
- **Warning Thresholds**: >85% may indicate data leakage or overly easy task

### Success Criteria
- Balanced per-class performance
- Minimal overfitting warnings
- Stable training convergence
- Accuracy gap <15% between train/val

## Monitoring and Debugging

### Training Logs
- Real-time progress bars
- Epoch-by-epoch metrics
- Overfitting warnings (adjusted for binary classification)
- Class distribution analysis

### Output Files
- `model.pth`: Trained model with metadata
- `training_metrics.json`: Detailed training history
- Overfitting detection summaries

### Common Issues
- **High accuracy (>85%)**: Check for data leakage or task simplicity
- **Large train/val gap**: Increase regularization
- **Missing images**: Verify image paths in JSON files
- **Memory errors**: Reduce batch size

## Data Preparation Details

### Dataset Combination Strategy
1. **Panoramic Dataset**: All images labeled as `panoramic`
2. **Merged BW/PA Dataset**: All images labeled as `non_panoramic`
3. **Original Type Preservation**: Maintains bitewing/periapical distinction for analysis
4. **Balanced Sampling**: Stratified splits ensure class balance

### Filtering Rules
- Standard image format validation (jpg, jpeg, png, tiff)
- Hidden file exclusion (files starting with `._`)
- Less restrictive than 3-class system (working with pre-processed datasets)

## Migration from BW_VS_PANS

### Key Changes
1. **Classification Type**: Binary instead of ternary
2. **Data Sources**: Two datasets instead of one
3. **Class Mapping**: Automatic panoramic vs non-panoramic assignment
4. **Thresholds**: Adjusted overfitting detection for binary classification

### Migration Steps
1. Use new `create_train_val_split.py` for dual dataset processing
2. Update class expectations (2 classes instead of 3)
3. Adjust evaluation thresholds for binary classification
4. Use new SageMaker launcher configuration

## Performance Optimization

### Binary Classification Advantages
- **Simpler Decision Boundary**: Easier to learn than 3-class problem
- **Better Class Balance**: More balanced than bitewing/periapical/panoramic split
- **Clinical Relevance**: Panoramic vs non-panoramic is a meaningful distinction

### Training Recommendations
- **Batch Size**: Start with 16-32 for GPU memory efficiency
- **Learning Rate**: 0.0005 works well with Adam optimizer
- **Regularization**: Dropout 0.5 + weight decay 1e-3
- **Early Stopping**: Patience of 10 epochs prevents overfitting

## Troubleshooting

### Data Issues
- **Imbalanced Classes**: Check class distribution in split summary
- **Missing Images**: Verify S3 paths and image directory structure
- **Format Errors**: Ensure JSON files follow expected format

### Training Issues
- **Overfitting**: Increase regularization or reduce model complexity
- **Underfitting**: Decrease regularization or increase model capacity
- **Memory Issues**: Reduce batch size or image resolution

### Performance Issues
- **Low Accuracy**: Check data quality and class balance
- **High Accuracy**: Investigate potential data leakage
- **Unstable Training**: Adjust learning rate or add more regularization

## Contributing

When making changes:
1. Test both local and SageMaker training modes
2. Verify JSON format compatibility with both datasets
3. Update documentation for new features
4. Test with different model architectures
5. Validate binary classification metrics

## Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
boto3>=1.18.0  # For S3 and SageMaker
sagemaker>=2.0.0  # For SageMaker only
```

## Example Workflow

```bash
# 1. Prepare data from existing S3 splits
python create_train_val_split.py \
    --bucket codentist-general \
    --panoramic-prefix datasets/panoramic_dataset \
    --merged-bw-pa-prefix datasets/merged_bw_pa \
    --output-dir ./data

# 2. Upload to S3 (already done - files are uploaded to codentist-general)
# Files are now available at:
# - s3://codentist-general/datasets/panoramic-vs-non-panoramic/train.json
# - s3://codentist-general/datasets/panoramic-vs-non-panoramic/val.json

# 3. Launch training
python sagemaker_launcher.py \
    --train-json s3://codentist-general/datasets/panoramic-vs-non-panoramic/train.json \
    --val-json s3://codentist-general/datasets/panoramic-vs-non-panoramic/val.json \
    --image-dir s3://codentist-general/datasets/ \
    --instance-type ml.g4dn.xlarge
```

## SageMaker Execution Role

The launcher uses a default SageMaker execution role: `arn:aws:iam::552401576656:role/SageMakerExecutionRole`

If you need to use a different role, specify it with the `--role` argument:

```bash
python sagemaker_launcher.py \
    --role arn:aws:iam::YOUR-ACCOUNT:role/YourSageMakerRole \
    --train-json s3://your-bucket/train.json \
    --val-json s3://your-bucket/val.json \
    --image-dir s3://your-bucket/images/
```

This system provides a robust, production-ready solution for binary panoramic vs non-panoramic dental X-ray classification with comprehensive monitoring and evaluation capabilities.
