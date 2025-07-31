# BW_VS_PANS - JSON-based Dental X-ray Classification

This directory contains a rewritten version of the BW_VS_PANS dental X-ray classification system that uses `train.json` and `val.json` files for training data instead of the previous S3-based approach.

## Overview

The system classifies dental X-rays into three categories:
- **Bitewing**: Side view X-rays showing upper and lower teeth
- **Periapical**: X-rays showing the entire tooth from crown to root
- **Panoramic**: Wide-angle X-rays showing the entire mouth

## Key Changes

### 1. JSON-based Data Loading
- **Before**: Used S3 bucket with `annotations.json` and runtime data splitting
- **After**: Uses pre-split `train.json` and `val.json` files with consistent train/val splits

### 2. Simplified Architecture
- Removed S3-specific code and boto3 dependencies
- Direct file-based image loading with better error handling
- Cleaner dataset class focused on JSON annotation format

### 3. Enhanced Training Features
- Comprehensive overfitting detection
- Progress bars with real-time metrics
- Detailed per-class evaluation every 5 epochs
- Early stopping with configurable patience
- Center cropping to remove corner artifacts and text labels

## File Structure

```
bw_vs_pans/
├── src/
│   ├── dataset.py          # JSON-based dataset class
│   ├── train.py           # Training script with overfitting detection
│   └── requirements.txt   # Python dependencies
├── sagemaker_launcher.py  # SageMaker training launcher
├── model/                 # Saved models and metrics
└── README.md             # This file
```

## Data Format

### JSON Annotation Format
Both `train.json` and `val.json` follow this structure:

```json
{
  "images": [
    {
      "id": 404437371,
      "file_name": "404437371.jpg",
      "width": 1692,
      "height": 1324,
      "xray_type": "bitewing"
    },
    {
      "id": 373067253,
      "file_name": "373067253.jpg", 
      "width": 844,
      "height": 660,
      "xray_type": "panoramic"
    }
  ]
}
```

### Expected Data Locations
- **Source Data**: `s3://codentist-general/datasets/bw_pa_pans/`
  - `annotations.json` (original annotations)
  - `bitewing/` (bitewing X-ray images)
  - `panoramic/` (panoramic X-ray images)
  - `periapical/` (periapical X-ray images)
- **Generated Files**: `train.json` and `val.json` (created by conversion script)
- **Images**: Organized in subdirectories by X-ray type

## Setup

### Step 1: Create Train/Val Split from Source Data

Before training, you need to convert the source data from `s3://codentist-general/datasets/bw_pa_pans/` into `train.json` and `val.json` files:

```bash
cd cej-ac-detection/bw_vs_pans

# Create train/val split from S3 data
python create_train_val_split.py \
    --bucket codentist-general \
    --dataset-prefix datasets/bw_pa_pans \
    --output-dir ./data \
    --val-ratio 0.2 \
    --download-annotations

# This will create:
# - ./data/train.json
# - ./data/val.json  
# - ./data/split_summary.json
```

**Key Features of the Split Script:**
- **Filename Filtering**: Automatically excludes test/calibration images, thumbnails, and system files
- **Stratified Splitting**: Ensures balanced class distribution in train/val sets
- **S3 Integration**: Directly reads from S3 bucket structure
- **Reproducible**: Uses fixed random seed for consistent splits

**Filtering Rules Applied:**
- **Bitewing-specific Rule**: Only keep files ending in `.JPG` (exclude `.png` files)
  - ❌ Exclude: `bitewing_0001_bitewing_2007.png` (PNG format)
  - ✅ Keep: `bitewing_0001_X12255_9.JPG` (JPG format with patient ID)
- **General Rule**: Excludes files where the X-ray type appears twice in the filename
- **Periapical/Panoramic**: Keep all standard image formats (jpg, jpeg, png, tiff)
- Excludes hidden files (starting with `._`)

### Step 2: Upload Generated Files to S3

```bash
# Upload the generated JSON files to S3 for SageMaker training
aws s3 cp ./data/train.json s3://codentist-general/datasets/bw_pa_pans/train.json
aws s3 cp ./data/val.json s3://codentist-general/datasets/bw_pa_pans/val.json
```

## Usage

### Local Training

```bash
cd cej-ac-detection/bw_vs_pans/src

# Basic training
python train.py --image_dir /path/to/images

# Custom configuration
python train.py \
    --image_dir /path/to/images \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --model_name efficientnet_b0
```

### SageMaker Training

```bash
cd cej-ac-detection/bw_vs_pans

# Launch SageMaker training job
python sagemaker_launcher.py \
    --train-json s3://bucket/path/to/train.json \
    --val-json s3://bucket/path/to/val.json \
    --image-dir s3://bucket/path/to/images/ \
    --instance-type ml.g4dn.xlarge \
    --epochs 30
```

## Key Features

### 1. Data Leakage Prevention
- **Center Cropping**: Removes 25% from all sides (keeps center 75%)
- **Artifact Removal**: Eliminates equipment markers, text labels, corner artifacts
- **Production Ready**: Works on both labeled and unlabeled images

### 2. Overfitting Detection
- Monitors train/validation accuracy gaps
- Detects suspiciously high accuracies (>75% triggers warning)
- Per-class F1 score gap analysis
- Automatic warnings for rapid convergence

### 3. Robust Training
- Progress bars with real-time loss/accuracy
- Detailed metrics logging every 5 epochs
- Early stopping to prevent overfitting
- Automatic handling of corrupted images
- Learning rate scheduling

### 4. Comprehensive Evaluation
- Per-class precision, recall, F1-score
- Confusion matrices
- Training history tracking
- Overfitting summary reports

## Model Architecture

### Supported Models
- **ResNet-50** (default): Good balance of accuracy and speed
- **EfficientNet-B0**: Efficient architecture for mobile deployment
- **MobileNet-V3-Large**: Lightweight for resource-constrained environments

### Regularization Features
- Batch normalization layers
- Configurable dropout rates (default: 0.5)
- Weight decay regularization
- Early stopping

## Expected Performance

### Realistic Expectations
- **Validation Accuracy**: 60-85% (task-dependent)
- **Random Baseline**: 33% (3 classes)
- **Warning Thresholds**: >90% may indicate data leakage

### Success Criteria
- Balanced per-class performance
- Minimal overfitting warnings
- Stable training convergence
- Accuracy gap <15% between train/val

## Monitoring and Debugging

### Training Logs
- Real-time progress bars
- Epoch-by-epoch metrics
- Overfitting warnings
- Class distribution analysis

### Output Files
- `model.pth`: Trained model with metadata
- `training_metrics.json`: Detailed training history
- Overfitting detection summaries

### Common Issues
- **High accuracy (>90%)**: Check for data leakage
- **Large train/val gap**: Increase regularization
- **Missing images**: Verify image paths in JSON files
- **Memory errors**: Reduce batch size

## Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
boto3>=1.18.0  # For SageMaker only
sagemaker>=2.0.0  # For SageMaker only
```

## Migration from Previous Version

### Key Differences
1. **Data Source**: JSON files instead of S3 annotations
2. **Splitting**: Pre-split data instead of runtime splitting
3. **Dependencies**: Removed S3-specific requirements for local training
4. **Paths**: Updated argument names and path handling

### Migration Steps
1. Convert existing annotations to train.json/val.json format
2. Update image paths to point to local directory
3. Use new argument names in training scripts
4. Update SageMaker launcher configuration

## Troubleshooting

### Common Errors
- **FileNotFoundError**: Check JSON file paths and image directory
- **KeyError**: Verify JSON format matches expected structure
- **CUDA errors**: Reduce batch size or use CPU training
- **Import errors**: Install required dependencies

### Performance Issues
- **Slow loading**: Reduce num_workers or check disk I/O
- **Memory usage**: Decrease batch size or image size
- **Training stalls**: Check for corrupted images or data issues

## Contributing

When making changes:
1. Test both local and SageMaker training modes
2. Verify JSON format compatibility
3. Update documentation for new features
4. Test with different model architectures
5. Validate overfitting detection accuracy
