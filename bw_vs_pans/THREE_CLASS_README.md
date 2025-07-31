# Three-Class X-ray Classification: Bitewing vs Periapical vs Panoramic

This document describes the three-class classification system for dental X-ray images, extending the original bitewing vs panoramic classification to include periapical images.

## Overview

The three-class system classifies dental X-ray images into:
- **Bitewing (label 0)**: Smaller rectangular images showing crowns of upper and lower teeth
- **Periapical (label 1)**: Roughly square images showing entire teeth including roots
- **Panoramic (label 2)**: Wide rectangular images showing full jaw view

## Dataset Structure

Source: `s3://codentist-general/datasets/bw_pa_pans/`
```
datasets/bw_pa_pans/
├── bitewing/     # Bitewing X-ray images (JPG only)
├── periapical/   # Periapical X-ray images (all formats)
└── panoramic/    # Panoramic X-ray images (all formats)
```

## Scripts

### 1. create_three_class_split.py

Main script for creating train/val splits for three-class classification.

**Key Features:**
- Processes images from three S3 directories
- Applies specific filtering rules per image type
- Creates stratified train/validation splits
- Generates JSON files with class labels and metadata
- Includes class imbalance analysis

**Usage:**
```bash
python create_three_class_split.py \
    --bucket codentist-general \
    --dataset-prefix datasets/bw_pa_pans \
    --output-dir ./data \
    --val-ratio 0.2 \
    --random-seed 42
```

**Arguments:**
- `--bucket`: S3 bucket name (default: codentist-general)
- `--dataset-prefix`: S3 prefix for dataset (default: datasets/bw_pa_pans)
- `--output-dir`: Output directory for JSON files (default: ./data)
- `--val-ratio`: Validation set ratio (default: 0.2)
- `--random-seed`: Random seed for reproducible splits (default: 42)

### 2. run_three_class_split.py

Simple runner script with default parameters.

**Usage:**
```bash
python run_three_class_split.py
```

## Filtering Rules

### Bitewing Images
- **File format**: Only `.JPG` and `.JPEG` files accepted
- **Exclusions**: `.png` files are excluded
- **Rationale**: Bitewing images in this dataset are primarily in JPG format

### Periapical and Panoramic Images
- **File format**: All standard image formats accepted (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`)
- **No format restrictions**

### General Rules (All Types)
- **Hidden files**: Files starting with `._` are excluded
- **Non-image files**: Only image formats are included
- **Duplicate names**: Files with duplicate X-ray type in filename are **INCLUDED** (changed from original behavior)
  - Example: `bitewing_1231_bitewing.jpg` is now included

## Output Files

### train.json and val.json
JSON files containing image metadata and labels:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "example.jpg",
      "width": 0,
      "height": 0,
      "xray_type": "bitewing",
      "label": 0,
      "s3_key": "datasets/bw_pa_pans/bitewing/example.jpg"
    }
  ],
  "class_mapping": {
    "bitewing": 0,
    "periapical": 1,
    "panoramic": 2
  },
  "num_classes": 3
}
```

### three_class_split_summary.json
Comprehensive summary including:
- Dataset statistics
- Class distribution
- Class balance analysis
- Split information
- Filtering rules applied

## Class Mapping

| Class Name | Label | Description |
|------------|-------|-------------|
| bitewing | 0 | Small rectangular images showing tooth crowns |
| periapical | 1 | Square-ish images showing entire teeth with roots |
| panoramic | 2 | Wide rectangular images showing full jaw |

## Classification Challenges

### Size/Aspect Ratio Variations
- **Panoramic**: Wide rectangles (high width/height ratio)
- **Periapical**: Roughly square (width/height ≈ 1)
- **Bitewing**: Smaller rectangles (moderate width/height ratio)

### Content Overlap
- Periapicals and bitewings can show similar tooth structures
- Scale and angle differences are key distinguishing features
- Root visibility is a major differentiator

## Recommended Solutions

### 1. Preprocessing Standardization
```python
# Resize with padding to preserve aspect ratios
def preprocess_image(image, target_size=(224, 224)):
    # Calculate padding to maintain aspect ratio
    # Resize with padding instead of stretching
    pass
```

### 2. Aspect Ratio as Feature
```python
# Add width/height ratio as explicit input
def extract_aspect_ratio(image):
    height, width = image.shape[:2]
    return width / height
```

### 3. Multi-Scale Training
- Use architectures like Feature Pyramid Networks (FPN)
- Train on multiple scales simultaneously
- Consider different input resolutions for different classes

### 4. Data Augmentation Strategy
```python
# Include crops of panoramics that might resemble periapicals
# Focus on root visibility and field of view differences
augmentation_strategies = {
    'panoramic': ['crop_sections', 'scale_variations'],
    'periapical': ['rotation', 'brightness'],
    'bitewing': ['horizontal_flip', 'contrast']
}
```

### 5. Ensemble Approach
1. **Stage 1**: Binary classifier (panoramic vs. non-panoramic)
2. **Stage 2**: Binary classifier (periapical vs. bitewing)

### 6. Architecture Considerations
- **Panoramic separation**: Should be easiest due to distinctive full-jaw view
- **Periapical vs. Bitewing**: Focus on root visibility and field of view
- Consider attention mechanisms to focus on distinguishing features

## Class Imbalance Handling

The script automatically detects class imbalance and provides warnings when the imbalance ratio exceeds 3:1.

**Strategies for imbalanced datasets:**
1. **Weighted Loss Functions**
2. **Oversampling** (SMOTE, ADASYN)
3. **Undersampling** 
4. **Focal Loss** for hard examples
5. **Class-balanced sampling** during training

## Integration with Training Pipeline

### Update Dataset Class
```python
class ThreeClassXrayDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.images = self.data['images']
        self.num_classes = self.data['num_classes']  # 3
        self.class_mapping = self.data['class_mapping']
        
    def __getitem__(self, idx):
        img_info = self.images[idx]
        # Load image from S3 using img_info['s3_key']
        # Return image, label (0, 1, or 2)
```

### Update Model Architecture
```python
# Change final layer to output 3 classes
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
```

### Update Loss Function
```python
# For imbalanced datasets
class_weights = torch.tensor([w0, w1, w2])  # Based on class frequencies
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## Evaluation Metrics

For three-class classification, use:
- **Overall Accuracy**
- **Per-class Precision, Recall, F1-score**
- **Confusion Matrix** (3x3)
- **Macro/Micro averaged metrics**
- **Cohen's Kappa** for multi-class agreement

## Next Steps

1. **Run the script** to generate train.json and val.json
2. **Analyze class distribution** and imbalance
3. **Update training pipeline** for 3-class classification
4. **Implement preprocessing** with aspect ratio preservation
5. **Consider ensemble approach** if single model struggles
6. **Evaluate on validation set** with comprehensive metrics

## Comparison with Original Script

| Feature | Original (create_train_val_split.py) | New (create_three_class_split.py) |
|---------|--------------------------------------|-----------------------------------|
| Classes | 2 (bitewing, panoramic) | 3 (bitewing, periapical, panoramic) |
| Bitewing format | JPG only | JPG only (unchanged) |
| Other formats | All image formats | All image formats (unchanged) |
| Duplicate names | Excluded | **INCLUDED** (changed) |
| Class labels | Binary (0, 1) | Multi-class (0, 1, 2) |
| Imbalance analysis | Basic | **Enhanced** with warnings |
| Output format | Standard | **Enhanced** with class mapping |

The new script maintains compatibility with the existing pipeline while extending functionality for three-class classification and providing better analysis tools for dataset quality assessment.
