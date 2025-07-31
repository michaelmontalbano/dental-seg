# Multi-Scale Feature Extraction for Length Classification

This directory contains an implementation of length classification using multi-scale feature extraction to capture both global context and fine details.

## Key Features

- **Multi-Scale ViT**: Extracts features at different spatial resolutions
- **Feature Pyramid**: Combines high-level and low-level features
- **Auxiliary Regression**: Additional regression head for fine-grained predictions
- **Smooth Label Distribution**: Accounts for measurement uncertainty

## Architecture

```
Image -> Multi-Scale Preprocessing
            |
            +-- 224x224 -> ViT_1 -> Global Features
            |
            +-- 384x384 -> ViT_2 -> Detail Features
            |
            +-- Crop Regions -> ViT_3 -> Local Features
                                            |
                                            v
                                    Feature Fusion
                                            |
                                    +-------+-------+
                                    |               |
                                    v               v
                            Classification    Regression
                                Head            Head
                                    |               |
                                    v               v
                                Length Bins    Fine Length
```

## Key Innovations

1. **Multi-Resolution Processing**: Different scales capture different aspects:
   - 224x224: Overall implant shape and context
   - 384x384: Thread patterns and surface details
   - Crops: Tip and collar regions

2. **Smooth Labels**: Instead of hard bin assignments:
   ```python
   # Gaussian distribution around true length
   smooth_label = exp(-((bin_centers - true_length) / σ)²)
   ```

3. **Dual Output**: 
   - Classification output for robust bin prediction
   - Regression output for precise length estimation

## Benefits

- Captures both macro (overall length) and micro (thread count) features
- Robust to image quality variations
- Provides both categorical and continuous predictions
- Better handles edge cases at bin boundaries

## Usage

```bash
python train_length_multiscale.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --scales "224,384" \
    --use_crops true \
    --smooth_sigma 0.5 \
    --aux_regression_weight 0.3
