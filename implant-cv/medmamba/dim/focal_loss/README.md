# Focal Loss for Company Classification

This directory contains an implementation of company classification using Focal Loss to handle class imbalance.

## Key Features

- **Focal Loss**: Addresses class imbalance by down-weighting easy examples and focusing on hard negatives
- **Alpha weighting**: Per-class weights based on inverse frequency
- **Gamma parameter**: Controls the rate at which easy examples are down-weighted

## Implementation Details

The focal loss is defined as:
```
FL(pt) = -α(1-pt)^γ log(pt)
```

Where:
- `pt` is the model's estimated probability for the correct class
- `α` is the weighting factor (inverse class frequency)
- `γ` is the focusing parameter (typically 2.0)

## Benefits

- Better performance on minority classes (rare companies)
- Reduces the impact of easy-to-classify examples
- Automatically handles class imbalance without manual reweighting

## Usage

```bash
python train_company_focal.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --focal_gamma 2.0 \
    --focal_alpha auto
