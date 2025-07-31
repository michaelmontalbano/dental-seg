# Ordinal Regression for Diameter Classification

This directory contains an implementation of diameter classification using ordinal regression, which respects the natural ordering of diameter values.

## Key Features

- **CORN (Consistent Rank Logits)**: Ensures predictions respect ordering
- **Ordinal Loss**: Penalizes predictions based on distance from true value
- **Confidence Intervals**: Provides uncertainty estimates for diameter predictions
- **Smooth Transitions**: Better handling of boundary cases between bins

## Architecture

```
Image -> ViT Backbone -> Features -> Ordinal Head
                                          |
                                          v
                                  K-1 Binary Classifiers
                                  (for K ordered classes)
                                          |
                                          v
                                  P(y > 0), P(y > 1), ..., P(y > K-2)
                                          |
                                          v
                                  Ordinal Probabilities
```

## Mathematical Foundation

For K ordered classes, we use K-1 binary classifiers:
- h₁(x): P(y > 0)
- h₂(x): P(y > 1)
- ...
- hₖ₋₁(x): P(y > K-2)

The probability of class k is:
- P(y = 0) = 1 - h₁(x)
- P(y = k) = hₖ(x) - hₖ₊₁(x) for 0 < k < K-1
- P(y = K-1) = hₖ₋₁(x)

## Benefits

- Respects the natural ordering of diameter values
- More accurate predictions near bin boundaries
- Provides calibrated confidence intervals
- Reduces large errors (e.g., predicting 3.0mm as 6.0mm)

## Loss Function

```python
# CORN loss ensures consistency: h₁(x) ≥ h₂(x) ≥ ... ≥ hₖ₋₁(x)
corn_loss = sum(max(0, hₖ₊₁(x) - hₖ(x))) + ordinal_ce_loss
```

## Usage

```bash
python train_diameter_ordinal.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --bin_size 0.5 \
    --use_corn true \
    --ordinal_weight 0.1
