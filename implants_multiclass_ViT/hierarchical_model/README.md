# Hierarchical Model Classification

This directory contains an implementation where model classification is conditioned on company predictions using a hierarchical approach.

## Key Features

- **Conditional Cross-Entropy**: Model loss weighted by company prediction confidence
- **Hierarchical Structure**: Exploits the natural hierarchy (Company → Model)
- **Attention Mechanism**: Uses company features to guide model predictions
- **Confidence Weighting**: Uncertain company predictions reduce model loss impact

## Architecture

```
Image -> ViT Backbone -> Features
                           |
                    +------+------+
                    |             |
                    v             v
              Company Head   Attention Module
                    |             |
             Company Logits       |
             & Confidence         |
                    |             |
                    +------+------+
                           |
                           v
                  Weighted Model Head -> Model Logits
```

## Loss Function

```python
model_loss = company_confidence * CE(model_logits, model_target)
total_loss = company_loss + λ * model_loss
```

## Benefits

- Prevents model misclassification when company is uncertain
- Learns company-model relationships implicitly
- Reduces error propagation in hierarchical structure
- Can handle models that appear across multiple companies

## Usage

```bash
python train_hierarchical_model.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --hierarchy_weight 0.5 \
    --confidence_threshold 0.8
