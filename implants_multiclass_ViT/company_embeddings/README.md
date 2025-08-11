# Company Embeddings for Hierarchical Classification

This directory contains an implementation where company classification produces embeddings that are reused by the model classifier.

## Key Features

- **Company Embeddings**: 256-dimensional learned representations for each company
- **Hierarchical Structure**: Model predictions are conditioned on company embeddings
- **Cross-task Knowledge Transfer**: Company features help model classification

## Architecture

```
Image -> ViT Backbone -> Features
                           |
                           v
                    Company Head -> Company Logits
                           |
                           v
                  Company Embeddings
                           |
                           v
      [Features + Company Embeddings] -> Model Head -> Model Logits
```

## Benefits

- Models learn company-specific patterns
- Reduces confusion between similar models from different companies
- Enables zero-shot model classification for new models from known companies
- Company embeddings can be visualized to understand company relationships

## Implementation Details

- Company embeddings are extracted from the penultimate layer of the company classifier
- These embeddings are concatenated with image features for model classification
- Optionally, attention mechanism can be used to weight company influence

## Usage

```bash
python train_hierarchical.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --embed_dim 256 \
    --use_attention true
