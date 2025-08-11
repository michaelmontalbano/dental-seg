# Vision Transformer for Implant Classification

Multi-task implant classification using Vision Transformers and CNNs for company, model, diameter, and length prediction.

## Method
- Vision Transformer (ViT-B/16) and CNN architectures
- Multi-task learning with uncertainty weighting
- Hierarchical classification (company → model → dimensions)
- Hyperparameter optimization (HPO) support

## Default Dataset
Uses implant classification datasets organized as train/company_name/image.jpg structure:
- `implants_cls_dataset`: Primary dataset with data.yaml
- Expects augmented implant radiographs organized by manufacturer
- Supports multi-scale length prediction and ordinal regression

## Usage
```bash
# Company classification
python train_company_classifier.py --data-path /path/to/implants_cls_dataset

# Multi-task ViT training
python train_vit_multitask.py --data-path /path/to/dataset --epochs 100

# SageMaker with HPO
python sagemaker_launcher_company_hpo.py --instance-type ml.g4dn.xlarge
```

## Key Features
- Company embeddings for improved model classification
- Focal loss for handling class imbalance
- Uncertainty weighting for multi-task optimization
- Diameter/length regression with company conditioning
