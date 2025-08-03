# Dual Embeddings System for Implant Dimension Prediction

This directory contains an advanced system that uses both company and model-type embeddings to predict implant dimensions (length and diameter).

## 🚀 Overview

The dual embeddings approach leverages:
1. **Company Embeddings**: Captures manufacturer-specific design patterns
2. **Model Embeddings**: Captures model-specific characteristics within manufacturers
3. **Attention Mechanisms**: Dynamically weights the importance of each embedding type

## 📁 Key Files

### Training Scripts
- `train_dual_embeddings_regression.py` - Main training script using both embeddings
- `train_model_classifier_hpo.py` - Model classification with HPO support
- `train_regression_with_embeddings.py` - Single embedding regression (updated with JSON fix)

### SageMaker Launchers
- `sagemaker_launcher_dual_embeddings_hpo.py` - HPO for dual embeddings
- `../sagemaker_launcher_model_hpo.py` - HPO for model classification

## 🏗️ Architecture

### Dual Embedding Model
```
Input Image → 
├── Company Embedding Model (Frozen → Gradual Unfreeze)
│   └── Company Features (256d)
├── Model Embedding Model (Frozen → Gradual Unfreeze)
│   └── Model Features (256d)
└── Visual Features (768d for ViT-Base)
    ↓
Attention Mechanism (3 types available):
- Dual: Weights company vs model embeddings
- Triple: Weights features vs company vs model
- Hierarchical: Two-level attention
    ↓
Regression Head → Length & Diameter Predictions
```

### Attention Types

1. **Dual Attention**: 
   - Focuses on balancing company vs model embeddings
   - Best for when visual features are less important

2. **Triple Attention**: 
   - Balances all three: visual features, company, and model embeddings
   - Most flexible, good default choice

3. **Hierarchical Attention**: 
   - First combines company + model embeddings
   - Then balances combined embeddings vs visual features
   - Best when embeddings should be considered together

## 🎯 Training Workflow

### Step 1: Train Model Classifier
```bash
cd ../
python sagemaker_launcher_model_hpo.py
```
This creates model-type embeddings. Wait for completion (~3-5 hours).

### Step 2: Train Dual Embeddings Regression
```bash
cd company_embeddings/
python sagemaker_launcher_dual_embeddings_hpo.py
```
Uses both company and model embeddings. Takes ~5-8 hours.

### Step 3: Analyze Results
```bash
python analyze_tuning_results.py --tuning-job-name <job-name>
```

## 📊 Hyperparameter Tuning

### Model Classification HPO
- **Metric**: Validation accuracy (maximize)
- **Key parameters**: model_size, learning_rate, batch_size
- **Jobs**: 20 total, 4 parallel

### Dual Embeddings HPO
- **Metric**: Average MAE for length+diameter (minimize)
- **Key parameters**: 
  - Embedding dimensions (128, 256, 512)
  - Attention type (dual, triple, hierarchical)
  - Unfreezing schedules
  - Learning rate multipliers

## 🔧 Key Features

### Progressive Unfreezing
- **Epoch 0-20**: Only regression head trains
- **Epoch 20-40**: 50% of embedding models unfreeze
- **Epoch 40+**: Full model trains

### Learning Rate Management
- Regression head: Full learning rate
- Embedding models: 10% of base LR (configurable)
- Cosine annealing with warmup

### Robust Data Handling
- Validates all numeric values before training
- Handles empty strings and invalid entries
- Provides detailed logging of skipped entries

## 📈 Expected Performance

With dual embeddings, expect:
- **Length MAE**: ~0.6-0.8mm (vs 0.75mm with single embedding)
- **Diameter MAE**: ~0.3-0.35mm (vs 0.34mm with single embedding)
- **Better generalization** to unseen company/model combinations

## 🚨 Common Issues

### JSON Serialization Error
- **Fixed**: Added `convert_to_serializable()` function
- Converts numpy types to Python native types

### Empty String Conversion
- **Fixed**: Robust validation in dataset initialization
- Filters out invalid entries before training

### GPU Memory
- If OOM, reduce batch_size or embedding dimensions
- Consider using gradient accumulation

## 💡 Tips

1. **Start with triple attention** - it's the most flexible
2. **Use base model size** for best accuracy/speed tradeoff
3. **Monitor both MAE metrics** - sometimes one improves at the expense of the other
4. **Save checkpoints** from best performing HPO trials

## 📝 Example Usage

### Quick Test (No HPO)
```python
# Test dual embeddings locally
python train_dual_embeddings_regression.py \
    --train_json augmented_train.json \
    --val_json augmented_val.json \
    --company_checkpoint vit_company.pth \
    --model_checkpoint vit_model.pth \
    --attention_type triple \
    --epochs 50
```

### Production Training
```bash
# Use HPO for best results
python sagemaker_launcher_dual_embeddings_hpo.py
```

## 🔮 Future Improvements

1. **Multi-task Learning**: Add auxiliary tasks (e.g., thread type prediction)
2. **Uncertainty Estimation**: Add dropout-based uncertainty
3. **Cross-Attention**: More sophisticated attention between embeddings
4. **Ensemble Methods**: Combine predictions from multiple architectures
