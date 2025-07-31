# Hyperparameter Optimization for Regression Models

This directory contains a comprehensive hyperparameter optimization system for training regression models (length/diameter) with company embeddings.

## 🚀 Quick Start

### 1. Single Training Job
```bash
# Train diameter regression
python sagemaker_launcher_diameter.py

# Train length regression  
python sagemaker_launcher_length.py

# Train both length and diameter
python sagemaker_launcher_both.py
```

### 2. Hyperparameter Tuning
```bash
# Launch hyperparameter tuning for diameter
python hyperparameter_tuning.py --task diameter --max_jobs 20

# Quick test with fewer jobs
python hyperparameter_tuning.py --task length --max_jobs 5 --max_parallel_jobs 2

# Analyze results after tuning completes
python analyze_tuning_results.py --task diameter
```

## 📊 Hyperparameter Search Space

### Key Parameters Being Optimized:
- **Learning Rate**: 1e-5 to 1e-3 (log scale)
- **Batch Size**: [16, 32, 64]
- **Model Size**: ['tiny', 'small', 'base']
- **Embedding Dimension**: [128, 256, 512]
- **Unfreeze Schedule**: When to start unfreezing the company model
- **Company LR Multiplier**: Learning rate ratio for company model
- **Attention Mechanism**: Whether to use attention weighting

## 🔧 Configuration Options

### tuning_config.yaml
Contains predefined search spaces:
- `default`: Standard search space
- `quick_test`: Reduced space for testing
- `extensive`: Extended search for thorough optimization
- `diameter_specific`: Optimized for diameter task
- `length_specific`: Optimized for length task

## 📈 Monitoring Training

### During Training:
- SageMaker console shows real-time metrics
- CloudWatch logs capture detailed training progress
- Metrics reported: val_loss, length_mae, diameter_mae, avg_mae

### After Training:
```bash
# Analyze tuning results
python analyze_tuning_results.py --task diameter

# This generates:
# - tuning_results_diameter_YYYYMMDD_HHMMSS.csv
# - best_hyperparameters_diameter.json
# - parameter_analysis_diameter.png
# - model_size_comparison_diameter.png
```

## 🏆 Best Practices

1. **Start Small**: Use `--max_jobs 5` for initial testing
2. **Monitor Early Stopping**: Jobs that aren't improving will be stopped automatically
3. **Progressive Tuning**: Start with quick_test config, then refine search space
4. **Task-Specific**: Use task-specific configs for better results

## 🐛 Troubleshooting

### Common Issues:

1. **Numpy String Error** (Fixed):
   - Cause: Length/diameter values stored as strings
   - Solution: Convert to float in dataset loading

2. **Out of Memory**:
   - Reduce batch_size in search space
   - Use smaller model_size options

3. **Slow Training**:
   - Reduce epochs in static_hyperparameters
   - Use fewer max_parallel_jobs

## 📁 Output Files

### Training Outputs:
- `best_model.pth`: Best model checkpoint
- `training_history.json`: Detailed training metrics
- `final_model.pth`: Final epoch model (if save_best_only=false)

### Tuning Outputs:
- `tuning_job_<task>.txt`: Tuning job name for later reference
- `tuning_results_<task>_<timestamp>.csv`: All job results
- `best_hyperparameters_<task>.json`: Best parameters found
- `*.png`: Visualization plots

## 🔄 Workflow Example

```bash
# 1. Launch hyperparameter tuning
python hyperparameter_tuning.py --task diameter --max_jobs 20

# 2. Monitor in SageMaker console
# Wait for jobs to complete...

# 3. Analyze results
python analyze_tuning_results.py --task diameter

# 4. Use best hyperparameters for production training
# Edit sagemaker_launcher_diameter.py with best params from JSON

# 5. Launch final training with best params
python sagemaker_launcher_diameter.py
```

## 🎯 Expected Performance

Based on the model architecture:
- **Diameter MAE**: Target < 0.5mm
- **Length MAE**: Target < 2.0mm  
- **Training Time**: 2-6 hours per job
- **Best Model Size**: Typically 'base' or 'small'

## 💡 Tips

1. **Company Model Unfreezing**: Critical for performance
   - Frozen: Poor results (like 2.5% accuracy)
   - Unfrozen: Much better (40%+ accuracy improvement seen)

2. **Attention Mechanism**: Usually helps
   - Learns to weight company vs visual features
   - Typically improves by 5-10%

3. **Learning Rate**: Company model needs lower LR
   - Main model: 1e-4
   - Company model: 1e-5 (0.1x multiplier)
