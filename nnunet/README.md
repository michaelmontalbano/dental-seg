# nnU-Net for Dental Segmentation

Medical-grade segmentation for dental X-ray analysis using nnU-Net framework.

## 🎯 Overview

nnU-Net is a self-configuring method for semantic segmentation that automatically adapts to any new dataset. This implementation converts dental COCO annotations to nnU-Net format and trains state-of-the-art segmentation models.

## 🚀 Quick Start

### Build Docker Container
```bash
cd container_nnunet
./build_and_push.sh
```

### Launch Training on SageMaker
```bash
# Using containerized approach (recommended)
python sagemaker_launcher_nnunet.py \
    --class-group bone-loss \
    --instance-type ml.g4dn.xlarge \
    --dataset-id 701 \
    --fold 0

# Or using PyTorch estimator
python sagemaker_launcher.py \
    --class-group bone-loss \
    --instance-type ml.g4dn.xlarge \
    --dataset-id 701 \
    --fold 0
```

## 📋 Features

- **Auto-configuration**: nnU-Net automatically determines optimal architecture and hyperparameters
- **COCO → nnU-Net conversion**: Seamlessly converts dental COCO annotations
- **Class group support**: Same flexible class grouping as YOLOv8
- **5-fold cross-validation**: Built-in support for robust evaluation
- **Medical optimization**: Specifically tuned for medical imaging

## 🦷 Supported Class Groups

Same as YOLOv8 implementation:
- `bone-loss`: CEJ, AC, Apex landmarks
- `tooth-anatomy`: Enamel, Pulp
- `decay-enamel-pulp-coronal`: Clinical structures
- `dental-work`: Fillings, crowns, implants
- And more...

## 🔧 Key Parameters

### Dataset Configuration
- `--dataset-id`: Unique identifier for nnU-Net dataset (default: 701)
- `--fold`: Which fold to train (0-4)
- `--class-group`: Subset of classes to segment

### SageMaker Specific
- `--instance-type`: Use GPU instances (ml.g4dn.xlarge or larger)
- `--volume-size`: 100GB recommended for preprocessing
- `--max-run`: 432000 (5 days) for complete training

## 📊 Training Pipeline

1. **Data Conversion**: COCO → nnU-Net format (.nii.gz files)
2. **Planning**: Automatic dataset analysis and configuration
3. **Preprocessing**: Resampling, normalization, augmentation setup
4. **Training**: 1000 epochs with automatic learning rate scheduling
5. **Export**: Model saved in nnU-Net format

## 💡 Tips

### For Best Results:
- Train all 5 folds and ensemble predictions
- Use larger GPU instances for faster training (ml.g4dn.2xlarge)
- Allow sufficient time - nnU-Net trains for 1000 epochs

### Multi-fold Training:
```bash
# Train all folds
for fold in 0 1 2 3 4; do
    python sagemaker_launcher.py \
        --class-group bone-loss \
        --fold $fold \
        --job-name nnunet-fold-$fold
done
```

### Memory Management:
- Reduce batch size if OOM errors occur
- Use gradient checkpointing (enabled by default)
- Consider 2D configuration for large images

## 🔬 Advanced Usage

### Custom Dataset IDs
Each class group should use a unique dataset ID:
```python
dataset_ids = {
    'bone-loss': 701,
    'tooth-anatomy': 702,
    'decay': 703,
    # etc...
}
```

### Inference Pipeline
After training:
```python
# Use nnU-Net inference
nnUNetv2_predict -i input_folder -o output_folder -d 701 -c 2d -f 0
```

### Ensemble Multiple Folds
```python
nnUNetv2_ensemble -f fold_0 fold_1 fold_2 fold_3 fold_4 -o ensemble_output
```

## 📈 Expected Performance

- **Training time**: 2-4 hours per fold on ml.g4dn.xlarge
- **Dice scores**: Typically 85-95% for dental structures
- **Memory usage**: ~8-16GB GPU memory
- **Disk space**: ~50-100GB for preprocessing

## 🐛 Troubleshooting

### Installation Issues
- Ensure CUDA toolkit matches PyTorch version
- Install SimpleITK and nibabel dependencies

### Training Errors
- Check dataset conversion logs
- Verify all images are grayscale
- Ensure sufficient disk space for preprocessing

### Performance Issues
- Use SSD storage for faster I/O
- Increase preprocessing workers
- Consider distributed training for large datasets

## 📚 References

- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [Medical Segmentation Best Practices](https://github.com/MIC-DKFZ/nnUNet/wiki)
