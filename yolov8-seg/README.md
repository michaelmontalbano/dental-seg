# YOLOv8 Dental Segmentation

A comprehensive YOLOv8-based segmentation system for detecting and segmenting dental landmarks including CEJ (Cemento-Enamel Junction), Alveolar Crest, and Apex regions in dental X-ray images.

## Features

- **Instance Segmentation**: Provides precise pixel-level segmentation masks for dental landmarks
- **Multi-class Segmentation**: Supports segmentation of various dental landmarks and conditions
- **Class Group Filtering**: Train on specific groups (bone-loss, teeth, conditions, surfaces)
- **X-ray Type Filtering**: Filter by panoramic, bitewing, or periapical X-rays
- **SageMaker Integration**: Full AWS SageMaker support for cloud training
- **Data Enhancement**: CLAHE enhancement for better contrast
- **Comprehensive Logging**: Detailed debugging and monitoring

## Project Structure

```
yolov8-seg/
├── src/
│   ├── train.py           # Main training script
│   ├── dataset.py         # Dataset processing and segmentation format conversion
│   ├── requirements.txt   # Python dependencies
│   ├── evaluate.py        # Model evaluation script
│   └── utils.py           # Utility functions for segmentation
├── sagemaker_launcher.py  # SageMaker job launcher
├── quick_test.py          # Quick testing script
├── config/
│   └── training_configs.yaml  # Training configurations
└── README.md             # This file
```

## Class Groups

### bone-loss (Default)
- `apex`: Root apex regions
- `cej mesial`: CEJ mesial regions
- `cej distal`: CEJ distal regions
- `ac distal`: Alveolar crest distal regions
- `ac mesial`: Alveolar crest mesial regions

### teeth
- `T01-T32`: Permanent teeth segmentation
- `P01-P20`: Primary teeth segmentation

### conditions
- `bridge`, `margin`, `enamel`, `coronal aspect`, `pulp`, `root aspect`
- `filling`, `crown`, `impaction`, `decay`, `rct`, `parl`, `missing`, `implant`

### surfaces
- `distal surface`, `occlusal surface`, `mesial surface`

## Quick Start

### Local Training

1. **Install Dependencies**:
```bash
pip install -r src/requirements.txt
```

2. **Prepare Data**:
```bash
python src/dataset.py --annotations /path/to/annotations.json \
                      --images /path/to/images \
                      --class_group bone-loss \
                      --xray_type panoramic \
                      --task segment
```

3. **Train Model**:
```bash
python src/train.py --epochs 100 \
                    --batch-size 16 \
                    --class-group bone-loss \
                    --annotations annotations.json \
                    --images /path/to/images \
                    --task segment
```

### SageMaker Training

```bash
python sagemaker_launcher.py --class-group bone-loss \
                             --xray-type panoramic \
                             --epochs 100 \
                             --batch-size 16 \
                             --instance-type ml.g4dn.xlarge \
                             --task segment
```

## Configuration

### Training Parameters

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 16)
- `--learning-rate`: Initial learning rate (default: 0.01)
- `--image-size`: Input image size (default: 640)
- `--model-size`: YOLOv8 model size [n,s,m,l,x] (default: n)
- `--task`: Task type - 'segment' for segmentation (default: segment)

### Dataset Filtering

- `--class-group`: Target class group (default: bone-loss)
- `--xray-type`: X-ray type filter (optional)
- `--subset-size`: Use subset for quick testing (optional)

### Advanced Options

- `--wandb`: Enable Weights & Biases logging
- `--resume`: Resume from checkpoint
- `--pretrained`: Use pretrained weights (True/False/path)

## Model Evaluation

```bash
python src/evaluate.py --model-path /path/to/best.pt \
                       --test-data /path/to/test/data \
                       --class-group bone-loss \
                       --task segment
```

## Segmentation vs Detection

Unlike YOLOv12 which provides bounding boxes, YOLOv8 segmentation provides:

- **Pixel-level masks** for precise landmark boundaries
- **Instance segmentation** to distinguish overlapping regions
- **Better spatial accuracy** for clinical measurements
- **Polygon annotations** support for irregular shapes

## Data Format

### Input Annotations
The system expects COCO-style annotations with segmentation masks:
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]], // Polygon points
      "area": 1234.5,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ]
}
```

### Output Format
- **Segmentation masks**: PNG files with pixel-level labels
- **Confidence scores**: Per-pixel confidence values
- **Bounding boxes**: Optional bbox extraction from masks

## AWS SageMaker Setup

1. **Configure AWS CLI**:
```bash
aws configure
```

2. **Set SageMaker Role**:
Update the role ARN in `sagemaker_launcher.py` or pass via `--role`

3. **Upload Data to S3**:
```bash
aws s3 sync /local/data s3://your-bucket/datasets/
```

## Performance Metrics

### Segmentation Metrics
- **IoU (Intersection over Union)**: Pixel-level overlap accuracy
- **Dice Coefficient**: Segmentation similarity measure
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Mean Average Precision (mAP)**: Detection + segmentation combined

### Clinical Relevance
- **Boundary Precision**: Critical for CEJ-AC distance measurements
- **Area Calculations**: Accurate lesion size estimation
- **Shape Analysis**: Morphological feature extraction

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **No Target Classes Found**: Check class group and annotation format
3. **Segmentation Quality**: Adjust confidence threshold and NMS settings
4. **SageMaker Permission Errors**: Verify IAM role permissions

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=/path/to/src
python src/train.py --class-group bone-loss --annotations annotations.json --images /path/to/images --task segment
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for training
2. **Optimize Batch Size**: Balance between memory usage and training speed
3. **Data Augmentation**: Built-in YOLOv8 augmentations optimized for segmentation
4. **Multi-scale Training**: Improves segmentation at different resolutions

## Comparison with Other Models

| Feature | YOLOv12 (Detection) | YOLOv8-Seg | UNet-Trans |
|---------|-------------------|------------|------------|
| Output | Bounding boxes | Instance masks | Dense segmentation |
| Speed | Fastest | Fast | Moderate |
| Precision | Good | Excellent | Excellent |
| Clinical Use | Screening | Measurement | Analysis |

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive logging for new features
3. Update documentation for API changes
4. Test with different class groups and X-ray types
5. Validate segmentation quality with clinical experts

## License

This project is part of the CEJ-AC detection system for dental landmark analysis.

## Future Enhancements

- [ ] **3D Segmentation**: Support for CBCT volumetric data
- [ ] **Real-time Inference**: Optimized models for clinical workflow
- [ ] **Active Learning**: Iterative annotation improvement
- [ ] **Multi-modal Fusion**: Combine with clinical metadata
- [ ] **Uncertainty Quantification**: Confidence estimation for predictions
