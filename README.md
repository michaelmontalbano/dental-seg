# Dental Implant Computer Vision

A comprehensive computer vision repository for dental implant detection, classification, and analysis using various deep learning architectures.

## Overview

This repository contains multiple computer vision models and pipelines for analyzing dental radiographs, with a focus on:
- Dental implant detection and segmentation
- Implant manufacturer classification
- Panoramic vs non-panoramic radiograph analysis
- Multi-stage detection pipelines
- Pseudo-labeling for dataset expansion

## Project Structure

```
computer-vision/
├── implant-cv/          # Advanced, core implant detection models, vision transformers, medmamba, etc. 
├── yolov8-seg/          # YOLOv8 segmentation models
├── yolo2stage/          # Two-stage YOLO detection pipeline
├── medsam/              # Medical SAM implementation
├── unet-trans/          # UNet with transformer architecture
├── unet-trans2/         # Enhanced UNet-transformer model
├── panoramic_vs_non_panoramic/  # Radiograph type classification
├── bw_vs_pans/          # Bitewing vs panoramic analysis
├── bbox_visualizations/ # Bounding box visualization tools
├── pseudo_labeling_pipeline.py  # Automated labeling pipeline
└── requirements.txt     # Python dependencies
```

## Models

### 1. YOLOv8 Models
- **yolov8-seg**: Instance segmentation for dental implants
- **yolo2stage**: Two-stage detection pipeline for improved accuracy

### 2. Transformer-based Models
- **unet-trans**: UNet architecture enhanced with transformer blocks
- **medsam**: Medical Segment Anything Model adaptation

### 3. Specialized Models
- **implant-cv**: Core implant detection and classification
- **panoramic_vs_non_panoramic**: Classifier for radiograph types

## Installation

```bash
# Clone the repository
git clone https://github.com/michaelmontalbano/computer-vision.git
cd computer-vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Key dependencies include:
- PyTorch >= 2.0.0
- Ultralytics >= 8.3.0 (for YOLOv8)
- Albumentations >= 1.3.0 (for data augmentation)
- AWS SageMaker SDK (for cloud training)
- OpenCV-Python
- Transformers libraries (timm, einops)

## Usage

### Pseudo-Labeling Pipeline

Generate predictions on unlabeled data for manual review:

```bash
python pseudo_labeling_pipeline.py \
    --model-s3 s3://your-bucket/model.tar.gz \
    --confidence 0.6 \
    --output-json pseudo_predictions.json
```

### Training Models on SageMaker

Most models include SageMaker launchers for cloud training:

```bash
cd yolov8-seg
python sagemaker_launcher.py
```

### Local Training

For local development and testing:

```bash
cd implant-cv
python train.py --data data.yaml --epochs 100
```

## Data Format

The repository supports multiple data formats:
- YOLO format (for detection/segmentation)
- Label Studio JSON format
- Custom JSON formats for specific tasks

## Features

### 1. Multi-Architecture Support
- YOLOv8 for real-time detection
- Transformer models for high accuracy
- Medical-specific architectures (MedSAM)

### 2. AWS SageMaker Integration
- Distributed training support
- Hyperparameter optimization
- Model versioning and deployment

### 3. Data Augmentation
- Dental-specific augmentations
- X-ray image enhancements
- Geometric and photometric transforms

### 4. Evaluation Tools
- Visualization utilities
- Performance metrics
- Model comparison tools

## Workflow

1. **Data Preparation**: Organize radiographs and annotations
2. **Model Selection**: Choose appropriate architecture for your task
3. **Training**: Use SageMaker launchers or local training scripts
4. **Evaluation**: Analyze results with visualization tools
5. **Deployment**: Export models for production use

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -am 'Add new model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Create a Pull Request

## License

This project is proprietary. Please contact the repository owner for licensing information.

## Contact

For questions or support, please open an issue in the repository.
