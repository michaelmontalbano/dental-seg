# MedSAM Dental Segmentation

A specialized implementation of MedSAM (Medical Segment Anything Model) for precise segmentation of dental landmarks including CEJ (Cemento-Enamel Junction), Alveolar Crest, and Apex regions in dental X-ray images.

## Overview

MedSAM is a foundation model for medical image segmentation that adapts the Segment Anything Model (SAM) for medical imaging tasks. This implementation specifically targets dental landmark segmentation with enhanced performance for clinical dental analysis.

## Features

- **Foundation Model**: Built on Meta's SAM with medical domain adaptation
- **Prompt-based Segmentation**: Support for point, box, and mask prompts
- **High Precision**: Optimized for fine-grained dental landmark boundaries
- **Interactive Segmentation**: Real-time refinement with user prompts
- **Multi-modal Input**: Support for various dental imaging modalities
- **Clinical Integration**: Designed for dental workflow integration

## Architecture

```
MedSAM Architecture:
├── Image Encoder (ViT-H)     # Vision Transformer backbone
├── Prompt Encoder            # Point/Box/Mask prompt processing
├── Mask Decoder             # Lightweight mask prediction head
└── Medical Adaptation       # Domain-specific fine-tuning
```

## Project Structure

```
medsam/
├── src/
│   ├── train.py              # Main training script
│   ├── dataset.py            # Dataset processing with prompt generation
│   ├── model.py              # MedSAM model implementation
│   ├── prompt_generator.py   # Automatic prompt generation
│   ├── evaluate.py           # Model evaluation and metrics
│   ├── inference.py          # Interactive inference pipeline
│   ├── utils.py              # Utility functions
│   └── requirements.txt      # Python dependencies
├── sagemaker_launcher.py     # SageMaker job launcher
├── config/
│   ├── model_config.yaml     # Model configuration
│   └── training_config.yaml  # Training parameters
├── notebooks/
│   └── interactive_demo.ipynb # Interactive segmentation demo
└── README.md                 # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r src/requirements.txt

# Download MedSAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth
```

### Local Training

```bash
python src/train.py \
    --data-path /path/to/dental/data \
    --annotations train.json \
    --class-group bone-loss \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-4
```

### Interactive Inference

```bash
python src/inference.py \
    --model-path checkpoints/medsam_dental.pth \
    --image-path /path/to/xray.jpg \
    --interactive
```

## Training Modes

### 1. Automatic Prompt Training
```bash
python src/train.py \
    --prompt-mode auto \
    --num-prompts 5 \
    --prompt-noise 0.1
```

### 2. Manual Prompt Training
```bash
python src/train.py \
    --prompt-mode manual \
    --prompt-file prompts.json
```

### 3. Mixed Prompt Training
```bash
python src/train.py \
    --prompt-mode mixed \
    --auto-ratio 0.7 \
    --manual-ratio 0.3
```

## Prompt Types

### Point Prompts
- **Positive points**: Inside target regions
- **Negative points**: Outside target regions
- **Multi-point**: Combination for complex shapes

### Box Prompts
- **Bounding boxes**: Rough region indication
- **Tight boxes**: Precise boundary hints
- **Multiple boxes**: For multiple instances

### Mask Prompts
- **Coarse masks**: Low-resolution guidance
- **Partial masks**: Incomplete annotations
- **Previous predictions**: Iterative refinement

## Class Groups

### bone-loss (Primary Focus)
- `apex`: Root apex regions
- `cej mesial`: CEJ mesial boundaries
- `cej distal`: CEJ distal boundaries
- `ac mesial`: Alveolar crest mesial
- `ac distal`: Alveolar crest distal

### conditions
- `decay`: Carious lesions
- `filling`: Restorative materials
- `crown`: Prosthetic crowns
- `implant`: Dental implants

### teeth
- `T01-T32`: Individual tooth segmentation
- `root`: Root structure segmentation
- `crown`: Crown structure segmentation

## Configuration

### Model Parameters
```yaml
model:
  backbone: "vit_h"
  checkpoint: "checkpoints/sam_vit_h.pth"
  freeze_image_encoder: false
  freeze_prompt_encoder: true
  mask_threshold: 0.5
  
training:
  epochs: 50
  batch_size: 4
  learning_rate: 1e-4
  weight_decay: 1e-4
  warmup_epochs: 5
```

### Prompt Generation
```yaml
prompts:
  num_points: 5
  num_boxes: 2
  point_noise: 0.1
  box_noise: 0.05
  negative_point_ratio: 0.2
```

## SageMaker Training

```bash
python sagemaker_launcher.py \
    --class-group bone-loss \
    --instance-type ml.g4dn.2xlarge \
    --epochs 50 \
    --batch-size 4 \
    --prompt-mode auto
```

## Evaluation Metrics

### Segmentation Quality
- **IoU (Intersection over Union)**: Overlap accuracy
- **Dice Coefficient**: Segmentation similarity
- **Boundary F1**: Edge precision
- **Hausdorff Distance**: Maximum boundary error

### Clinical Metrics
- **CEJ-AC Distance**: Bone loss measurement accuracy
- **Area Calculation**: Lesion size precision
- **Boundary Smoothness**: Clinical usability

### Prompt Efficiency
- **Clicks to Convergence**: Interactive efficiency
- **Prompt Sensitivity**: Robustness to input variation
- **Refinement Quality**: Iterative improvement

## Interactive Usage

### Command Line Interface
```bash
# Start interactive session
python src/inference.py --interactive --model checkpoints/medsam_dental.pth

# Commands:
# - Click: Add positive point
# - Right-click: Add negative point
# - Drag: Draw bounding box
# - 'r': Reset prompts
# - 's': Save segmentation
# - 'q': Quit
```

### Python API
```python
from src.model import MedSAM
from src.inference import InteractiveSegmenter

# Load model
model = MedSAM.from_pretrained("checkpoints/medsam_dental.pth")
segmenter = InteractiveSegmenter(model)

# Segment with points
points = [(100, 150), (120, 180)]  # (x, y) coordinates
labels = [1, 1]  # 1 for positive, 0 for negative
mask = segmenter.segment_with_points(image, points, labels)

# Segment with box
box = [50, 50, 200, 200]  # [x1, y1, x2, y2]
mask = segmenter.segment_with_box(image, box)
```

## Advanced Features

### Multi-Scale Segmentation
```python
# Segment at multiple resolutions
masks = segmenter.multiscale_segment(
    image, 
    scales=[0.5, 1.0, 1.5],
    prompts=points
)
```

### Uncertainty Estimation
```python
# Get segmentation uncertainty
mask, uncertainty = segmenter.segment_with_uncertainty(
    image, 
    prompts,
    num_samples=10
)
```

### Active Learning
```python
# Suggest next annotation points
next_points = segmenter.suggest_points(
    image, 
    current_mask,
    strategy="entropy"
)
```

## Performance Optimization

### Memory Efficiency
- **Gradient Checkpointing**: Reduce memory usage
- **Mixed Precision**: FP16 training
- **Patch-based Processing**: Handle large images

### Speed Optimization
- **TensorRT**: GPU acceleration
- **ONNX Export**: Cross-platform deployment
- **Quantization**: Model compression

## Clinical Integration

### DICOM Support
```python
from src.utils import load_dicom

# Load DICOM image
image, metadata = load_dicom("path/to/xray.dcm")
mask = segmenter.segment(image, prompts)
```

### Measurement Tools
```python
from src.clinical import measure_cej_ac_distance

# Clinical measurements
distance = measure_cej_ac_distance(
    cej_mask, 
    ac_mask, 
    pixel_spacing=metadata['PixelSpacing']
)
```

## Comparison with Other Models

| Feature | MedSAM | YOLOv8-Seg | UNet-Trans |
|---------|--------|------------|------------|
| Interactivity | ✅ Excellent | ❌ None | ❌ None |
| Precision | ✅ Very High | ✅ High | ✅ High |
| Speed | ⚠️ Moderate | ✅ Fast | ⚠️ Moderate |
| Flexibility | ✅ Very High | ⚠️ Limited | ⚠️ Limited |
| Training Data | ⚠️ Moderate | ✅ Large | ✅ Large |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use gradient checkpointing
   python src/train.py --batch-size 2 --gradient-checkpointing
   ```

2. **Slow Inference**
   ```bash
   # Use smaller image size or TensorRT
   python src/inference.py --image-size 512 --use-tensorrt
   ```

3. **Poor Segmentation Quality**
   ```bash
   # Adjust prompt strategy or fine-tune longer
   python src/train.py --prompt-mode mixed --epochs 100
   ```

## Research Applications

### Longitudinal Studies
- Track bone loss progression over time
- Quantify treatment effectiveness
- Population-level dental health analysis

### AI-Assisted Diagnosis
- Interactive lesion delineation
- Treatment planning support
- Educational tool for dental students

### Dataset Creation
- Semi-automatic annotation
- Quality control for existing datasets
- Rapid prototyping for new classes

## Future Enhancements

- [ ] **3D MedSAM**: CBCT volume segmentation
- [ ] **Real-time Processing**: Live clinical integration
- [ ] **Multi-modal Fusion**: Combine with clinical data
- [ ] **Federated Learning**: Privacy-preserving training
- [ ] **Mobile Deployment**: Point-of-care applications

## Citation

```bibtex
@article{medsam2023,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  year={2024}
}
```

## License

This project adapts MedSAM for dental applications. Please refer to the original MedSAM license for usage terms.

## Contributing

1. Follow medical imaging best practices
2. Validate clinical relevance with dental experts
3. Ensure patient privacy compliance
4. Test interactive functionality thoroughly
5. Document prompt engineering strategies
