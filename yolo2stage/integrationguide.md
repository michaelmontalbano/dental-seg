# Using Your Existing T01-T32 Model with Dual YOLO

Perfect! Your existing T01-T32 teeth detection model is ideal for this approach. Here's how to integrate it:

## Quick Integration

### 1. Use Your Model as Stage 1 (Done!)
```python
# Your existing model becomes Stage 1
teeth_model = YOLO("your_t01_t32_model.pt")  # ← Your existing model

# Train only Stage 2 (landmark segmentation)  
landmark_model = YOLO("yolov8n-seg.pt")  # ← New model to train
```

### 2. Train Only the Landmark Segmentation Model
Since you already have teeth detection, you only need to train the landmark model:

```bash
# Prepare landmark segmentation dataset using your existing teeth detections
python prepare_landmark_dataset.py \
    --teeth-model your_t01_t32_model.pt \
    --annotations annotations.json \
    --images images/ \
    --output landmark_seg_dataset

# Train landmark segmentation model  
python sagemaker_launcher.py \
    --data landmark_seg_dataset/data.yaml \
    --class-group bone-loss \
    --model-size n \
    --epochs 200 \
    --seed 3407
```

### 3. Use the Dual Pipeline
```python
from existing_teeth_dual_pipeline import ExistingTeethDualPipeline

# Initialize with your existing model + new landmark model
pipeline = ExistingTeethDualPipeline(
    existing_teeth_model_path="your_t01_t32_model.pt",      # Your model
    landmark_segmentation_model_path="landmarks_seg.pt"      # New model
)

# Run prediction
results = pipeline.predict(image)

# Get clinical results
print(f"Found landmarks in teeth: {list(results['clinical_summary']['teeth_with_landmarks'].keys())}")
```

## Data Preparation Script

Create `prepare_landmark_dataset.py`:

```python
#!/usr/bin/env python3
"""
Prepare landmark segmentation dataset using your existing T01-T32 model
"""

import cv2
import json
from ultralytics import YOLO
from pathlib import Path

def create_landmark_crops_with_existing_model(
    teeth_model_path: str,
    annotations_file: str, 
    images_dir: str,
    output_dir: str = "landmark_seg_dataset"
):
    """Use your existing T01-T32 model to create landmark training crops"""
    
    # Load your existing teeth model
    teeth_model = YOLO(teeth_model_path)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each image
    crop_count = 0
    
    for img_info in annotations['images']:
        image_path = Path(images_dir) / img_info['file_name']
        image = cv2.imread(str(image_path))
        
        if image is None:
            continue
        
        # Use your existing model to detect teeth
        teeth_results = teeth_model(image, conf=0.3)
        
        if not teeth_results or len(teeth_results[0].boxes) == 0:
            continue
        
        # For each detected tooth, create a crop
        for box in teeth_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            tooth_name = f"T{cls+1:02d}"  # Or however your model maps classes
            
            # Expand crop for context
            width, height = x2-x1, y2-y1
            expand = 0.2
            crop_x1 = max(0, int(x1 - width*expand))
            crop_y1 = max(0, int(y1 - height*expand))  
            crop_x2 = min(image.shape[1], int(x2 + width*expand))
            crop_y2 = min(image.shape[0], int(y2 + height*expand))
            
            # Extract crop
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save crop
            split = 'train' if crop_count % 5 != 0 else 'val'
            crop_filename = f"{Path(img_info['file_name']).stem}_{tooth_name}_crop.jpg"
            crop_path = output_path / split / 'images' / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            # Create landmark labels for this crop (from your original annotations)
            # ... (convert landmark annotations to crop coordinates)
            
            crop_count += 1
    
    print(f"✅ Created {crop_count} tooth crops for landmark training")

if __name__ == "__main__":
    create_landmark_crops_with_existing_model(
        teeth_model_path="your_t01_t32_model.pt",
        annotations_file="annotations.json", 
        images_dir="images/",
        output_dir="landmark_seg_dataset"
    )
```

## Expected Clinical Output

With your T01-T32 model, you'll get clinically meaningful results:

```json
{
  "clinical_summary": {
    "teeth_with_landmarks": {
      "T06": ["cej mesial", "cej distal", "ac mesial", "ac distal"],
      "T07": ["cej mesial", "ac mesial", "apex"],
      "T08": ["cej distal", "ac distal"]
    },
    "landmark_counts": {
      "cej mesial": 15,
      "cej distal": 12, 
      "ac mesial": 18,
      "ac distal": 16,
      "apex": 8
    },
    "complete_teeth": ["T06", "T11", "T14"],
    "missing_landmarks": {
      "T07": ["cej distal", "ac distal"],
      "T08": ["cej mesial", "ac mesial", "apex"]
    }
  }
}
```

## Performance Benefits

Since you already have accurate T01-T32 detection:

- **Higher Accuracy**: Landmarks constrained to correct teeth
- **Clinical Relevance**: "T06 has bone loss at mesial CEJ" 
- **Less Training**: Only need to train landmark segmentation
- **Better Evaluation**: Can measure per-tooth performance

## Next Steps

1. **Test your existing model**: Make sure it works well on your X-ray types
2. **Prepare landmark dataset**: Use the script above to create tooth crops
3. **Train landmark model**: Focus on CEJ, AC, apex segmentation within crops
4. **Integrate**: Use the dual pipeline for inference

Your existing T01-T32 model gives you a huge head start - you're already 50% done with the dual approach!

## Quick Test

Want to test this quickly? Try this:

```python
# Test your existing model + simple landmark detection
teeth_model = YOLO("your_t01_t32_model.pt")
image = cv2.imread("test_xray.jpg")

# Get teeth detections
results = teeth_model(image)
print(f"Detected {len(results[0].boxes)} teeth")

# Extract and save crops
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cls = int(box.cls[0].cpu().numpy())
    
    # Save tooth crop
    crop = image[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite(f"tooth_T{cls+1:02d}_crop.jpg", crop)

print("✅ Tooth crops saved - ready for landmark segmentation!")
```

This shows exactly how your existing model crops will look for landmark training.
