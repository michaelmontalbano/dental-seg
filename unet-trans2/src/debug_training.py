#!/usr/bin/env python3
"""
Debug script to analyze training pipeline step by step
This will help us identify exactly where things are failing
"""

import os
import json
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import traceback
import time

# Import our dataset
try:
    from dataset import BboxDataset, collate_fn
    print("✅ Successfully imported BboxDataset and collate_fn")
except Exception as e:
    print(f"❌ FAILED to import dataset: {e}")
    print(traceback.format_exc())
    exit(1)

def debug_step(step_name, func, *args, **kwargs):
    """Execute a step and report success/failure with timing"""
    print(f"\n{'='*60}")
    print(f"🔍 STEP: {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {step_name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {step_name} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

def check_file_exists(filepath):
    """Check if file exists and show size"""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ File exists: {filepath} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ File missing: {filepath}")
        return False

def analyze_annotations(annotations_file):
    """Deep dive into annotations structure"""
    print(f"🔍 Analyzing annotations file: {annotations_file}")
    
    if not check_file_exists(annotations_file):
        return None
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Annotations type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Dictionary keys: {list(data.keys())}")
        
        if 'categories' in data:
            categories = data['categories']
            print(f"🏷️  Found {len(categories)} categories:")
            for i, cat in enumerate(categories[:10]):  # Show first 10
                print(f"   {cat.get('id', 'no_id')}: {cat.get('name', 'no_name')}")
            if len(categories) > 10:
                print(f"   ... and {len(categories) - 10} more categories")
        
        if 'images' in data:
            print(f"🖼️  Found {len(data['images'])} images")
            
        if 'annotations' in data:
            annotations = data['annotations']
            print(f"📝 Found {len(annotations)} annotations")
            
            # Sample some annotations
            if annotations:
                print("📋 Sample annotations:")
                for i in range(min(3, len(annotations))):
                    ann = annotations[i]
                    print(f"   Annotation {i}: {ann}")
    
    return data

def create_debug_dataset(annotations_file, images_dir):
    """Create dataset with detailed logging"""
    print(f"🏗️  Creating dataset...")
    print(f"   Annotations: {annotations_file}")
    print(f"   Images dir: {images_dir}")
    
    if not check_file_exists(annotations_file):
        return None
        
    if not os.path.exists(images_dir):
        print(f"❌ Images directory missing: {images_dir}")
        return None
    
    # Count images in directory
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🖼️  Found {len(image_files)} image files in directory")
    
    try:
        dataset = BboxDataset(annotations_file, images_dir, transforms=None)
        print(f"✅ Dataset created successfully")
        print(f"📊 Dataset length: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        print(traceback.format_exc())
        return None

def test_dataset_samples(dataset, num_samples=5):
    """Test loading samples from dataset"""
    if dataset is None:
        return False
        
    print(f"🧪 Testing {num_samples} dataset samples...")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            print(f"\n📋 Testing sample {i}:")
            image, target = dataset[i]
            
            print(f"   Image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
            print(f"   Target keys: {list(target.keys()) if isinstance(target, dict) else type(target)}")
            
            if isinstance(target, dict):
                if 'boxes' in target:
                    boxes = target['boxes']
                    print(f"   Boxes shape: {boxes.shape}")
                    print(f"   Boxes range: min={boxes.min().item():.2f}, max={boxes.max().item():.2f}")
                    
                if 'labels' in target:
                    labels = target['labels']
                    print(f"   Labels shape: {labels.shape}")
                    print(f"   Labels: {labels.tolist()}")
                    print(f"   Label range: min={labels.min().item()}, max={labels.max().item()}")
                    
            print(f"   ✅ Sample {i} loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Sample {i} failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    return True

def create_debug_model(num_classes):
    """Create model with detailed logging"""
    print(f"🤖 Creating model with {num_classes} classes...")
    
    try:
        # Load pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        print("✅ Base model loaded")
        
        # Modify classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        total_classes = num_classes + 1  # +1 for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, total_classes)
        print(f"✅ Modified head for {total_classes} total classes")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✅ Model moved to device: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print(traceback.format_exc())
        return None, None

def test_dataloader(dataset, batch_size=2):
    """Test dataloader creation and batch loading"""
    if dataset is None:
        return None
        
    print(f"🔄 Creating dataloader with batch_size={batch_size}...")
    
    try:
        # Create train/val split
        indices = list(range(len(dataset)))
        if len(indices) < 10:
            print(f"⚠️  Dataset too small ({len(indices)} samples) for train/val split")
            train_indices = indices
        else:
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
            print(f"📊 Split: {len(train_indices)} train, {len(val_indices)} val")
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        
        dataloader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for debugging
            num_workers=0,  # Single threaded for debugging
            collate_fn=collate_fn
        )
        
        print(f"✅ Dataloader created, {len(dataloader)} batches")
        return dataloader
        
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        print(traceback.format_exc())
        return None

def test_training_step(model, dataloader, device):
    """Test a single training step"""
    if model is None or dataloader is None:
        return False
        
    print(f"🎯 Testing single training step...")
    
    try:
        model.train()
        
        # Get first batch
        batch_iter = iter(dataloader)
        images, targets = next(batch_iter)
        
        print(f"📦 Batch loaded:")
        print(f"   Batch size: {len(images)}")
        print(f"   Image shapes: {[img.shape for img in images]}")
        print(f"   Target count: {len(targets)}")
        
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        print(f"✅ Moved batch to device: {device}")
        
        # Validate targets
        for i, target in enumerate(targets):
            print(f"   Target {i}:")
            for key, value in target.items():
                if torch.is_tensor(value):
                    print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
                    if key == 'labels' and len(value) > 0:
                        print(f"     {key}: min={value.min().item()}, max={value.max().item()}")
                    elif key == 'boxes' and len(value) > 0:
                        print(f"     {key}: min={value.min().item():.3f}, max={value.max().item():.3f}")
                else:
                    print(f"     {key}: {value}")
        
        # Forward pass
        print("🚀 Attempting forward pass...")
        loss_dict = model(images, targets)
        
        print("✅ Forward pass successful!")
        print(f"📊 Losses: {loss_dict}")
        
        # Compute total loss
        total_loss = sum(loss for loss in loss_dict.values())
        print(f"📈 Total loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main debug pipeline"""
    print("🔍 DENTAL LANDMARK DETECTION - DEBUG PIPELINE")
    print("=" * 80)
    
    # Configuration
    annotations_file = "/opt/ml/input/data/train/annotations.json"
    images_dir = "/opt/ml/input/data/train/images"
    num_classes = 8
    batch_size = 2
    
    # Alternative local paths for testing
    if not os.path.exists(annotations_file):
        annotations_file = "./data/train/annotations.json"
        images_dir = "./data/train/images"
        print("🏠 Using local paths for debugging")
    
    print(f"📋 Configuration:")
    print(f"   Annotations: {annotations_file}")
    print(f"   Images: {images_dir}")
    print(f"   Classes: {num_classes}")
    print(f"   Batch size: {batch_size}")
    
    # Step 1: Analyze annotations file
    annotations_data = debug_step(
        "Analyze Annotations", 
        analyze_annotations, 
        annotations_file
    )
    
    if annotations_data is None:
        print("❌ Cannot proceed without annotations")
        return
    
    # Step 2: Create dataset
    dataset = debug_step(
        "Create Dataset",
        create_debug_dataset,
        annotations_file,
        images_dir
    )
    
    if dataset is None:
        print("❌ Cannot proceed without dataset")
        return
    
    # Step 3: Test dataset samples
    samples_ok = debug_step(
        "Test Dataset Samples",
        test_dataset_samples,
        dataset,
        5
    )
    
    if not samples_ok:
        print("❌ Dataset samples are broken")
        return
    
    # Step 4: Create model
    model, device = debug_step(
        "Create Model",
        create_debug_model,
        num_classes
    )
    
    if model is None:
        print("❌ Cannot proceed without model")
        return
    
    # Step 5: Test dataloader
    dataloader = debug_step(
        "Create DataLoader",
        test_dataloader,
        dataset,
        batch_size
    )
    
    if dataloader is None:
        print("❌ Cannot proceed without dataloader")
        return
    
    # Step 6: Test training step
    training_ok = debug_step(
        "Test Training Step",
        test_training_step,
        model,
        dataloader,
        device
    )
    
    # Final report
    print("\n" + "=" * 80)
    print("🎯 FINAL DEBUG REPORT")
    print("=" * 80)
    
    if training_ok:
        print("✅ ALL SYSTEMS GO! Training pipeline is working correctly.")
        print("🚀 You can proceed with full training.")
    else:
        print("❌ PIPELINE BROKEN! Check the failed steps above.")
        print("🔧 Fix the issues before attempting full training.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()