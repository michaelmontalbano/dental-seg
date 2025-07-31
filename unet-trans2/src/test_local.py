#!/usr/bin/env python3
"""
Test the dataset and model locally before SageMaker training
"""

import os
import torch
import boto3
from torch.utils.data import DataLoader
import sys

# Import our modules
from dataset import BboxDataset, collate_fn
from train import get_detection_model, get_transform

def download_sample_data():
    """Download a subset of data for local testing"""
    print("📥 Downloading sample data...")
    
    s3 = boto3.client('s3')
    
    # Create local directories
    os.makedirs('test_data/images', exist_ok=True)
    
    try:
        # Download annotations
        s3.download_file('codentist-general', 'datasets/master/annotations.json', 'test_data/annotations.json')
        print("✅ Downloaded annotations.json")
        
        # Download a few sample images
        print("📥 Downloading sample images...")
        
        # List some images from S3
        response = s3.list_objects_v2(
            Bucket='codentist-general',
            Prefix='datasets/master/images/',
            MaxKeys=10
        )
        
        downloaded_images = 0
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith(('.jpg', '.jpeg', '.png')) and downloaded_images < 5:
                filename = os.path.basename(key)
                local_path = f'test_data/images/{filename}'
                
                try:
                    s3.download_file('codentist-general', key, local_path)
                    print(f"✅ Downloaded {filename}")
                    downloaded_images += 1
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
        
        print(f"✅ Downloaded {downloaded_images} sample images")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading and data pipeline"""
    print("\n🧪 TESTING DATASET LOADING")
    print("=" * 50)
    
    try:
        # Test dataset creation
        print("1️⃣ Creating dataset...")
        dataset = BboxDataset(
            annotations_file='test_data/annotations.json',
            images_dir='test_data/images',
            transforms=get_transform(train=True, image_size=512)
        )
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Dataset size: {len(dataset)} items")
        
        if len(dataset) == 0:
            print("❌ No items in dataset - check data filtering")
            return False
        
        # Test getting first item
        print("\n2️⃣ Testing data loading...")
        try:
            image, target = dataset[0]
            print(f"✅ Successfully loaded first item")
            print(f"📸 Image shape: {image.shape}")
            print(f"📋 Target keys: {list(target.keys())}")
            print(f"📦 Boxes shape: {target['boxes'].shape}")
            print(f"🏷️  Labels: {target['labels']}")
            print(f"💯 Scores available: {'scores' in target}")
            
        except Exception as e:
            print(f"❌ Error loading first item: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test DataLoader
        print("\n3️⃣ Testing DataLoader...")
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0  # Avoid multiprocessing issues locally
            )
            
            # Get first batch
            batch = next(iter(dataloader))
            images, targets = batch
            
            print(f"✅ DataLoader working")
            print(f"📦 Batch images: {len(images)} items")
            print(f"📦 Batch targets: {len(targets)} items")
            print(f"📸 First image shape: {images[0].shape}")
            
        except Exception as e:
            print(f"❌ DataLoader error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print("\n🧪 TESTING MODEL")
    print("=" * 50)
    
    try:
        # Create model
        print("1️⃣ Creating model...")
        model = get_detection_model(num_classes=5, pretrained=True)
        print("✅ Model created successfully")
        
        # Test model architecture
        print("2️⃣ Testing model architecture...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🧪 TESTING TRAINING STEP")
    print("=" * 50)
    
    try:
        # Create dataset and model
        dataset = BboxDataset(
            annotations_file='test_data/annotations.json',
            images_dir='test_data/images',
            transforms=get_transform(train=True, image_size=512)
        )
        
        if len(dataset) == 0:
            print("❌ No data for training test")
            return False
        
        model = get_detection_model(num_classes=5, pretrained=True)
        model.train()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Small batch for testing
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Get a batch
        images, targets = next(iter(dataloader))
        
        print(f"✅ Got batch: {len(images)} images")
        
        # Test forward pass (training mode)
        print("1️⃣ Testing training forward pass...")
        device = torch.device('cpu')  # Use CPU for local testing
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model = model.to(device)
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        print(f"✅ Training forward pass successful")
        print(f"📊 Loss components: {list(loss_dict.keys())}")
        print(f"💰 Total loss: {losses.item():.4f}")
        
        # Test inference mode
        print("2️⃣ Testing inference mode...")
        model.eval()
        with torch.no_grad():
            predictions = model(images)
            
        print(f"✅ Inference successful")
        print(f"📦 Predictions: {len(predictions)} items")
        if len(predictions) > 0:
            pred = predictions[0]
            print(f"🎯 Prediction keys: {list(pred.keys())}")
            print(f"📦 Boxes found: {len(pred['boxes'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 LOCAL TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Download data
    if not download_sample_data():
        print("❌ Failed to download data - aborting tests")
        return
    
    # Test 2: Dataset loading
    if not test_dataset_loading():
        print("❌ Dataset loading failed - aborting further tests")
        return
    
    # Test 3: Model creation
    if not test_model_creation():
        print("❌ Model creation failed - aborting training test")
        return
    
    # Test 4: Training step
    if not test_training_step():
        print("❌ Training step failed")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Ready for SageMaker training")
    print("\n💡 To run SageMaker training:")
    print("python sagemaker_launcher.py --region us-west-2 --train-data s3://codentist-general/datasets/master --num-classes 5 --epochs 50 --batch-size 4 --instance-type ml.g6.12xlarge --wait")

if __name__ == "__main__":
    main()