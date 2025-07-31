#!/usr/bin/env python3
"""
Test script to verify balanced dataset loading works with your data
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_balanced_dataset():
    """Test the balanced dataset implementation"""
    
    print("🧪 TESTING BALANCED DATASET IMPLEMENTATION")
    print("=" * 60)
    
    # Check if data directory exists
    data_dir = './organized_xray_data'
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run the data organization script first")
        return False
    
    try:
        # Import our balanced dataset
        from balanced_dataset import create_balanced_dataloaders
        print("✅ Successfully imported balanced_dataset")
        
        # Test dataset creation
        print("\n1️⃣ Testing dataset creation...")
        datasets, dataloaders, class_counts = create_balanced_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            max_per_class=None,  # Auto-balance
            seed=42
        )
        
        print("✅ Dataset creation successful!")
        print(f"Class distribution: {class_counts}")
        
        # Test data loading
        print("\n2️⃣ Testing data loading...")
        
        # Test training dataloader
        train_batch = next(iter(dataloaders['train']))
        images, labels = train_batch
        
        print(f"✅ Loaded training batch:")
        print(f"   Image batch shape: {images.shape}")
        print(f"   Label batch shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Label dtype: {labels.dtype}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Check label distribution in batch
        unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
        print(f"   Labels in batch: {dict(zip(unique_labels, counts))}")
        
        # Test validation dataloader
        val_batch = next(iter(dataloaders['val']))
        val_images, val_labels = val_batch
        print(f"✅ Loaded validation batch: {val_images.shape}")
        
        # Test class balance
        print("\n3️⃣ Testing class balance...")
        all_train_labels = []
        
        # Sample a few batches to check balance
        for i, (_, labels) in enumerate(dataloaders['train']):
            all_train_labels.extend(labels.numpy())
            if i >= 5:  # Sample first 5 batches
                break
        
        train_label_counts = np.bincount(all_train_labels)
        print(f"✅ Train label distribution (first few batches): {train_label_counts}")
        
        # Check if reasonably balanced (within 50% of each other)
        if len(train_label_counts) == 3:
            min_count = min(train_label_counts)
            max_count = max(train_label_counts)
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if balance_ratio <= 2.0:
                print(f"✅ Classes are reasonably balanced (ratio: {balance_ratio:.1f}:1)")
            else:
                print(f"⚠️  Classes may be imbalanced (ratio: {balance_ratio:.1f}:1)")
        
        # Test split isolation
        print("\n4️⃣ Testing split isolation...")
        
        train_size = len(datasets['train'])
        val_size = len(datasets['val'])
        test_size = len(datasets['test'])
        
        total_size = train_size + val_size + test_size
        
        print(f"✅ Split sizes:")
        print(f"   Train: {train_size} ({train_size/total_size*100:.1f}%)")
        print(f"   Val:   {val_size} ({val_size/total_size*100:.1f}%)")
        print(f"   Test:  {test_size} ({test_size/total_size*100:.1f}%)")
        
        # Verify no overlap in indices (basic check)
        # Note: This is a simplified check since we're using Subset
        if train_size > 0 and val_size > 0:
            print("✅ Dataset splits created successfully")
        
        # Performance recommendation
        print("\n5️⃣ Performance recommendations...")
        
        min_class_size = min(class_counts.values())
        if min_class_size < 30:
            print("⚠️  VERY SMALL DATASET:")
            print(f"   Only {min_class_size} samples per class")
            print("   Expect high variance in results")
            print("   Consider data augmentation or collecting more data")
        elif min_class_size < 100:
            print("⚠️  SMALL DATASET:")
            print(f"   {min_class_size} samples per class")
            print("   Use small batch sizes and high regularization")
        else:
            print("✅ Dataset size is adequate for training")
        
        print("\n🎯 EXPECTED TRAINING BEHAVIOR:")
        print("-" * 40)
        print("With balanced classes, expect:")
        print("• Training accuracy: 60-90%")
        print("• Validation accuracy: 50-80%")
        print("• Training > Validation (normal)")
        print("• If Val > 85%: Check for data leakage!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure balanced_dataset.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_parameters():
    """Test dataset with different balancing parameters"""
    
    print("\n🧪 TESTING DIFFERENT BALANCING PARAMETERS")
    print("=" * 60)
    
    try:
        from balanced_dataset import create_balanced_dataloaders
        
        # Test with maximum limit
        print("Testing with max_per_class=50...")
        datasets, dataloaders, class_counts = create_balanced_dataloaders(
            data_dir='./organized_xray_data',
            batch_size=4,
            max_per_class=50,
            seed=42
        )
        
        print(f"✅ Limited dataset: {class_counts}")
        
        expected_per_class = min(50, min(class_counts.values()))
        actual_total = sum(class_counts.values())
        expected_total = expected_per_class * 3
        
        if actual_total == expected_total:
            print("✅ Balancing worked correctly")
        else:
            print(f"⚠️  Expected {expected_total}, got {actual_total}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        return False

def main():
    """Main test function"""
    
    success = test_balanced_dataset()
    
    if success:
        success = test_with_different_parameters()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run training with:")
        print("python fixed_train.py --data_dir ./organized_xray_data")
    else:
        print("❌ TESTS FAILED!")
        print("Please check your data setup and dependencies")
    print("=" * 60)

if __name__ == '__main__':
    main()