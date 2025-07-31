#!/usr/bin/env python3
"""
Inspect the model format to understand its structure
"""

import boto3
import torch
import tempfile
import json
from pathlib import Path

def inspect_s3_model(s3_path):
    """Download and inspect the model structure"""
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    s3_parts = s3_path.replace('s3://', '').split('/')
    bucket = s3_parts[0]
    key = '/'.join(s3_parts[1:])
    
    print(f"Downloading model from: {s3_path}")
    
    # Download model
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        
        print(f"Model downloaded to: {tmp.name}")
        
        # Load and inspect
        try:
            # Try with weights_only=False first (for newer PyTorch models)
            model_data = torch.load(tmp.name, map_location='cpu', weights_only=False)
            print("\n🔍 MODEL STRUCTURE ANALYSIS:")
            print("=" * 50)
            
            print(f"Model type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Dictionary keys: {list(model_data.keys())}")
                
                # Check for our expected structure
                expected_keys = ['model_state_dict', 'class_names', 'num_classes', 'model_name']
                has_expected = all(key in model_data for key in expected_keys)
                
                if has_expected:
                    print("✅ MODEL HAS EXPECTED CLASSIFICATION FORMAT!")
                    print(f"  Model name: {model_data.get('model_name', 'unknown')}")
                    print(f"  Classes: {model_data.get('class_names', [])}")
                    print(f"  Number of classes: {model_data.get('num_classes', 'unknown')}")
                    print(f"  Image size: {model_data.get('image_size', 'unknown')}")
                    print(f"  Final accuracy: {model_data.get('final_accuracy', 'unknown')}")
                    
                    if 'model_state_dict' in model_data:
                        state_dict = model_data['model_state_dict']
                        print(f"  State dict parameters: {len(state_dict)}")
                        print(f"  Sample keys: {list(state_dict.keys())[:3]}...")
                else:
                    print("⚠️  Model structure doesn't match expected format")
                    # Check common keys anyway
                    for key in ['model', 'state_dict', 'model_state_dict', 'epoch', 'class_names', 'classes', 'names']:
                        if key in model_data:
                            value = model_data[key]
                            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            
            return model_data
            
        except Exception as e:
            print(f"❌ Error loading with weights_only=False: {e}")
            
            # Try with weights_only=True as fallback
            try:
                print("\n🔄 Trying with weights_only=True...")
                model_data = torch.load(tmp.name, map_location='cpu', weights_only=True)
                print(f"✅ Loaded as weights-only. Type: {type(model_data)}")
                if isinstance(model_data, dict):
                    print(f"Keys: {list(model_data.keys())[:10]}")
                return model_data
            except Exception as e2:
                print(f"❌ Also failed with weights_only=True: {e2}")
                return None

def create_compatible_model_structure(original_model, target_path):
    """Create a compatible model structure for our classifier"""
    
    # This is where we'd convert your model to the expected format
    # The exact conversion depends on what we find in your model
    
    compatible_structure = {
        'model_state_dict': None,  # Will be filled based on inspection
        'model_name': 'resnet50',  # Default, may need adjustment
        'class_names': ['bitewing', 'periapical', 'panoramic'],
        'num_classes': 3,
        'image_size': 224,
        'final_accuracy': 0.995  # Your reported accuracy
    }
    
    print(f"\n📝 Would create compatible model at: {target_path}")
    print("Compatible structure keys:", list(compatible_structure.keys()))
    
    return compatible_structure

def main():
    # Check both local and S3 models
    local_model_path = "./model.pth"
    s3_path = "s3://codentist-general/modeling/bw_pa_pans_classifier/best.pt"
    
    print("🔍 INSPECTING MODEL FORMAT")
    print("=" * 60)
    
    # First check local model if it exists
    if Path(local_model_path).exists():
        print(f"📁 Found local model: {local_model_path}")
        print("Inspecting local model...")
        
        try:
            model_data = torch.load(local_model_path, map_location='cpu', weights_only=False)
            print("\n�� LOCAL MODEL STRUCTURE ANALYSIS:")
            print("=" * 50)
            
            print(f"Model type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Dictionary keys: {list(model_data.keys())}")
                
                # Check for our expected structure
                expected_keys = ['model_state_dict', 'class_names', 'num_classes', 'model_name']
                has_expected = all(key in model_data for key in expected_keys)
                
                if has_expected:
                    print("✅ MODEL HAS EXPECTED CLASSIFICATION FORMAT!")
                    print(f"  Model name: {model_data.get('model_name', 'unknown')}")
                    print(f"  Classes: {model_data.get('class_names', [])}")
                    print(f"  Number of classes: {model_data.get('num_classes', 'unknown')}")
                    print(f"  Image size: {model_data.get('image_size', 'unknown')}")
                    print(f"  Final accuracy: {model_data.get('final_accuracy', 'unknown')}")
                    
                    if 'model_state_dict' in model_data:
                        state_dict = model_data['model_state_dict']
                        print(f"  State dict parameters: {len(state_dict)}")
                        print(f"  Sample keys: {list(state_dict.keys())[:3]}...")
                    
                    print(f"\n🎉 This model is ready to use for classification!")
                    return model_data
                else:
                    print("⚠️  Model structure doesn't match expected format")
                    # Check common keys anyway
                    for key, value in model_data.items():
                        print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
    
    else:
        print(f"❌ Local model not found: {local_model_path}")
        
        # Inspect S3 model as fallback
        model_data = inspect_s3_model(s3_path)
        
        if model_data:
            print("\n✅ S3 model loaded successfully!")
            
            # Determine if we can use it directly or need conversion
            expected_keys = ['model_state_dict', 'class_names', 'num_classes']
            has_expected_format = all(key in model_data for key in expected_keys)
            
            if has_expected_format:
                print("✅ Model has expected format - can use directly")
            else:
                print("⚠️  Model needs format conversion")
                print("\nSuggested solutions:")
                print("1. Convert the model to expected format")
                print("2. Modify the classifier to handle this format")
                print("3. Use the original training script's model loading")
            
            # Show what we'd need to create
            compatible_path = "s3://codentist-general/modeling/bw_pa_pans_classifier/model_compatible.pth"
            create_compatible_model_structure(model_data, compatible_path)
        
        else:
            print("❌ Could not inspect model")

if __name__ == '__main__':
    main()
