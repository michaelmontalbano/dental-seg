import torch
import json

def inspect_checkpoint(checkpoint_path):
    """Inspect the contents of a PyTorch checkpoint file"""
    print(f"📋 Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check what type of object it is
        checkpoint_type = type(checkpoint).__name__
        print(f"✅ Checkpoint type: {checkpoint_type}")
        
        if isinstance(checkpoint, dict):
            print(f"\n📦 Dictionary with {len(checkpoint)} keys:")
            for i, (key, value) in enumerate(checkpoint.items()):
                value_type = type(value).__name__
                
                # Special handling for different types
                if isinstance(value, torch.Tensor):
                    print(f"  [{i+1}] '{key}': {value_type} - shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  [{i+1}] '{key}': {value_type} - {len(value)} items")
                    # Show first few items if it's a small dict
                    if len(value) <= 5:
                        for k, v in value.items():
                            print(f"      • {k}: {v}")
                elif isinstance(value, (list, tuple)):
                    print(f"  [{i+1}] '{key}': {value_type} - {len(value)} items")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"  [{i+1}] '{key}': {value_type} - {value}")
                else:
                    print(f"  [{i+1}] '{key}': {value_type}")
            
            # If it has model_state_dict, show some layer info
            if 'model_state_dict' in checkpoint:
                print(f"\n🔍 Model state dict analysis:")
                state_dict = checkpoint['model_state_dict']
                print(f"   Total parameters: {len(state_dict)}")
                
                # Show first and last few layers
                layer_names = list(state_dict.keys())
                print(f"\n   First 5 layers:")
                for name in layer_names[:5]:
                    print(f"     • {name}: {state_dict[name].shape}")
                
                if len(layer_names) > 10:
                    print(f"\n   Last 5 layers:")
                    for name in layer_names[-5:]:
                        print(f"     • {name}: {state_dict[name].shape}")
                        
        elif isinstance(checkpoint, torch.nn.Module):
            print(f"\n⚠️  Checkpoint is a full model object (not recommended)")
            print(f"   Model class: {checkpoint.__class__.__name__}")
            
        else:
            print(f"\n❓ Unexpected checkpoint format")
            
            # Try to access as state_dict directly
            if hasattr(checkpoint, 'keys'):
                print(f"\n🔍 Attempting to treat as state_dict...")
                print(f"   Number of parameters: {len(checkpoint)}")
                
                # Show first few parameters
                param_names = list(checkpoint.keys())[:5]
                print(f"\n   First few parameters:")
                for name in param_names:
                    if isinstance(checkpoint[name], torch.Tensor):
                        print(f"     • {name}: {checkpoint[name].shape}")
                    else:
                        print(f"     • {name}: {type(checkpoint[name]).__name__}")
                        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    # Inspect the company model checkpoint
    checkpoint_path = "vit_company.pth"
    inspect_checkpoint(checkpoint_path)
    
    print("\n" + "=" * 60)
    print("💡 Based on this inspection, we can update the loading code accordingly.")
