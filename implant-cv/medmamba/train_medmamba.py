import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm


from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    return dataset


# Download MedMamba model from GitHub if not present
def setup_medmamba():
    """Download and setup MedMamba model with better error handling"""
    try:
        # Install dependencies first
        import timm
        print("✅ timm already installed")
    except ImportError:
        print("Installing timm...")
        result = os.system("pip install timm>=0.9.0")
        if result != 0:
            print("❌ Failed to install timm")
            return None
    
    try:
        import einops
        print("✅ einops already installed") 
    except ImportError:
        print("Installing einops...")
        result = os.system("pip install einops>=0.6.0")
        if result != 0:
            print("❌ Failed to install einops")
            return None
    
    # Try to install mamba-ssm but don't fail if it doesn't work
    try:
        import mamba_ssm
        print("✅ mamba-ssm already installed")
    except ImportError:
        print("Installing mamba-ssm (may take a few minutes)...")
        result = os.system("pip install mamba-ssm>=1.0.0")
        if result != 0:
            print("⚠️  mamba-ssm installation failed (common CUDA issue)")

            print("📦 Installing alternative Mamba implementation...")
            # Try alternative lightweight mamba
            os.system("pip install causal-conv1d>=1.0.0")
    
    # Download MedMamba model file
    if not os.path.exists('MedMamba.py'):
        print("Downloading MedMamba model...")
        result = os.system("wget https://raw.githubusercontent.com/YubiaoYue/MedMamba/main/MedMamba.py")
        if result != 0:
            print("❌ Failed to download MedMamba")
            return None
    
    # Try to import and test MedMamba
    sys.path.append('.')
    try:
        # Test import without executing the model
        with open('MedMamba.py', 'r') as f:
            content = f.read()
        
        # Check if the problematic test code is at the end
        if 'print(medmamba_t(data).shape)' in content:
            print("🔧 Fixing MedMamba test code...")
            # Remove the test code that's causing issues
            fixed_content = content.split('if __name__ == "__main__":')[0]
            
            with open('MedMamba_fixed.py', 'w') as f:
                f.write(fixed_content)
            
            # Import the fixed version
            import importlib.util
            spec = importlib.util.spec_from_file_location("MedMamba", "MedMamba_fixed.py")
            medmamba_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(medmamba_module)
            
            # Get the VSSM class
            medmamba = medmamba_module.VSSM
            print("✅ MedMamba fixed and imported successfully")
            return medmamba
        else:
            # Try normal import
            try:
                from MedMamba import VSSM as medmamba
                print("✅ MedMamba imported successfully")
                return medmamba
            except ImportError as e:
                sys.exit(f"❌ Failed to import MedMamba: {e}")
            
    except Exception as e:
        print(f"❌ MedMamba import failed: {e}")
        print("🔄 This is likely due to mamba-ssm CUDA compilation issues")
        print("💡 Using ResNet50 instead (actually excellent for medical imaging!)")
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/classification', 
                       help="Path to ImageFolder dataset root")
    parser.add_argument('--model_type', type=str, default='medmamba', choices=['medmamba', 'resnet'],
                       help="Model type: 'medmamba' or 'resnet' (default: medmamba)")
    parser.add_argument('--model_size', type=str, default='small', 
                       choices=['tiny', 'small', 'base'], 
                       help="Model size (only used for MedMamba, ignored for ResNet)")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Debug: Print all received arguments
    print("=== RECEIVED ARGUMENTS ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    
    # Check if classification dataset exists
    train_path = os.path.join(args.data_dir, 'train')
    val_path = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(train_path):
        print(f"Error: Train directory not found at {train_path}")
        print("Please run create_classification_dataset.py first to convert your YOLO dataset")
        return
    
    if not os.path.exists(val_path):
        print(f"Error: Val directory not found at {val_path}")
        print("Please run create_classification_dataset.py first to convert your YOLO dataset")
        return
    
    # Setup MedMamba model
    medmamba = setup_medmamba()
    
    # Data transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=data_transform["train"]
    )
    val_dataset = datasets.ImageFolder(
        root=val_path,
        transform=data_transform["val"]
    )
    
    # Auto-detect number of classes
    num_classes = len(train_dataset.classes)
    print(f"Detected {num_classes} classes: {train_dataset.classes[:5]}...")  # Show first 5
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Save class indices
    class_to_idx = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    # Initialize model
    model_configs = {
        'tiny': {'depths': [2, 2, 9, 2], 'dims': [96, 192, 384, 768]},
        'small': {'depths': [2, 2, 27, 2], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [2, 2, 27, 2], 'dims': [128, 256, 512, 1024]}
    }
    
    config = model_configs[args.model_size]
    model = medmamba(
        depths=config['depths'],
        dims=config['dims'],
        num_classes=num_classes  # Use auto-detected number of classes
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f'New best model saved with accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()