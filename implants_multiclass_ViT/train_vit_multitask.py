import argparse
import os
import json

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
import subprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ImplantDataset(Dataset):
    def __init__(self, json_path, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution for length and diameter
        lengths = []
        diameters = []
        
        for e in self.entries:
            # Convert to float, skip if invalid
            if e.get("length") is not None:
                try:
                    length_val = float(e["length"])
                    lengths.append(length_val)
                except (ValueError, TypeError):
                    pass
            
            if e.get("diameter") is not None:
                try:
                    diameter_val = float(e["diameter"])
                    diameters.append(diameter_val)
                except (ValueError, TypeError):
                    pass
        
        if lengths:
            print(f"📊 Length - samples: {len(lengths)}, range: {min(lengths):.1f}-{max(lengths):.1f}mm, mean: {np.mean(lengths):.1f}mm")
        else:
            print("📊 Length - no valid samples found")
            
        if diameters:
            print(f"📊 Diameter - samples: {len(diameters)}, range: {min(diameters):.1f}-{max(diameters):.1f}mm, mean: {np.mean(diameters):.1f}mm")
        else:
            print("📊 Diameter - no valid samples found")
        
        # Show sample entry structure
        if self.entries:
            sample = self.entries[0]
            print(f"📋 Sample entry keys: {list(sample.keys())}")
            if "bboxes" in sample:
                bbox_count = len(sample["bboxes"]) if sample["bboxes"] else 0
                print(f"📦 Sample has {bbox_count} bboxes (ignored for regression)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Handle both old and new path formats
        if self.image_root and entry["image"].startswith("/opt/ml/"):
            # New SageMaker format: /opt/ml/input/data/train_augmented/train/company/image.jpg
            # Extract the relative path from train_augmented onwards
            path_parts = entry["image"].split("/opt/ml/input/data/train_augmented/")[-1]
            image_path = os.path.join(self.image_root, path_parts)
        else:
            # Fallback to original handling
            image_filename = os.path.basename(entry["image"])
            company = entry.get("company", "Unknown")
            # Clean company name for filesystem
            clean_company = "".join(c for c in company if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_company = clean_company.replace(' ', '_')
            image_filename = f"{clean_company}/{image_filename}" 
            image_path = os.path.join(self.image_root, image_filename) if self.image_root else entry["image"]
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            # Return a dummy image to avoid crashing
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)

        # Return length and diameter as regression targets
        # Use -1 to indicate missing values, convert to float
        length = entry.get("length", -1)
        diameter = entry.get("diameter", -1)
        
        # Convert to float, use -1.0 for invalid/missing values
        try:
            length = float(length) if length != -1 and length is not None else -1.0
        except (ValueError, TypeError):
            length = -1.0
            
        try:
            diameter = float(diameter) if diameter != -1 and diameter is not None else -1.0
        except (ValueError, TypeError):
            diameter = -1.0
        
        return image, {
            "length": length,
            "diameter": diameter,
        }

class ViTRegressionModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Regression heads for length and diameter
        self.head_length = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.head_diameter = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        print(f"🏗️  Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Length head: {dim} -> 256 -> 1 (regression)")
        print(f"   Diameter head: {dim} -> 256 -> 1 (regression)")

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "length": self.head_length(feat).squeeze(-1),
            "diameter": self.head_diameter(feat).squeeze(-1),
        }

def evaluate_model(model, dataloader, criterion, device, detailed=False):
    """Evaluate model on validation set with regression metrics"""
    model.eval()
    total_loss = 0.0
    
    # Track predictions and targets for detailed analysis
    all_predictions = {"length": [], "diameter": []}
    all_targets = {"length": [], "diameter": []}
    all_valid_mask = {"length": [], "diameter": []}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0.0
            for key in ["length", "diameter"]:
                target = labels[key].to(device, dtype=torch.float32)
                mask = target != -1
                
                if mask.sum() > 0:
                    batch_loss += criterion(outputs[key][mask], target[mask])
                    
                    # Store predictions and targets for metrics
                    all_predictions[key].extend(outputs[key].cpu().numpy())
                    all_targets[key].extend(target.cpu().numpy())
                    all_valid_mask[key].extend(mask.cpu().numpy())
            
            total_loss += batch_loss.item()
    
    # Calculate regression metrics
    metrics = {}
    for key in ["length", "diameter"]:
        # Filter to only valid predictions
        valid_preds = []
        valid_targets = []
        
        for pred, target, valid in zip(all_predictions[key], all_targets[key], all_valid_mask[key]):
            if valid:
                valid_preds.append(pred)
                valid_targets.append(target)
        
        if len(valid_preds) > 0:
            valid_preds = np.array(valid_preds)
            valid_targets = np.array(valid_targets)
            
            # Calculate regression metrics
            mse = mean_squared_error(valid_targets, valid_preds)
            mae = mean_absolute_error(valid_targets, valid_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(valid_targets, valid_preds)
            
            # Calculate percentage error
            mape = np.mean(np.abs((valid_targets - valid_preds) / np.clip(valid_targets, 1e-8, None))) * 100
            
            metrics[key] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'predictions': valid_preds,
                'targets': valid_targets,
                'sample_count': len(valid_preds)
            }
        else:
            metrics[key] = {
                'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'),
                'r2': -float('inf'), 'mape': float('inf'),
                'predictions': [], 'targets': [], 'sample_count': 0
            }
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics

def print_detailed_metrics(metrics, epoch):
    """Print detailed regression metrics"""
    print(f"\n📊 Detailed Metrics - Epoch {epoch}")
    print("=" * 60)
    
    for key in ["length", "diameter"]:
        if metrics[key]['sample_count'] == 0:
            print(f"{key.upper()}: No valid predictions")
            continue
            
        print(f"\n🎯 {key.upper()} Regression:")
        print(f"  Samples: {metrics[key]['sample_count']}")
        print(f"  MAE: {metrics[key]['mae']:.3f}mm")
        print(f"  RMSE: {metrics[key]['rmse']:.3f}mm")
        print(f"  R²: {metrics[key]['r2']:.4f}")
        print(f"  MAPE: {metrics[key]['mape']:.1f}%")
        
        # Show prediction statistics
        preds = metrics[key]['predictions']
        targets = metrics[key]['targets']
        if len(preds) > 0:
            print(f"  Predicted range: {np.min(preds):.1f} - {np.max(preds):.1f}mm")
            print(f"  Actual range: {np.min(targets):.1f} - {np.max(targets):.1f}mm")

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Length + Diameter Regression')
    parser.add_argument('--train_json', type=str, required=True, 
                        help='Path to augmented_train.json')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Path to augmented_val.json (optional)')
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'],
                        help='ViT model size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for AdamW')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and results')
    parser.add_argument('--image_root', type=str, default='aug_dataset',
                        help='Root directory for images')
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--save_best_only', type=str, default='false',
                        help='Only save the model with best validation performance')
    parser.add_argument('--detailed_metrics', type=str, default='false',
                        help='Compute detailed regression metrics')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')
    return parser.parse_args()

def list_instance_files():
    """List directory structure for debugging"""
    if os.path.exists("/opt/ml/"):
        print("\n=== 🔍 SAGEMAKER INSTANCE STRUCTURE ===")
        dirs_to_check = [
            "/opt/ml/input",
            "/opt/ml/input/data", 
            "/opt/ml/input/data/train_augmented",
        ]
        for d in dirs_to_check:
            print(f"\n📁 {d}")
            try:
                output = subprocess.check_output(["ls", "-lh", d], stderr=subprocess.STDOUT).decode()
                print(output)
            except subprocess.CalledProcessError as e:
                print(f"❌ Could not list {d}: {e.output.decode()}")
        
        # Check for JSON files specifically
        try:
            print(f"\n📋 JSON files in /opt/ml/input/data/train_augmented:")
            output = subprocess.check_output(["find", "/opt/ml/input/data/train_augmented", "-name", "*.json"], 
                                           stderr=subprocess.STDOUT).decode()
            print(output)
        except subprocess.CalledProcessError:
            print("❌ No JSON files found")
        
        print("==========================================\n")
    else:
        print("🖥️  Running locally (not in SageMaker)")

def main():
    args = parse_args()
    
    # Convert string boolean arguments
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')
    args.detailed_metrics = args.detailed_metrics.lower() in ('true', '1', 'yes')

    print("🦷 Length + Diameter Regression Training")
    print("=" * 60)
    print("🎯 Predicting continuous implant dimensions:")
    print("   • Length (mm)")
    print("   • Diameter (mm)")
    print("=" * 60)
    print("=== TRAINING CONFIGURATION ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using {device} device")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🚀 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load training data
    print(f"\n📊 Loading training data from {args.train_json}")
    with open(args.train_json) as f:
        train_entries = json.load(f)

    # Load validation data if provided
    val_entries = None
    if args.val_json and os.path.exists(args.val_json):
        print(f"📊 Loading validation data from {args.val_json}")
        with open(args.val_json) as f:
            val_entries = json.load(f)
        print(f"✅ Validation data: {len(val_entries)} entries")
    else:
        print("⚠️  No validation data provided - training without validation monitoring")

    # Create transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImplantDataset(args.train_json, train_transform, 
                                 image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, val_transform,
                                   image_root=args.image_root)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model_map = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224', 
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }

    model = ViTRegressionModel(model_map[args.model_size]).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and loss function for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, betas=(0.9, 0.999))
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print(f"📊 Training samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"📊 Validation samples: {len(val_dataset)}")

    best_val_mae = float('inf')  # Best mean absolute error (lower is better)
    training_history = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        total_mae = {"length": 0.0, "diameter": 0.0}
        total_samples = {"length": 0, "diameter": 0}
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            loss = 0.0
            outputs = model(images)

            # Calculate loss and MAE for both length and diameter
            for key in ["length", "diameter"]:
                target = labels[key].to(device, dtype=torch.float32)
                mask = target != -1
                if mask.sum() > 0:
                    loss += criterion(outputs[key][mask], target[mask])
                    
                    # Track training MAE
                    mae = torch.mean(torch.abs(outputs[key][mask] - target[mask]))
                    total_mae[key] += mae.item() * mask.sum().item()
                    total_samples[key] += mask.sum().item()

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar
            train_mae_length = total_mae["length"] / max(total_samples["length"], 1)
            train_mae_diameter = total_mae["diameter"] / max(total_samples["diameter"], 1)
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'length_mae': train_mae_length,
                'diameter_mae': train_mae_diameter
            })

        scheduler.step()
        
        # Training metrics
        train_metrics = {
            'loss': running_loss / len(train_dataloader),
            'length_mae': train_mae_length,
            'diameter_mae': train_mae_diameter,
            'combined_mae': (train_mae_length + train_mae_diameter) / 2
        }
        
        print(f"🔼 Train - Loss: {train_metrics['loss']:.4f}, "
              f"Length MAE: {train_metrics['length_mae']:.3f}mm, "
              f"Diameter MAE: {train_metrics['diameter_mae']:.3f}mm, "
              f"Combined MAE: {train_metrics['combined_mae']:.3f}mm")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_metrics = evaluate_model(model, val_dataloader, criterion, device, 
                                                 detailed=args.detailed_metrics)
            
            length_mae = val_metrics["length"]["mae"]
            diameter_mae = val_metrics["diameter"]["mae"]
            combined_mae = (length_mae + diameter_mae) / 2
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, "
                  f"Length MAE: {length_mae:.3f}mm, "
                  f"Diameter MAE: {diameter_mae:.3f}mm, "
                  f"Combined MAE: {combined_mae:.3f}mm")
            
            # Print detailed metrics if requested
            if args.detailed_metrics:
                print_detailed_metrics(val_metrics, epoch + 1)
            
            # Save best model (based on combined MAE)
            if combined_mae < best_val_mae:
                best_val_mae = combined_mae
                print(f"💾 New best combined MAE: {best_val_mae:.3f}mm! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mae': best_val_mae,
                    'length_mae': length_mae,
                    'diameter_mae': diameter_mae,
                    'args': vars(args),
                    'training_history': training_history
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            # Track training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_length_mae': train_metrics['length_mae'],
                'train_diameter_mae': train_metrics['diameter_mae'],
                'train_combined_mae': train_metrics['combined_mae'],
                'val_loss': val_loss,
                'val_length_mae': length_mae,
                'val_diameter_mae': diameter_mae,
                'val_combined_mae': combined_mae,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Save final model and results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'args': vars(args),
            'training_history': training_history
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save training info
    training_info = {
        "model_size": args.model_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataloader else 0,
        "best_val_mae": best_val_mae,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "final_training_history": training_history[-5:] if training_history else []
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    # Final validation if available
    if val_dataloader:
        print(f"\n🔍 Running final detailed validation...")
        final_val_loss, final_val_metrics = evaluate_model(
            model, val_dataloader, criterion, device, detailed=True
        )
        final_combined_mae = (final_val_metrics["length"]["mae"] + 
                            final_val_metrics["diameter"]["mae"]) / 2
        
        print(f"\n🏁 FINAL RESULTS:")
        print(f"📊 Best validation MAE: {best_val_mae:.3f}mm")
        print(f"📊 Final validation MAE: {final_combined_mae:.3f}mm")
        print_detailed_metrics(final_val_metrics, "FINAL")

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best combined validation MAE: {best_val_mae:.3f}mm")
    print(f"📋 Training history: {os.path.join(args.output_dir, 'training_history.json')}")

if __name__ == '__main__':
    list_instance_files()
    main()