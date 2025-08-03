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

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class AdaptiveLossWeighting:
    """Automatically adjust loss weights based on task performance"""
    def __init__(self, tasks=None, adaptation_rate=0.1):
        self.tasks = tasks or ["company", "model", "diameter", "length"]
        self.weights = {task: 1.0 for task in self.tasks}
        self.adaptation_rate = adaptation_rate
        self.prev_accuracies = {task: 0.0 for task in self.tasks}
        
    def update_weights(self, current_accuracies):
        """Update weights based on relative task performance"""
        for task in self.tasks:
            current_acc = current_accuracies.get(task, 0.0)
            
            # If task is performing worse relative to others, increase its weight
            avg_acc = sum(current_accuracies.values()) / len(current_accuracies)
            
            if current_acc < avg_acc:
                # Increase weight for underperforming task
                self.weights[task] += self.adaptation_rate * (avg_acc - current_acc)
            else:
                # Slightly decrease weight for overperforming task
                self.weights[task] = max(0.1, self.weights[task] - self.adaptation_rate * 0.5 * (current_acc - avg_acc))
                
            # Keep weights reasonable
            self.weights[task] = max(0.1, min(3.0, self.weights[task]))
            
        # Normalize weights to sum to number of tasks
        total_weight = sum(self.weights.values())
        for task in self.tasks:
            self.weights[task] = (self.weights[task] / total_weight) * len(self.tasks)
            
        self.prev_accuracies = current_accuracies.copy()
        
    def get_weights(self):
        return self.weights

class ImplantDataset(Dataset):
    def __init__(self, json_path, vocabularies, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.vocabs = vocabularies
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        # Show sample entry structure
        if self.entries:
            sample = self.entries[0]
            print(f"📋 Sample entry keys: {list(sample.keys())}")
            if "bboxes" in sample:
                bbox_count = len(sample["bboxes"]) if sample["bboxes"] else 0
                print(f"📦 Sample has {bbox_count} bboxes (ignored for classification)")

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

        # Return labels, ignoring bboxes for classification task
        return image, {
            "company": self.vocabs["company"].get(entry.get("company"), -1),
            "model": self.vocabs["model"].get(entry.get("model"), -1),
            "diameter": self.vocabs["diameter"].get(entry.get("diameter"), -1),
            "length": self.vocabs["length"].get(entry.get("length"), -1),
        }

class ViTMultiHead(nn.Module):
    def __init__(self, base_model_name, vocab_sizes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        self.head_company = nn.Linear(dim, vocab_sizes['company'])
        self.head_model = nn.Linear(dim, vocab_sizes['model'])
        self.head_diameter = nn.Linear(dim, vocab_sizes['diameter'])
        self.head_length = nn.Linear(dim, vocab_sizes['length'])
        
        print(f"🏗️  Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Company classes: {vocab_sizes['company']}")
        print(f"   Model classes: {vocab_sizes['model']}")
        print(f"   Diameter classes: {vocab_sizes['diameter']}")
        print(f"   Length classes: {vocab_sizes['length']}")

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "company": self.head_company(feat),
            "model": self.head_model(feat),
            "diameter": self.head_diameter(feat),
            "length": self.head_length(feat),
        }

def build_vocab(entries, key):
    unique = sorted(set(e[key] for e in entries if e.get(key) is not None))
    print(f"📋 {key} vocabulary: {len(unique)} unique values")
    if len(unique) <= 10:  # Show values if not too many
        print(f"   Values: {unique}")
    else:
        print(f"   Sample values: {unique[:5]} ... {unique[-2:]}")
    return {k: i for i, k in enumerate(unique)}

def inverse_vocab(vocab):
    return {i: k for k, i in vocab.items()}

def evaluate_model(model, dataloader, criterion, device, detailed=False):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    correct_predictions = {"company": 0, "model": 0, "diameter": 0, "length": 0}
    total_predictions = {"company": 0, "model": 0, "diameter": 0, "length": 0}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0.0
            for key in ["company", "model", "diameter", "length"]:
                target = labels[key].to(device)
                mask = target != -1
                if mask.sum() > 0:
                    batch_loss += criterion(outputs[key][mask], target[mask])
                    
                    # Calculate accuracy
                    pred = torch.argmax(outputs[key][mask], dim=1)
                    correct_predictions[key] += (pred == target[mask]).sum().item()
                    total_predictions[key] += mask.sum().item()
            
            total_loss += batch_loss.item()
    
    # Calculate accuracies
    accuracies = {}
    for key in ["company", "model", "diameter", "length"]:
        if total_predictions[key] > 0:
            accuracies[key] = correct_predictions[key] / total_predictions[key]
        else:
            accuracies[key] = 0.0
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracies

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced ViT Multi-Task Training')
    parser.add_argument('--train_json', type=str, required=True, 
                        help='Path to augmented_train.json')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Path to augmented_val.json (optional)')
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--image_root', type=str, default='/opt/ml/input/data/train_augmented',
                        help='Root directory for images inside the container')
    parser.add_argument('--validate_every', type=int, default=10,
                        help='Run validation every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--adaptive_loss_weighting', type=str, default='true',
                        help='Enable adaptive loss weighting between tasks')
    parser.add_argument('--detailed_metrics', type=str, default='false',
                        help='Print detailed metrics per task')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')
    return parser.parse_args()

def list_instance_files():
    print("\n=== 🔍 LISTING INSTANCE FILE STRUCTURE ===")
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

def main():
    args = parse_args()
    
    # Convert string boolean arguments
    args.adaptive_loss_weighting = args.adaptive_loss_weighting.lower() in ('true', '1', 'yes')
    args.detailed_metrics = args.detailed_metrics.lower() in ('true', '1', 'yes')

    print("🦷 Enhanced Multi-Task Classification Training")
    print("=" * 60)
    print("🎯 Enhanced features:")
    print("   • Company identification") 
    print("   • Model identification")
    print("   • Diameter classification")
    print("   • Length classification")
    print("   • Early stopping")
    print("   • Adaptive loss weighting")
    print("=" * 60)
    print("=== TRAINING CONFIGURATION ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using {device} device")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # Load training data
    print(f"📊 Loading training data from {args.train_json}")
    with open(args.train_json) as f:
        train_entries = json.load(f)

    # Load validation data if provided
    val_entries = None
    if args.val_json and os.path.exists(args.val_json):
        print(f"📊 Loading validation data from {args.val_json}")
        with open(args.val_json) as f:
            val_entries = json.load(f)
    else:
        print("ℹ️  No validation data provided")

    # Build vocabularies from all data (train + val)
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)

    print(f"📚 Building vocabularies from {len(all_entries)} total entries")
    vocabs = {
        "company": build_vocab(all_entries, "company"),
        "model": build_vocab(all_entries, "model"),
        "diameter": build_vocab(all_entries, "diameter"),
        "length": build_vocab(all_entries, "length"),
    }

    # Enhanced training transforms
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
    train_dataset = ImplantDataset(args.train_json, vocabs, train_transform, 
                                 image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, vocabs, val_transform,
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

    model = ViTMultiHead(model_map[args.model_size], 
                        {k: len(v) for k, v in vocabs.items()}).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Enhanced optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize early stopping and adaptive loss weighting
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=0.001)
    loss_weighter = AdaptiveLossWeighting() if args.adaptive_loss_weighting else None
    
    if args.adaptive_loss_weighting:
        print(f"🔧 Adaptive loss weighting enabled")
    print(f"🛑 Early stopping patience: {args.early_stopping_patience} epochs")

    print(f"🚀 Starting training for {args.epochs} epochs")
    print(f"📊 Training samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"📊 Validation samples: {len(val_dataset)}")

    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch+1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        total_correct = {"company": 0, "model": 0, "diameter": 0, "length": 0}
        total_samples = {"company": 0, "model": 0, "diameter": 0, "length": 0}
        
        # Get current loss weights
        if loss_weighter and epoch > 0:
            train_accuracies = {
                key: total_correct[key] / max(total_samples[key], 1) 
                for key in ["company", "model", "diameter", "length"]
            }
            loss_weighter.update_weights(train_accuracies)
            current_weights = loss_weighter.get_weights()
            print(f"🔧 Loss weights: " + ", ".join([f"{k}={v:.3f}" for k, v in current_weights.items()]))
        
        pbar = tqdm(train_dataloader, desc="Training")

        for images, labels in pbar:
            images = images.to(device)
            loss = 0.0
            outputs = model(images)

            for key in ["company", "model", "diameter", "length"]:
                target = labels[key].to(device)
                mask = target != -1
                if mask.sum() > 0:
                    task_loss = criterion(outputs[key][mask], target[mask])
                    
                    # Apply adaptive weight if enabled
                    if loss_weighter:
                        weight = loss_weighter.get_weights()[key]
                        task_loss = task_loss * weight
                    
                    loss += task_loss
                    
                    # Track training accuracy
                    pred = torch.argmax(outputs[key][mask], dim=1)
                    total_correct[key] += (pred == target[mask]).sum().item()
                    total_samples[key] += mask.sum().item()

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        scheduler.step()
        
        # Print training metrics
        train_accuracies = {
            key: total_correct[key] / max(total_samples[key], 1) 
            for key in ["company", "model", "diameter", "length"]
        }
        avg_train_acc = sum(train_accuracies.values()) / len(train_accuracies)
        
        print(f"🔼 Train - Loss: {running_loss/len(train_dataloader):.4f}, Avg Acc: {avg_train_acc:.4f}")
        if args.detailed_metrics:
            for key, acc in train_accuracies.items():
                print(f"   {key}: {acc:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_accuracies = evaluate_model(model, val_dataloader, criterion, device, args.detailed_metrics)
            avg_val_acc = sum(val_accuracies.values()) / len(val_accuracies)
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, Avg Acc: {avg_val_acc:.4f}")
            if args.detailed_metrics:
                for key, acc in val_accuracies.items():
                    print(f"   {key}: {acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"💾 New best validation loss: {best_val_loss:.4f}! Saving model...")
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': best_val_loss,
                    'val_accuracies': val_accuracies,
                    'vocabs': vocabs,
                    'args': vars(args),
                    'loss_weights': loss_weighter.get_weights() if loss_weighter else None
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            # Check early stopping
            if early_stopping(val_loss, model):
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"💾 Best validation loss: {early_stopping.best_loss:.4f}")
                break
            
            # Track training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_dataloader),
                'train_accuracies': train_accuracies,
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'learning_rate': scheduler.get_last_lr()[0],
                'loss_weights': loss_weighter.get_weights() if loss_weighter else None,
                'early_stopping_counter': early_stopping.counter
            })

    # Save final model and vocabularies
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    
    with open(os.path.join(args.output_dir, "vocabularies.json"), "w") as f:
        json.dump({k: inverse_vocab(v) for k, v in vocabs.items()}, f, indent=2)
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save training info
    training_info = {
        "model_size": args.model_size,
        "epochs": epoch + 1,  # Actual epochs trained
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataloader else 0,
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "best_val_loss": best_val_loss,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "early_stopping_triggered": early_stopping.counter >= early_stopping.patience,
        "early_stopping_patience": args.early_stopping_patience,
        "adaptive_loss_weighting": args.adaptive_loss_weighting,
        "final_loss_weights": loss_weighter.get_weights() if loss_weighter else None,
        "final_training_history": training_history[-5:] if training_history else []
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    print("\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    if early_stopping.counter >= early_stopping.patience:
        print(f"🛑 Training stopped early due to no improvement")
    if loss_weighter:
        print(f"🔧 Final loss weights: {loss_weighter.get_weights()}")
    print(f"📋 Vocabularies saved to: {os.path.join(args.output_dir, 'vocabularies.json')}")

if __name__ == '__main__':
    list_instance_files()
    main()