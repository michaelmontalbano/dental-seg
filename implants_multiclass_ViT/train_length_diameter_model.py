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
import numpy as np

class ImplantDataset(Dataset):
    def __init__(self, json_path, vocabularies, transform=None, image_root=None, train_length=True, train_diameter=True):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.vocabs = vocabularies
        self.transform = transform
        self.image_root = image_root
        self.train_length = train_length
        self.train_diameter = train_diameter
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        print(f"🎯 Training targets: Length={train_length}, Diameter={train_diameter}")
        
        # Show data distribution for active tasks
        if train_length and 'length' in vocabularies:
            length_values = [e.get("length") for e in self.entries if e.get("length") is not None]
            print(f"📊 Length - unique values: {len(vocabularies['length'])}, samples: {len(length_values)}")
            
        if train_diameter and 'diameter' in vocabularies:
            diameter_values = [e.get("diameter") for e in self.entries if e.get("diameter") is not None]
            print(f"📊 Diameter - unique values: {len(vocabularies['diameter'])}, samples: {len(diameter_values)}")
        
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

        # Return labels based on training configuration
        targets = {}
        
        if self.train_length and 'length' in self.vocabs:
            targets["length"] = self.vocabs["length"].get(entry.get("length"), -1)
                
        if self.train_diameter and 'diameter' in self.vocabs:
            targets["diameter"] = self.vocabs["diameter"].get(entry.get("diameter"), -1)
        
        return image, targets

class ViTClassificationModel(nn.Module):
    def __init__(self, base_model_name, vocab_sizes, train_length=True, train_diameter=True):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        self.train_length = train_length
        self.train_diameter = train_diameter
        
        # Classification heads based on training configuration
        if train_length and 'length' in vocab_sizes:
            self.head_length = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, vocab_sizes['length'])
            )
        
        if train_diameter and 'diameter' in vocab_sizes:
            self.head_diameter = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, vocab_sizes['diameter'])
            )
        
        print(f"🏗️  Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        if train_length and 'length' in vocab_sizes:
            print(f"   Length classes: {vocab_sizes['length']}")
        if train_diameter and 'diameter' in vocab_sizes:
            print(f"   Diameter classes: {vocab_sizes['diameter']}")

    def forward(self, x):
        feat = self.backbone(x)
        outputs = {}
        
        if self.train_length and hasattr(self, 'head_length'):
            outputs["length"] = self.head_length(feat)
        if self.train_diameter and hasattr(self, 'head_diameter'):
            outputs["diameter"] = self.head_diameter(feat)
            
        return outputs

class EarlyStopping:
    """Early stopping based on accuracy instead of loss."""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_score = 0.0  # For accuracy, higher is better
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, accuracy, model):
        score = accuracy
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def build_vocab(entries, key):
    """Build vocabulary for classification task"""
    unique = sorted(set(str(e[key]) for e in entries if e.get(key) is not None))
    print(f"📋 {key} vocabulary: {len(unique)} unique values")
    if len(unique) <= 10:  # Show values if not too many
        print(f"   Values: {unique}")
    else:
        print(f"   Sample values: {unique[:5]} ... {unique[-2:]}")
    return {k: i for i, k in enumerate(unique)}

def evaluate_model(model, dataloader, criterion, device, train_length=True, train_diameter=True, detailed=False):
    """Evaluate model on validation set with accuracy metrics"""
    model.eval()
    total_loss = 0.0
    
    # Track predictions and targets for accuracy calculation
    correct_predictions = {}
    total_predictions = {}
    
    active_tasks = []
    if train_length:
        active_tasks.append("length")
        correct_predictions["length"] = 0
        total_predictions["length"] = 0
    if train_diameter:
        active_tasks.append("diameter")
        correct_predictions["diameter"] = 0
        total_predictions["diameter"] = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0.0
            for key in active_tasks:
                if key in labels:  # Check if this target exists in the batch
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
    for key in active_tasks:
        if total_predictions[key] > 0:
            accuracies[key] = correct_predictions[key] / total_predictions[key]
        else:
            accuracies[key] = 0.0
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracies

def print_detailed_metrics(accuracies, epoch, train_length=True, train_diameter=True):
    """Print detailed accuracy metrics"""
    print(f"\n📊 Detailed Metrics - Epoch {epoch}")
    print("=" * 60)
    
    active_tasks = []
    if train_length:
        active_tasks.append("length")
    if train_diameter:
        active_tasks.append("diameter")
    
    for key in active_tasks:
        if key not in accuracies:
            print(f"{key.upper()}: No valid predictions")
            continue
            
        print(f"\n🎯 {key.upper()} Classification:")
        print(f"  Accuracy: {accuracies[key]:.4f} ({accuracies[key]*100:.2f}%)")

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Length + Diameter Classification with Accuracy-Based Early Stopping')
    parser.add_argument('--train_json', type=str, required=True, 
                        help='Path to augmented_train.json')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Path to augmented_val.json (optional)')
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'],
                        help='ViT model size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
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
                        help='Compute detailed accuracy metrics')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')
    
    # Configurable training targets
    parser.add_argument('--train_length', type=str, default='true',
                        help='Train length classification head (true/false)')
    parser.add_argument('--train_diameter', type=str, default='true',
                        help='Train diameter classification head (true/false)')
    
    # Early stopping and learning rate adjustment based on accuracy
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without accuracy improvement)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum accuracy improvement for early stopping')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience (epochs without accuracy improvement)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='Factor to reduce learning rate by')
    parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-7,
                        help='Minimum learning rate')
    
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
    args.train_length = args.train_length.lower() in ('true', '1', 'yes')
    args.train_diameter = args.train_diameter.lower() in ('true', '1', 'yes')
    
    # Validate training targets
    if not args.train_length and not args.train_diameter:
        raise ValueError("At least one of --train_length or --train_diameter must be True")

    print("🦷 Enhanced Length + Diameter Classification Training")
    print("=" * 60)
    print("🎯 Predicting implant dimensions with accuracy-based early stopping:")
    if args.train_length:
        print("   • Length (classification)")
    if args.train_diameter:
        print("   • Diameter (classification)")
    print("=" * 60)
    print("🚀 Key Features:")
    print(f"   • Accuracy-Based Early Stopping: patience={args.early_stopping_patience}")
    print(f"   • Accuracy-Based LR Adjustment: patience={args.lr_scheduler_patience}")
    print(f"   • Configurable Targets: Length={args.train_length}, Diameter={args.train_diameter}")
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

    # Build vocabularies from all data for active tasks
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)

    print(f"📚 Building vocabularies from {len(all_entries)} total entries")
    vocabs = {}
    vocab_sizes = {}
    
    if args.train_length:
        vocabs["length"] = build_vocab(all_entries, "length")
        vocab_sizes["length"] = len(vocabs["length"])
        
    if args.train_diameter:
        vocabs["diameter"] = build_vocab(all_entries, "diameter")
        vocab_sizes["diameter"] = len(vocabs["diameter"])

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

    # Create datasets with configurable targets
    train_dataset = ImplantDataset(args.train_json, vocabs, train_transform, 
                                 image_root=args.image_root,
                                 train_length=args.train_length,
                                 train_diameter=args.train_diameter)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, vocabs, val_transform,
                                   image_root=args.image_root,
                                   train_length=args.train_length,
                                   train_diameter=args.train_diameter)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # Create model with configurable heads
    model_map = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224', 
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }

    model = ViTClassificationModel(model_map[args.model_size], 
                                  vocab_sizes,
                                  train_length=args.train_length,
                                  train_diameter=args.train_diameter).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and loss function for classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Classification with label smoothing
    
    # Learning rate scheduler with warmup (existing cosine annealing)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Additional learning rate scheduler for auto-adjustment based on accuracy
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_scheduler_factor,  # mode='max' for accuracy
        patience=args.lr_scheduler_patience, min_lr=args.lr_scheduler_min_lr,
        verbose=True
    )
    
    # Initialize early stopping based on accuracy
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        restore_best_weights=True
    )

    print(f"\n🚀 Starting enhanced training for max {args.epochs} epochs")
    print(f"📊 Training samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"⏰ Early stopping: patience={args.early_stopping_patience} (accuracy-based)")
    print(f"📈 LR reduction: patience={args.lr_scheduler_patience}, factor={args.lr_scheduler_factor} (accuracy-based)")

    best_val_accuracy = 0.0  # Best accuracy (higher is better)
    training_history = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine active tasks for metric calculation
    active_tasks = []
    if args.train_length:
        active_tasks.append("length")
    if args.train_diameter:
        active_tasks.append("diameter")
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {optimizer.param_groups[0]["lr"]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        total_correct = {task: 0 for task in active_tasks}
        total_samples = {task: 0 for task in active_tasks}
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            loss = 0.0
            outputs = model(images)

            # Calculate loss and accuracy for active tasks
            for key in active_tasks:
                if key in labels:  # Check if this target exists in the batch
                    target = labels[key].to(device)
                    mask = target != -1
                    if mask.sum() > 0:
                        loss += criterion(outputs[key][mask], target[mask])
                        
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
            pbar_postfix = {'loss': running_loss / (pbar.n + 1)}
            for task in active_tasks:
                if total_samples[task] > 0:
                    pbar_postfix[f'{task}_acc'] = total_correct[task] / total_samples[task]
            pbar.set_postfix(pbar_postfix)

        # Step warmup scheduler
        warmup_scheduler.step()
        
        # Calculate training metrics
        train_metrics = {
            'loss': running_loss / len(train_dataloader),
        }
        
        train_accuracies = {}
        for task in active_tasks:
            if total_samples[task] > 0:
                train_accuracies[task] = total_correct[task] / total_samples[task]
                train_metrics[f'{task}_accuracy'] = train_accuracies[task]
            else:
                train_accuracies[task] = 0.0
                train_metrics[f'{task}_accuracy'] = 0.0
        
        # Calculate combined accuracy
        valid_accuracies = [train_accuracies[task] for task in active_tasks if total_samples[task] > 0]
        train_metrics['combined_accuracy'] = np.mean(valid_accuracies) if valid_accuracies else 0.0
        
        # Print training metrics
        train_acc_str = ", ".join([f"{task.title()} Acc: {train_accuracies[task]:.3f}" 
                                  for task in active_tasks if total_samples[task] > 0])
        print(f"🔼 Train - Loss: {train_metrics['loss']:.4f}, {train_acc_str}, "
              f"Combined Acc: {train_metrics['combined_accuracy']:.3f}")
        
        # Validation phase
        val_accuracies = None
        val_loss = None
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_accuracies = evaluate_model(model, val_dataloader, criterion, device, 
                                                 train_length=args.train_length,
                                                 train_diameter=args.train_diameter,
                                                 detailed=args.detailed_metrics)
            
            # Calculate combined validation accuracy
            valid_val_accuracies = [val_accuracies[task] for task in active_tasks if task in val_accuracies]
            combined_accuracy = np.mean(valid_val_accuracies) if valid_val_accuracies else 0.0
            
            # Print validation metrics
            val_acc_str = ", ".join([f"{task.title()} Acc: {val_accuracies[task]:.3f}" 
                                   for task in active_tasks if task in val_accuracies])
            print(f"🔽 Val - Loss: {val_loss:.4f}, {val_acc_str}, "
                  f"Combined Acc: {combined_accuracy:.3f}")
            
            # Print detailed metrics if requested
            if args.detailed_metrics:
                print_detailed_metrics(val_accuracies, epoch + 1, 
                                     train_length=args.train_length,
                                     train_diameter=args.train_diameter)
            
            # Step plateau scheduler with combined accuracy
            plateau_scheduler.step(combined_accuracy)
            
            # Check early stopping based on accuracy
            if early_stopping(combined_accuracy, model):
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"⏰ No accuracy improvement for {early_stopping.patience} epochs")
                print(f"🔄 Restored best weights with accuracy {early_stopping.best_score:.3f}")
                break
            
            # Save best model (based on combined accuracy)
            if combined_accuracy > best_val_accuracy:
                best_val_accuracy = combined_accuracy
                print(f"💾 New best combined accuracy: {best_val_accuracy:.3f}! Saving model...")
                
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_accuracy': best_val_accuracy,
                    'vocabularies': vocabs,
                    'args': vars(args),
                    'training_history': training_history
                }
                
                # Add individual accuracies
                for task in active_tasks:
                    if task in val_accuracies:
                        save_dict[f'{task}_accuracy'] = val_accuracies[task]
                
                torch.save(save_dict, os.path.join(args.output_dir, "best_model.pth"))
            
            # Track training history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_combined_accuracy': train_metrics['combined_accuracy'],
                'val_loss': val_loss,
                'val_combined_accuracy': combined_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Add individual metrics
            for task in active_tasks:
                history_entry[f'train_{task}_accuracy'] = train_accuracies[task]
                if task in val_accuracies:
                    history_entry[f'val_{task}_accuracy'] = val_accuracies[task]
            
            training_history.append(history_entry)

    # Save final model and results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': len(training_history),
            'vocabularies': vocabs,
            'args': vars(args),
            'training_history': training_history
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save vocabularies
    with open(os.path.join(args.output_dir, "vocabularies.json"), "w") as f:
        # Convert vocab to inverse format for easier reading
        readable_vocabs = {}
        for task, vocab in vocabs.items():
            readable_vocabs[task] = {str(i): k for k, i in vocab.items()}
        json.dump(readable_vocabs, f, indent=2)
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save training info
    training_info = {
        "model_size": args.model_size,
        "epochs_completed": len(training_history),
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataloader else 0,
        "best_val_accuracy": best_val_accuracy,
        "vocab_sizes": vocab_sizes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_length": args.train_length,
        "train_diameter": args.train_diameter,
        "early_stopping_patience": args.early_stopping_patience,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "early_stopped": early_stopping.wait >= early_stopping.patience,
        "stopping_criteria": "accuracy_based",
        "final_training_history": training_history[-5:] if training_history else []
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    # Final validation if available
    if val_dataloader:
        print(f"\n🔍 Running final detailed validation...")
        final_val_loss, final_val_accuracies = evaluate_model(
            model, val_dataloader, criterion, device, 
            train_length=args.train_length,
            train_diameter=args.train_diameter,
            detailed=True
        )
        
        valid_final_accuracies = [final_val_accuracies[task] for task in active_tasks 
                                 if task in final_val_accuracies]
        final_combined_accuracy = np.mean(valid_final_accuracies) if valid_final_accuracies else 0.0
        
        print(f"\n🏁 FINAL RESULTS:")
        print(f"📊 Best validation accuracy: {best_val_accuracy:.3f}")
        print(f"📊 Final validation accuracy: {final_combined_accuracy:.3f}")
        if early_stopping.wait >= early_stopping.patience:
            print(f"⏰ Training stopped early at epoch {len(training_history)}/{args.epochs}")
        print_detailed_metrics(final_val_accuracies, "FINAL", 
                             train_length=args.train_length,
                             train_diameter=args.train_diameter)

    print(f"\n✅ Enhanced classification training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best combined validation accuracy: {best_val_accuracy:.3f}")
    print(f"📋 Training history: {os.path.join(args.output_dir, 'training_history.json')}")
    print(f"📚 Vocabularies: {os.path.join(args.output_dir, 'vocabularies.json')}")
    if early_stopping.wait >= early_stopping.patience:
        print(f"⏰ Early stopping saved approximately {args.epochs - len(training_history)} epochs")

if __name__ == '__main__':
    list_instance_files()
    main()