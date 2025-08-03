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
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class EarlyStopping:
    """Early stopping to stop training when validation accuracy stops improving"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_accuracy = 0.0
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_accuracy, model):
        if val_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = val_accuracy
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
    def __init__(self, initial_weights=None, adaptation_rate=0.1):
        self.weights = initial_weights or {"company": 1.0, "model": 1.0}
        self.adaptation_rate = adaptation_rate
        self.prev_accuracies = {"company": 0.0, "model": 0.0}
        
    def update_weights(self, current_accuracies):
        """Update weights based on relative task performance"""
        for task in self.weights.keys():
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
        for task in self.weights.keys():
            self.weights[task] = (self.weights[task] / total_weight) * len(self.weights)
            
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
        
        # Show data distribution
        companies = [e.get("company") for e in self.entries if e.get("company")]
        models = [e.get("model") for e in self.entries if e.get("model")]
        print(f"📊 Unique companies: {len(set(companies))}")
        print(f"📊 Unique models: {len(set(models))}")
        
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

        # Return only company and model labels
        return image, {
            "company": self.vocabs["company"].get(entry.get("company"), -1),
            "model": self.vocabs["model"].get(entry.get("model"), -1),
        }

class ViTCompanyModel(nn.Module):
    def __init__(self, base_model_name, vocab_sizes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Only company and model heads
        self.head_company = nn.Linear(dim, vocab_sizes['company'])
        self.head_model = nn.Linear(dim, vocab_sizes['model'])
        
        print(f"🏗️  Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Company classes: {vocab_sizes['company']}")
        print(f"   Model classes: {vocab_sizes['model']}")

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "company": self.head_company(feat),
            "model": self.head_model(feat),
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

def evaluate_model(model, dataloader, criterion, device, vocabs, detailed=False):
    """Evaluate model on validation set with detailed metrics"""
    model.eval()
    total_loss = 0.0
    
    # Track predictions and targets for detailed analysis
    all_predictions = {"company": [], "model": []}
    all_targets = {"company": [], "model": []}
    all_valid_mask = {"company": [], "model": []}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0.0
            for key in ["company", "model"]:
                target = labels[key].to(device)
                mask = target != -1
                
                if mask.sum() > 0:
                    batch_loss += criterion(outputs[key][mask], target[mask])
                    
                    # Store predictions and targets for metrics
                    pred = torch.argmax(outputs[key], dim=1)
                    all_predictions[key].extend(pred.cpu().numpy())
                    all_targets[key].extend(target.cpu().numpy())
                    all_valid_mask[key].extend(mask.cpu().numpy())
            
            total_loss += batch_loss.item()
    
    # Calculate detailed metrics
    metrics = {}
    for key in ["company", "model"]:
        # Filter to only valid predictions
        valid_preds = []
        valid_targets = []
        
        for pred, target, valid in zip(all_predictions[key], all_targets[key], all_valid_mask[key]):
            if valid:
                valid_preds.append(pred)
                valid_targets.append(target)
        
        if len(valid_preds) > 0:
            accuracy = np.mean(np.array(valid_preds) == np.array(valid_targets))
            metrics[key] = {
                'accuracy': accuracy,
                'predictions': valid_preds,
                'targets': valid_targets,
                'vocab_size': len(vocabs[key])
            }
            
            # Add detailed classification metrics if requested
            if detailed:
                try:
                    inv_vocab = inverse_vocab(vocabs[key])
                    label_names = [inv_vocab[i] for i in range(len(vocabs[key]))]
                    
                    # Only compute detailed metrics if reasonable number of classes
                    if len(label_names) <= 50:
                        report = classification_report(
                            valid_targets, valid_preds,
                            target_names=label_names,
                            zero_division=0,
                            output_dict=True
                        )
                        metrics[key]['classification_report'] = report
                        metrics[key]['macro_f1'] = report['macro avg']['f1-score']
                        metrics[key]['weighted_f1'] = report['weighted avg']['f1-score']
                except Exception as e:
                    print(f"⚠️  Could not compute detailed metrics for {key}: {e}")
        else:
            metrics[key] = {'accuracy': 0.0, 'predictions': [], 'targets': [], 'vocab_size': 0}
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics

def print_detailed_metrics(metrics, vocabs, epoch):
    """Print detailed classification metrics"""
    print(f"\n📊 Detailed Metrics - Epoch {epoch}")
    print("=" * 60)
    
    for key in ["company", "model"]:
        if len(metrics[key]['predictions']) == 0:
            print(f"{key.upper()}: No valid predictions")
            continue
            
        accuracy = metrics[key]['accuracy']
        print(f"\n🎯 {key.upper()} Classification:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Valid samples: {len(metrics[key]['predictions'])}")
        
        # Print additional metrics if available
        if 'macro_f1' in metrics[key]:
            print(f"  Macro F1: {metrics[key]['macro_f1']:.4f}")
            print(f"  Weighted F1: {metrics[key]['weighted_f1']:.4f}")
        
        # Show top-5 most common predictions vs targets
        if len(metrics[key]['predictions']) > 0:
            pred_counts = {}
            target_counts = {}
            
            for pred in metrics[key]['predictions']:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            for target in metrics[key]['targets']:
                target_counts[target] = target_counts.get(target, 0) + 1
            
            inv_vocab = inverse_vocab(vocabs[key])
            
            print(f"  Top predicted classes:")
            for class_id, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                class_name = inv_vocab.get(class_id, f"Unknown_{class_id}")
                print(f"    {class_name}: {count}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Company + Model Classification')
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
                        help='Compute detailed classification metrics')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with lower learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--adaptive_loss_weighting', type=str, default='true',
                        help='Enable adaptive loss weighting between tasks')
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
    args.adaptive_loss_weighting = args.adaptive_loss_weighting.lower() in ('true', '1', 'yes')

    print("🦷 Company + Model Classification Training (Enhanced)")
    print("=" * 60)
    print("🎯 Enhanced features:")
    print("   • Company identification") 
    print("   • Model identification")
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

    # Build vocabularies from all data (train + val) - only company and model
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)

    print(f"\n📚 Building vocabularies from {len(all_entries)} total entries")
    vocabs = {
        "company": build_vocab(all_entries, "company"),
        "model": build_vocab(all_entries, "model"),
    }

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

    model = ViTCompanyModel(model_map[args.model_size], 
                           {k: len(v) for k, v in vocabs.items()}).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Advanced optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize early stopping and adaptive loss weighting
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=0.001)
    loss_weighter = AdaptiveLossWeighting() if args.adaptive_loss_weighting else None
    
    if args.adaptive_loss_weighting:
        print(f"🔧 Adaptive loss weighting enabled")
    print(f"🛑 Early stopping patience: {args.early_stopping_patience} epochs")

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print(f"📊 Training samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"📊 Validation samples: {len(val_dataset)}")

    best_val_accuracy = 0.0
    training_history = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        total_correct = {"company": 0, "model": 0}
        total_samples = {"company": 0, "model": 0}
        
        # Get current loss weights
        if loss_weighter and epoch > 0:
            train_accuracies = {
                "company": total_correct["company"] / max(total_samples["company"], 1),
                "model": total_correct["model"] / max(total_samples["model"], 1)
            }
            loss_weighter.update_weights(train_accuracies)
            current_weights = loss_weighter.get_weights()
            print(f"🔧 Loss weights: Company={current_weights['company']:.3f}, Model={current_weights['model']:.3f}")
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            loss = 0.0
            outputs = model(images)

            # Calculate loss and accuracy for both company and model
            for key in ["company", "model"]:
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
            train_acc_company = total_correct["company"] / max(total_samples["company"], 1)
            train_acc_model = total_correct["model"] / max(total_samples["model"], 1)
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'company_acc': train_acc_company,
                'model_acc': train_acc_model
            })

        scheduler.step()
        
        # Training metrics
        train_metrics = {
            'loss': running_loss / len(train_dataloader),
            'company_accuracy': train_acc_company,
            'model_accuracy': train_acc_model,
            'combined_accuracy': (train_acc_company + train_acc_model) / 2
        }
        
        print(f"🔼 Train - Loss: {train_metrics['loss']:.4f}, "
              f"Company: {train_metrics['company_accuracy']:.4f}, "
              f"Model: {train_metrics['model_accuracy']:.4f}, "
              f"Combined: {train_metrics['combined_accuracy']:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_metrics = evaluate_model(model, val_dataloader, criterion, device, vocabs, 
                                                 detailed=args.detailed_metrics)
            
            company_acc = val_metrics["company"]["accuracy"]
            model_acc = val_metrics["model"]["accuracy"]
            combined_acc = (company_acc + model_acc) / 2
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, "
                  f"Company: {company_acc:.4f}, "
                  f"Model: {model_acc:.4f}, "
                  f"Combined: {combined_acc:.4f}")
            
            # Print detailed metrics if requested
            if args.detailed_metrics:
                print_detailed_metrics(val_metrics, vocabs, epoch + 1)
            
            # Save best model
            if combined_acc > best_val_accuracy:
                best_val_accuracy = combined_acc
                print(f"💾 New best combined accuracy: {best_val_accuracy:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_accuracy': best_val_accuracy,
                    'company_accuracy': company_acc,
                    'model_accuracy': model_acc,
                    'vocabs': vocabs,
                    'args': vars(args),
                    'training_history': training_history,
                    'loss_weights': loss_weighter.get_weights() if loss_weighter else None
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            # Check early stopping
            if early_stopping(combined_acc, model):
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"💾 Best validation accuracy: {early_stopping.best_accuracy:.4f}")
                break
            
            # Track training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_company_acc': train_metrics['company_accuracy'],
                'train_model_acc': train_metrics['model_accuracy'],
                'train_combined_acc': train_metrics['combined_accuracy'],
                'val_loss': val_loss,
                'val_company_acc': company_acc,
                'val_model_acc': model_acc,
                'val_combined_acc': combined_acc,
                'learning_rate': scheduler.get_last_lr()[0],
                'loss_weights': loss_weighter.get_weights() if loss_weighter else None,
                'early_stopping_counter': early_stopping.counter
            })

    # Save final model and results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'vocabs': vocabs,
            'args': vars(args),
            'training_history': training_history
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save vocabularies
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
        "best_val_accuracy": best_val_accuracy,
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

    # Final validation if available
    if val_dataloader:
        print(f"\n🔍 Running final detailed validation...")
        final_val_loss, final_val_metrics = evaluate_model(
            model, val_dataloader, criterion, device, vocabs, detailed=True
        )
        final_combined_acc = (final_val_metrics["company"]["accuracy"] + 
                            final_val_metrics["model"]["accuracy"]) / 2
        
        print(f"\n🏁 FINAL RESULTS:")
        print(f"📊 Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"📊 Final validation accuracy: {final_combined_acc:.4f}")
        if early_stopping.counter >= early_stopping.patience:
            print(f"🛑 Training stopped early due to no improvement")
        if loss_weighter:
            print(f"🔧 Final loss weights: {loss_weighter.get_weights()}")
        print_detailed_metrics(final_val_metrics, vocabs, "FINAL")

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best combined validation accuracy: {best_val_accuracy:.4f}")
    print(f"📋 Training history: {os.path.join(args.output_dir, 'training_history.json')}")
    print(f"📋 Vocabularies: {os.path.join(args.output_dir, 'vocabularies.json')}")

if __name__ == '__main__':
    list_instance_files()
    main()