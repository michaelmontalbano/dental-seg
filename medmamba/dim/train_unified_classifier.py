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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class ImplantDataset(Dataset):
    def __init__(self, json_path, vocabularies, diameter_bins, length_bins, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.vocabs = vocabularies
        self.diameter_bins = diameter_bins
        self.length_bins = length_bins
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution
        companies = [e.get("company") for e in self.entries if e.get("company")]
        models = [e.get("model") for e in self.entries if e.get("model")]
        diameters = []
        lengths = []
        
        for e in self.entries:
            if e.get("diameter") is not None:
                try:
                    d = float(e["diameter"])
                    if d > 0:
                        diameters.append(d)
                except (ValueError, TypeError):
                    pass
            if e.get("length") is not None:
                try:
                    l = float(e["length"])
                    if l > 0:
                        lengths.append(l)
                except (ValueError, TypeError):
                    pass
        
        print(f"📊 Dataset statistics:")
        print(f"   Companies: {len(set(companies))} unique")
        print(f"   Models: {len(set(models))} unique")
        print(f"   Diameters: {len(diameters)} samples, range: {min(diameters):.1f}-{max(diameters):.1f}mm")
        print(f"   Lengths: {len(lengths)} samples, range: {min(lengths):.1f}-{max(lengths):.1f}mm")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Handle both old and new path formats
        if self.image_root and entry["image"].startswith("/opt/ml/"):
            path_parts = entry["image"].split("/opt/ml/input/data/train_augmented/")[-1]
            image_path = os.path.join(self.image_root, path_parts)
        else:
            image_filename = os.path.basename(entry["image"])
            company = entry.get("company", "Unknown")
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
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)

        # Get labels for all four tasks
        company_label = self.vocabs["company"].get(entry.get("company"), -1)
        model_label = self.vocabs["model"].get(entry.get("model"), -1)
        
        # Convert diameter to bin label
        diameter = entry.get("diameter", -1)
        try:
            diameter = float(diameter)
            if diameter > 0:
                # np.digitize returns 0 if x < bins[0], n if x >= bins[n-1]
                diameter_label = np.digitize(diameter, self.diameter_bins) - 1
                # Ensure label is within valid range [0, num_bins-1]
                diameter_label = max(0, min(diameter_label, len(self.diameter_bins) - 2))
            else:
                diameter_label = -1
        except (ValueError, TypeError):
            diameter_label = -1
            
        # Convert length to bin label
        length = entry.get("length", -1)
        try:
            length = float(length)
            if length > 0:
                # np.digitize returns 0 if x < bins[0], n if x >= bins[n-1]
                length_label = np.digitize(length, self.length_bins) - 1
                # Ensure label is within valid range [0, num_bins-1]
                length_label = max(0, min(length_label, len(self.length_bins) - 2))
            else:
                length_label = -1
        except (ValueError, TypeError):
            length_label = -1
            
        return image, {
            "company": company_label,
            "model": model_label,
            "diameter": diameter_label,
            "length": length_label
        }

class UnifiedClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes_dict):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Separate heads for each task
        self.company_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_dict['company'])
        )
        
        self.model_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_dict['model'])
        )
        
        self.diameter_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_dict['diameter'])
        )
        
        self.length_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_dict['length'])
        )
        
        print(f"🏗️  Unified Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Company classes: {num_classes_dict['company']}")
        print(f"   Model classes: {num_classes_dict['model']}")
        print(f"   Diameter bins: {num_classes_dict['diameter']}")
        print(f"   Length bins: {num_classes_dict['length']}")

    def forward(self, x):
        features = self.backbone(x)
        return {
            "company": self.company_head(features),
            "model": self.model_head(features),
            "diameter": self.diameter_head(features),
            "length": self.length_head(features)
        }

def build_vocab(entries, key):
    unique = sorted(set(e[key] for e in entries if e.get(key) is not None))
    print(f"📋 {key} vocabulary: {len(unique)} unique values")
    return {k: i for i, k in enumerate(unique)}

def create_bins(values, bin_size=0.5, strategy='fixed'):
    """Create bins for continuous values"""
    if not values:
        return np.array([0, 1])  # Default bins if no values
    
    if strategy == 'fixed':
        min_val = min(values)
        max_val = max(values)
        min_bin = np.floor(min_val / bin_size) * bin_size
        max_bin = np.ceil(max_val / bin_size) * bin_size
        bin_edges = np.arange(min_bin, max_bin + bin_size, bin_size)
    else:  # adaptive
        n_bins = int(1 + (max(values) - min(values)) / bin_size)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(values, percentiles)
        bin_edges = np.unique(bin_edges)
    
    return bin_edges

def evaluate_model(model, dataloader, criterion, device, vocabs, diameter_bins, length_bins):
    """Evaluate unified model on validation set"""
    model.eval()
    total_loss = 0.0
    
    # Track metrics for each task
    task_metrics = {
        "company": {"predictions": [], "targets": [], "loss": 0.0},
        "model": {"predictions": [], "targets": [], "loss": 0.0},
        "diameter": {"predictions": [], "targets": [], "loss": 0.0},
        "length": {"predictions": [], "targets": [], "loss": 0.0}
    }
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0.0
            for task in ["company", "model", "diameter", "length"]:
                target = labels[task].to(device)
                mask = target != -1
                
                if mask.sum() > 0:
                    task_loss = criterion(outputs[task][mask], target[mask])
                    batch_loss += task_loss
                    task_metrics[task]["loss"] += task_loss.item()
                    
                    predictions = torch.argmax(outputs[task][mask], dim=1)
                    task_metrics[task]["predictions"].extend(predictions.cpu().numpy())
                    task_metrics[task]["targets"].extend(target[mask].cpu().numpy())
            
            total_loss += batch_loss.item()
    
    # Calculate metrics for each task
    results = {}
    for task in ["company", "model", "diameter", "length"]:
        if len(task_metrics[task]["predictions"]) > 0:
            accuracy = accuracy_score(
                task_metrics[task]["targets"], 
                task_metrics[task]["predictions"]
            )
            avg_loss = task_metrics[task]["loss"] / len(dataloader)
            
            results[task] = {
                "accuracy": accuracy,
                "loss": avg_loss,
                "n_samples": len(task_metrics[task]["predictions"])
            }
            
            # Calculate MAE for diameter and length
            if task in ["diameter", "length"]:
                bins = diameter_bins if task == "diameter" else length_bins
                mae = 0.0
                for pred, target in zip(task_metrics[task]["predictions"], 
                                      task_metrics[task]["targets"]):
                    pred_center = (bins[pred] + bins[pred+1]) / 2
                    target_center = (bins[target] + bins[target+1]) / 2
                    mae += abs(pred_center - target_center)
                mae /= len(task_metrics[task]["predictions"])
                results[task]["mae"] = mae
        else:
            results[task] = {
                "accuracy": 0.0,
                "loss": float('inf'),
                "n_samples": 0
            }
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, results

def parse_args():
    parser = argparse.ArgumentParser(description='Train Unified Classifier for All Tasks')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_unified')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--diameter_bin_size', type=float, default=0.5)
    parser.add_argument('--length_bin_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'fixed'])
    parser.add_argument('--task_weights', type=str, default='1,1,1,1',
                        help='Loss weights for company,model,diameter,length')
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')
    args.task_weights = [float(w) for w in args.task_weights.split(',')]

    print("🎯 Unified Classification Training")
    print("=" * 60)
    print("📊 Training all four tasks simultaneously:")
    print("   • Company identification")
    print("   • Model identification")
    print("   • Diameter classification")
    print("   • Length classification")
    print("=" * 60)
    print("=== TRAINING CONFIGURATION ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using {device} device")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\n📊 Loading training data from {args.train_json}")
    with open(args.train_json) as f:
        train_entries = json.load(f)

    val_entries = None
    if args.val_json and os.path.exists(args.val_json):
        print(f"📊 Loading validation data from {args.val_json}")
        with open(args.val_json) as f:
            val_entries = json.load(f)

    # Build vocabularies and bins
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)
    
    print(f"\n📚 Building vocabularies and bins from {len(all_entries)} total entries")
    
    # Build vocabularies for company and model
    vocabs = {
        "company": build_vocab(all_entries, "company"),
        "model": build_vocab(all_entries, "model")
    }
    
    # Create bins for diameter and length
    diameters = []
    lengths = []
    for e in all_entries:
        if e.get("diameter") is not None:
            try:
                d = float(e["diameter"])
                if d > 0:
                    diameters.append(d)
            except (ValueError, TypeError):
                pass
        if e.get("length") is not None:
            try:
                l = float(e["length"])
                if l > 0:
                    lengths.append(l)
            except (ValueError, TypeError):
                pass
    
    diameter_bins = create_bins(diameters, bin_size=args.diameter_bin_size, strategy='fixed')
    length_bins = create_bins(lengths, bin_size=2.0, strategy=args.length_bin_strategy)
    
    print(f"📏 Diameter bins: {len(diameter_bins)-1} bins")
    print(f"📐 Length bins: {len(length_bins)-1} bins")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImplantDataset(args.train_json, vocabs, diameter_bins, length_bins,
                                 train_transform, image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, vocabs, diameter_bins, length_bins,
                                   val_transform, image_root=args.image_root)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model_map = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224', 
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }

    num_classes = {
        'company': len(vocabs['company']),
        'model': len(vocabs['model']),
        'diameter': len(diameter_bins) - 1,
        'length': len(length_bins) - 1
    }

    model = UnifiedClassifier(model_map[args.model_size], num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print(f"⚖️  Task weights: Company={args.task_weights[0]:.1f}, Model={args.task_weights[1]:.1f}, "
          f"Diameter={args.task_weights[2]:.1f}, Length={args.task_weights[3]:.1f}")
    
    best_val_score = 0.0
    training_history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        task_correct = {"company": 0, "model": 0, "diameter": 0, "length": 0}
        task_total = {"company": 0, "model": 0, "diameter": 0, "length": 0}
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            
            loss = 0.0
            task_weights_dict = {
                "company": args.task_weights[0],
                "model": args.task_weights[1],
                "diameter": args.task_weights[2],
                "length": args.task_weights[3]
            }
            
            # Calculate loss for each task
            for task in ["company", "model", "diameter", "length"]:
                target = labels[task].to(device)
                mask = target != -1
                
                if mask.sum() > 0:
                    task_loss = criterion(outputs[task][mask], target[mask])
                    loss += task_loss * task_weights_dict[task]
                    
                    predictions = torch.argmax(outputs[task][mask], dim=1)
                    task_correct[task] += (predictions == target[mask]).sum().item()
                    task_total[task] += mask.sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar with accuracies
            accuracies = {}
            for task in ["company", "model", "diameter", "length"]:
                if task_total[task] > 0:
                    accuracies[f"{task[:3]}_acc"] = task_correct[task] / task_total[task]
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                **accuracies
            })

        scheduler.step()
        
        # Print training metrics
        print(f"🔼 Train - Loss: {running_loss / len(train_dataloader):.4f}")
        for task in ["company", "model", "diameter", "length"]:
            if task_total[task] > 0:
                acc = task_correct[task] / task_total[task]
                print(f"   {task.capitalize()}: {acc:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_results = evaluate_model(
                model, val_dataloader, criterion, device, vocabs, diameter_bins, length_bins
            )
            
            print(f"🔽 Val - Loss: {val_loss:.4f}")
            
            # Calculate combined score for model selection
            combined_score = 0.0
            for task in ["company", "model", "diameter", "length"]:
                acc = val_results[task]["accuracy"]
                print(f"   {task.capitalize()}: {acc:.4f}", end="")
                if task in ["diameter", "length"] and "mae" in val_results[task]:
                    print(f" (MAE: {val_results[task]['mae']:.2f}mm)")
                else:
                    print()
                combined_score += acc
            combined_score /= 4
            
            # Save best model
            if combined_score > best_val_score:
                best_val_score = combined_score
                print(f"💾 New best combined score: {best_val_score:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_val_score,
                    'val_results': val_results,
                    'vocabs': vocabs,
                    'diameter_bins': diameter_bins.tolist(),
                    'length_bins': length_bins.tolist(),
                    'args': vars(args)
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_dataloader),
                'val_loss': val_loss,
                'val_results': val_results,
                'combined_score': combined_score,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Save final model
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'vocabs': vocabs,
            'diameter_bins': diameter_bins.tolist(),
            'length_bins': length_bins.tolist(),
            'args': vars(args)
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history and model info
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    model_info = {
        "num_classes": num_classes,
        "vocabularies": {k: {v: i for i, v in enumerate(sorted(vocab.keys()))} 
                        for k, vocab in vocabs.items()},
        "diameter_bins": diameter_bins.tolist(),
        "length_bins": length_bins.tolist(),
        "task_weights": args.task_weights,
        "best_combined_score": best_val_score,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }
    
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best combined validation score: {best_val_score:.4f}")

if __name__ == '__main__':
    main()
