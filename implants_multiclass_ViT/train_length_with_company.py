import argparse
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class ImplantDataset(Dataset):
    def __init__(self, json_path, bin_edges, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.bin_edges = bin_edges
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution
        lengths = []
        for e in self.entries:
            if e.get("length") is not None:
                try:
                    l = float(e["length"])
                    if l > 0:  # Valid length
                        lengths.append(l)
                except (ValueError, TypeError):
                    pass
        
        if lengths:
            print(f"📊 Length statistics:")
            print(f"   Samples: {len(lengths)}")
            print(f"   Range: {min(lengths):.1f} - {max(lengths):.1f} mm")
            print(f"   Mean: {np.mean(lengths):.1f} mm")
            print(f"   Std: {np.std(lengths):.1f} mm")
            
            # Show bin distribution
            bin_counts = np.histogram(lengths, bins=self.bin_edges)[0]
            print(f"📊 Bin distribution:")
            for i, count in enumerate(bin_counts):
                print(f"   [{self.bin_edges[i]:.1f}, {self.bin_edges[i+1]:.1f}): {count} samples")

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

        # Convert length to bin label
        length = entry.get("length", -1)
        try:
            length = float(length)
            if length > 0:
                # Find which bin the length belongs to
                # np.digitize returns 0 if x < bins[0], n if x >= bins[n-1]
                label = np.digitize(length, self.bin_edges) - 1
                # Ensure label is within valid range [0, num_bins-1]
                label = max(0, min(label, len(self.bin_edges) - 2))
            else:
                label = -1
        except (ValueError, TypeError):
            label = -1
            
        return image, label

class CompanyModel(nn.Module):
    """Pretrained company model used for embeddings"""
    def __init__(self, base_model_name, num_companies):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_companies)
        )

    def forward(self, x, return_embeddings=True):
        features = self.backbone(x)
        
        # Get embeddings from intermediate layers
        x = self.classifier[0](features)  # Linear(768, 512)
        x = self.classifier[1](x)         # ReLU
        x = self.classifier[2](x)         # Dropout
        embeddings = self.classifier[3](x) # Linear(512, 256) - This is our embedding
        
        if return_embeddings:
            return features, embeddings
        
        # Complete forward pass for company prediction
        x = self.classifier[4](embeddings)  # ReLU
        x = self.classifier[5](embeddings)  # Dropout
        logits = self.classifier[6](embeddings)  # Linear(256, num_companies)
        return logits

class LengthWithCompanyClassifier(nn.Module):
    def __init__(self, company_checkpoint_path, num_length_classes, use_attention=True):
        super().__init__()
        
        # Load pretrained company model
        print(f"📥 Loading pretrained company model from {company_checkpoint_path}")
        checkpoint = torch.load(company_checkpoint_path, map_location='cpu')
        
        # Get model configuration from checkpoint
        vocab = checkpoint.get('vocab', {})
        num_companies = len(vocab)
        model_args = checkpoint.get('args', {})
        model_size = model_args.get('model_size', 'base')
        
        model_map = {
            'tiny': 'vit_tiny_patch16_224',
            'small': 'vit_small_patch16_224', 
            'base': 'vit_base_patch16_224',
            'large': 'vit_large_patch16_224'
        }
        
        # Create and load company model
        self.company_model = CompanyModel(model_map[model_size], num_companies)
        self.company_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze company model
        for param in self.company_model.parameters():
            param.requires_grad = False
        
        print(f"✅ Loaded company model with {num_companies} companies")
        
        # Get dimensions
        self.feat_dim = self.company_model.backbone.num_features  # 768 for base
        self.embed_dim = 256  # Company embedding dimension
        self.use_attention = use_attention
        
        if use_attention:
            # Attention mechanism to weight company embedding influence
            self.attention = nn.Sequential(
                nn.Linear(self.feat_dim + self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2),  # 2 weights: one for features, one for embeddings
                nn.Softmax(dim=1)
            )
        
        # Length classification head (uses both features and company embeddings)
        input_dim = self.feat_dim + self.embed_dim
        self.length_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_length_classes)
        )
        
        print(f"🏗️  Length with Company Classifier:")
        print(f"   Company embeddings: {self.embed_dim}")
        print(f"   Length bins: {num_length_classes}")
        print(f"   Use attention: {use_attention}")

    def forward(self, x):
        # Get company features and embeddings (no gradients)
        with torch.no_grad():
            features, company_embed = self.company_model(x, return_embeddings=True)
        
        # Concatenate features and company embeddings
        combined = torch.cat([features, company_embed], dim=1)
        
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(combined)
            
            # Apply attention weights
            weighted_features = features * attention_weights[:, 0:1]
            weighted_embed = company_embed * attention_weights[:, 1:2]
            combined = torch.cat([weighted_features, weighted_embed], dim=1)
        
        # Length classification
        length_logits = self.length_head(combined)
        
        return length_logits

def create_length_bins(entries, bin_strategy='adaptive', n_bins=10):
    """Create length bins based on data distribution"""
    lengths = []
    for e in entries:
        if e.get("length") is not None:
            try:
                l = float(e["length"])
                if l > 0:
                    lengths.append(l)
            except (ValueError, TypeError):
                pass
    
    if not lengths:
        raise ValueError("No valid length values found in dataset")
    
    if bin_strategy == 'adaptive':
        # Create adaptive bins based on quantiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(lengths, percentiles)
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
    else:
        # Fixed size bins with buffer
        min_l = min(lengths)
        max_l = max(lengths)
        buffer = (max_l - min_l) * 0.05  # 5% buffer
        bin_edges = np.linspace(min_l - buffer, max_l + buffer, n_bins + 1)
    
    print(f"📏 Created {len(bin_edges)-1} length bins")
    print(f"   Strategy: {bin_strategy}")
    print(f"   Range: {bin_edges[0]:.1f} - {bin_edges[-1]:.1f} mm")
    print(f"   Data range: {min(lengths):.1f} - {max(lengths):.1f} mm")
    print(f"   Number of unique values: {len(set(lengths))}")
    
    return bin_edges

def bin_label_to_name(label, bin_edges):
    """Convert bin label to human-readable name"""
    if label < 0 or label >= len(bin_edges) - 1:
        return "Unknown"
    return f"{bin_edges[label]:.1f}-{bin_edges[label+1]:.1f}mm"

def evaluate_model(model, dataloader, criterion, device, bin_edges):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_classes = len(bin_edges) - 1
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter out invalid labels AND labels outside the valid range
            mask = (labels >= 0) & (labels < num_classes)
            if mask.sum() == 0:
                continue
            
            # Debug: Check for any out-of-range labels
            if (labels[labels >= 0] >= num_classes).any():
                invalid_labels = labels[labels >= num_classes].cpu().numpy()
                print(f"⚠️  Warning: Found labels outside valid range [0, {num_classes-1}]: {invalid_labels}")
                
            outputs = model(images)
            loss = criterion(outputs[mask], labels[mask])
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs[mask], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels[mask].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Get classification report with bin names
    # Only include labels that actually appear in the data
    unique_labels = sorted(set(all_targets + all_predictions))
    label_names = [bin_label_to_name(i, bin_edges) for i in unique_labels]
    
    report = classification_report(
        all_targets, all_predictions,
        labels=unique_labels,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )
    
    # Calculate mean absolute error in mm
    mae_mm = 0.0
    for pred, target in zip(all_predictions, all_targets):
        # Get center of predicted and target bins
        pred_center = (bin_edges[pred] + bin_edges[pred+1]) / 2
        target_center = (bin_edges[target] + bin_edges[target+1]) / 2
        mae_mm += abs(pred_center - target_center)
    mae_mm /= len(all_predictions)
    
    return avg_loss, accuracy, report, all_predictions, all_targets, mae_mm

def parse_args():
    parser = argparse.ArgumentParser(description='Train Length Classifier with Company Embeddings')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--company_checkpoint', type=str, 
                        default='/home/ec2-user/work/repos/models/company_vit/vit_company.pth',
                        help='Path to pretrained company model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_length_company')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--bin_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'fixed'],
                        help='Binning strategy for length classification')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of bins for length classification')
    parser.add_argument('--use_attention', type=str, default='true',
                        help='Use attention mechanism for feature weighting')
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')
    args.use_attention = args.use_attention.lower() in ('true', '1', 'yes')

    print("📐 Length Classification with Company Embeddings")
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

    # Create length bins
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)
    
    print(f"\n📐 Creating length bins from {len(all_entries)} total entries")
    bin_edges = create_length_bins(all_entries, bin_strategy=args.bin_strategy, n_bins=args.n_bins)
    num_classes = len(bin_edges) - 1
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImplantDataset(args.train_json, bin_edges, train_transform, 
                                 image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, bin_edges, val_transform,
                                   image_root=args.image_root)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # Create model with company embeddings
    model = LengthWithCompanyClassifier(
        args.company_checkpoint, 
        num_classes,
        use_attention=args.use_attention
    ).to(device)
    
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
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    print(f"⚡ Mixed precision training enabled")

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    
    best_val_accuracy = 0.0
    best_val_mae = float('inf')
    training_history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter out invalid labels AND labels outside the valid range
            mask = (labels >= 0) & (labels < num_classes)
            if mask.sum() == 0:
                continue
            
            # Debug check during training
            if (labels >= 0).any() and (labels[labels >= 0] >= num_classes).any():
                invalid_labels = labels[labels >= num_classes].cpu().numpy()
                print(f"⚠️  Training: Found labels outside valid range [0, {num_classes-1}]: {invalid_labels}")
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs[mask], labels[mask])
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            predictions = torch.argmax(outputs[mask], dim=1)
            correct += (predictions == labels[mask]).sum().item()
            total += mask.sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct / total if total > 0 else 0
            })

        scheduler.step()
        
        train_accuracy = correct / total if total > 0 else 0
        print(f"🔼 Train - Loss: {running_loss / len(train_dataloader):.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_accuracy, val_report, _, _, val_mae = evaluate_model(
                model, val_dataloader, criterion, device, bin_edges
            )
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, MAE: {val_mae:.2f}mm")
            
            # Save best model (based on MAE)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_accuracy = val_accuracy
                print(f"💾 New best MAE: {best_val_mae:.2f}mm! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_accuracy': best_val_accuracy,
                    'best_mae': best_val_mae,
                    'bin_edges': bin_edges.tolist(),
                    'args': vars(args)
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_dataloader),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_mae': val_mae,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Save final model
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'bin_edges': bin_edges.tolist(),
            'args': vars(args)
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save bin information and training history
    bin_info = {
        'bin_edges': bin_edges.tolist(),
        'bin_strategy': args.bin_strategy,
        'num_bins': num_classes,
        'bin_labels': [bin_label_to_name(i, bin_edges) for i in range(num_classes)]
    }
    
    with open(os.path.join(args.output_dir, "bin_info.json"), "w") as f:
        json.dump(bin_info, f, indent=2)
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"📐 Best validation MAE: {best_val_mae:.2f}mm")
    print(f"🏢 Used company embeddings from: {args.company_checkpoint}")

if __name__ == '__main__':
    main()
