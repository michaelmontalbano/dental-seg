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
    def __init__(self, json_path, vocabulary, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.vocab = vocabulary
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution
        models = [e.get("model") for e in self.entries if e.get("model")]
        print(f"📊 Unique models: {len(set(models))}")
        print(f"📊 Model distribution:")
        model_counts = {}
        for m in models:
            model_counts[m] = model_counts.get(m, 0) + 1
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {model}: {count}")

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

        # Return model label
        label = self.vocab.get(entry.get("model"), -1)
        return image, label

class ModelClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Classification head with deeper architecture for complex model variations
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"🏗️  Model Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Model classes: {num_classes}")
        print(f"   Classifier: {dim} -> 512 -> 256 -> {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def build_vocab(entries):
    """Build vocabulary for model classification"""
    models = sorted(set(e["model"] for e in entries if e.get("model") is not None))
    print(f"📋 Model vocabulary: {len(models)} unique models")
    if len(models) <= 20:
        print(f"   Models: {models}")
    else:
        print(f"   Sample models: {models[:10]} ... {models[-5:]}")
    return {k: i for i, k in enumerate(models)}

def inverse_vocab(vocab):
    return {i: k for k, i in vocab.items()}

def evaluate_model(model, dataloader, criterion, device, vocab):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter out invalid labels
            mask = labels != -1
            if mask.sum() == 0:
                continue
                
            outputs = model(images)
            loss = criterion(outputs[mask], labels[mask])
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs[mask], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels[mask].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Get classification report
    inv_vocab = inverse_vocab(vocab)
    label_names = [inv_vocab[i] for i in range(len(vocab))]
    
    report = classification_report(
        all_targets, all_predictions,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )
    
    return avg_loss, accuracy, report, all_predictions, all_targets

def parse_args():
    parser = argparse.ArgumentParser(description='Train Model Classifier')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_model')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')

    print("🔧 Model Classification Training")
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

    # Build vocabulary
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)
    
    print(f"\n📚 Building vocabulary from {len(all_entries)} total entries")
    vocab = build_vocab(all_entries)
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
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
    train_dataset = ImplantDataset(args.train_json, vocab, train_transform, 
                                 image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, vocab, val_transform,
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

    model = ModelClassifier(model_map[args.model_size], len(vocab)).to(device)
    
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
    
    best_val_accuracy = 0.0
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
            
            # Filter out invalid labels
            mask = labels != -1
            if mask.sum() == 0:
                continue
            
            outputs = model(images)
            loss = criterion(outputs[mask], labels[mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
            val_loss, val_accuracy, val_report, _, _ = evaluate_model(
                model, val_dataloader, criterion, device, vocab
            )
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"💾 New best accuracy: {best_val_accuracy:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_accuracy': best_val_accuracy,
                    'vocab': vocab,
                    'args': vars(args)
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_dataloader),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Save final model
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'vocab': vocab,
            'args': vars(args)
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save vocabulary and training history
    with open(os.path.join(args.output_dir, "vocabulary.json"), "w") as f:
        json.dump(inverse_vocab(vocab), f, indent=2)
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()
