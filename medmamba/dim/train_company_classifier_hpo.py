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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def report_metric(metric_name, value, epoch=None):
    """Report metric to SageMaker and CloudWatch"""
    if epoch is not None:
        # Format for SageMaker hyperparameter tuning
        print(f"[{epoch}] {metric_name} = {value:.6f};")
        logger.info(f"Epoch {epoch}: {metric_name} = {value:.6f}")
    else:
        print(f"{metric_name} = {value:.6f};")
        logger.info(f"{metric_name} = {value:.6f}")

class ImplantDataset(Dataset):
    def __init__(self, json_path, vocabulary, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.vocab = vocabulary
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution
        companies = [e.get("company") for e in self.entries if e.get("company")]
        print(f"📊 Unique companies: {len(set(companies))}")
        print(f"📊 Company distribution:")
        company_counts = {}
        for c in companies:
            company_counts[c] = company_counts.get(c, 0) + 1
        for company, count in sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {company}: {count}")

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

        # Return company label
        label = self.vocab.get(entry.get("company"), -1)
        return image, label

class CompanyClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Classification head with deeper architecture for complex company variations
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"🏗️  Company Classifier Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Company classes: {num_classes}")
        print(f"   Classifier: {dim} -> 512 -> 256 -> {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def build_vocab(entries):
    """Build vocabulary for company classification"""
    companies = sorted(set(e["company"] for e in entries if e.get("company") is not None))
    print(f"📋 Company vocabulary: {len(companies)} unique companies")
    if len(companies) <= 20:
        print(f"   Companies: {companies}")
    else:
        print(f"   Sample companies: {companies[:10]} ... {companies[-5:]}")
    return {k: i for i, k in enumerate(companies)}

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
    parser = argparse.ArgumentParser(description='Train Company Classifier with HPO Support')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_company')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')

    print("🔧 Company Classification Training with HPO Support")
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

    model = CompanyClassifier(model_map[args.model_size], len(vocab)).to(device)
    
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
        train_loss = running_loss / len(train_dataloader)
        print(f"🔼 Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Report training metrics for HPO
        report_metric('train_loss', train_loss, epoch + 1)
        report_metric('train_accuracy', train_accuracy, epoch + 1)
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_loss, val_accuracy, val_report, _, _ = evaluate_model(
                model, val_dataloader, criterion, device, vocab
            )
            
            print(f"🔽 Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Report validation metrics for HPO
            report_metric('val_loss', val_loss, epoch + 1)
            report_metric('val_accuracy', val_accuracy, epoch + 1)
            
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
                'train_loss': train_loss,
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
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(convert_to_serializable(training_history), f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Report final best metric for HPO
    report_metric('best_val_accuracy', best_val_accuracy)

if __name__ == '__main__':
    main()
