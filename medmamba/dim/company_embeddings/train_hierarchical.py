import argparse
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class ImplantDataset(Dataset):
    def __init__(self, json_path, company_vocab, model_vocab, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.company_vocab = company_vocab
        self.model_vocab = model_vocab
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Filter entries that have both company and model
        valid_entries = []
        for e in self.entries:
            if e.get("company") and e.get("model"):
                valid_entries.append(e)
        
        self.entries = valid_entries
        print(f"📊 {len(self.entries)} entries have both company and model labels")
        
        # Show data distribution
        companies = [e["company"] for e in self.entries]
        models = [e["model"] for e in self.entries]
        print(f"📊 Companies: {len(set(companies))} unique")
        print(f"📊 Models: {len(set(models))} unique")

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

        company_label = self.company_vocab.get(entry["company"], -1)
        model_label = self.model_vocab.get(entry["model"], -1)
            
        return image, company_label, model_label

class CompanyEmbeddingModel(nn.Module):
    def __init__(self, base_model_name, num_companies, embed_dim=256):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.embed_dim = embed_dim
        
        # Company classification branch
        self.company_head = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),  # This becomes our embedding layer
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final company classifier (after embeddings)
        self.company_classifier = nn.Linear(embed_dim, num_companies)
        
        print(f"🏗️  Company Embedding Model:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {self.feat_dim}")
        print(f"   Embedding dim: {embed_dim}")
        print(f"   Companies: {num_companies}")

    def forward(self, x):
        features = self.backbone(x)
        company_embed = self.company_head(features)
        company_logits = self.company_classifier(company_embed)
        return company_logits, company_embed, features

class HierarchicalModelClassifier(nn.Module):
    def __init__(self, base_model_name, num_companies, num_models, embed_dim=256, use_attention=True):
        super().__init__()
        
        # Company embedding model
        self.company_model = CompanyEmbeddingModel(base_model_name, num_companies, embed_dim)
        
        # Freeze company model initially (will unfreeze later for fine-tuning)
        for param in self.company_model.parameters():
            param.requires_grad = False
        
        self.use_attention = use_attention
        feat_dim = self.company_model.feat_dim
        
        if use_attention:
            # Attention mechanism to weight company embedding influence
            self.attention = nn.Sequential(
                nn.Linear(feat_dim + embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2),  # 2 weights: one for features, one for embeddings
                nn.Softmax(dim=1)
            )
        
        # Model classification head (uses both features and company embeddings)
        input_dim = feat_dim + embed_dim
        self.model_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_models)
        )
        
        print(f"🏗️  Hierarchical Model Classifier:")
        print(f"   Company embeddings: {embed_dim}")
        print(f"   Model classes: {num_models}")
        print(f"   Use attention: {use_attention}")

    def forward(self, x, return_embeddings=False):
        # Get company predictions and embeddings
        company_logits, company_embed, features = self.company_model(x)
        
        # Concatenate features and company embeddings
        combined = torch.cat([features, company_embed], dim=1)
        
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(combined)
            
            # Apply attention weights
            weighted_features = features * attention_weights[:, 0:1]
            weighted_embed = company_embed * attention_weights[:, 1:2]
            combined = torch.cat([weighted_features, weighted_embed], dim=1)
        
        # Model classification
        model_logits = self.model_head(combined)
        
        if return_embeddings:
            return company_logits, model_logits, company_embed
        return company_logits, model_logits

    def unfreeze_company_model(self, unfreeze_ratio=0.5):
        """Gradually unfreeze company model layers"""
        all_params = list(self.company_model.parameters())
        n_params = len(all_params)
        n_unfreeze = int(n_params * unfreeze_ratio)
        
        # Unfreeze the last n_unfreeze parameters
        for param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        
        print(f"📌 Unfroze {n_unfreeze}/{n_params} company model parameters")

def build_vocab(entries, key):
    unique = sorted(set(e[key] for e in entries if e.get(key) is not None))
    print(f"📋 {key} vocabulary: {len(unique)} unique values")
    return {k: i for i, k in enumerate(unique)}

def evaluate_model(model, dataloader, company_criterion, model_criterion, device, 
                  company_names, model_names):
    """Evaluate hierarchical model"""
    model.eval()
    
    company_loss = 0.0
    model_loss = 0.0
    
    company_preds = []
    company_targets = []
    model_preds = []
    model_targets = []
    
    with torch.no_grad():
        for images, company_labels, model_labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            company_labels = company_labels.to(device)
            model_labels = model_labels.to(device)
            
            company_logits, model_logits = model(images)
            
            # Company loss
            c_loss = company_criterion(company_logits, company_labels)
            company_loss += c_loss.item()
            
            # Model loss
            m_loss = model_criterion(model_logits, model_labels)
            model_loss += m_loss.item()
            
            # Predictions
            company_pred = torch.argmax(company_logits, dim=1)
            model_pred = torch.argmax(model_logits, dim=1)
            
            company_preds.extend(company_pred.cpu().numpy())
            company_targets.extend(company_labels.cpu().numpy())
            model_preds.extend(model_pred.cpu().numpy())
            model_targets.extend(model_labels.cpu().numpy())
    
    # Calculate metrics
    company_acc = accuracy_score(company_targets, company_preds)
    model_acc = accuracy_score(model_targets, model_preds)
    
    avg_company_loss = company_loss / len(dataloader)
    avg_model_loss = model_loss / len(dataloader)
    
    # Classification reports
    company_report = classification_report(
        company_targets, company_preds,
        target_names=company_names[:len(set(company_targets))],
        zero_division=0,
        output_dict=True
    )
    
    model_report = classification_report(
        model_targets, model_preds,
        target_names=model_names[:len(set(model_targets))],
        zero_division=0,
        output_dict=True
    )
    
    return {
        'company_loss': avg_company_loss,
        'model_loss': avg_model_loss,
        'company_acc': company_acc,
        'model_acc': model_acc,
        'company_report': company_report,
        'model_report': model_report
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train Hierarchical Model with Company Embeddings')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_hierarchical')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--use_attention', type=str, default='true')
    parser.add_argument('--pretrain_company_epochs', type=int, default=20,
                        help='Epochs to pretrain company model before joint training')
    parser.add_argument('--unfreeze_epoch', type=int, default=40,
                        help='Epoch to start unfreezing company model')
    parser.add_argument('--company_checkpoint', type=str, default=None,
                        help='Path to pretrained company model checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')
    args.use_attention = args.use_attention.lower() in ('true', '1', 'yes')

    print("🎯 Hierarchical Classification with Company Embeddings")
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

    # Build vocabularies
    all_entries = train_entries[:]
    if val_entries:
        all_entries.extend(val_entries)
    
    # Filter for entries with both company and model
    filtered_entries = [e for e in all_entries if e.get("company") and e.get("model")]
    print(f"\n📚 Building vocabularies from {len(filtered_entries)} entries with both labels")
    
    company_vocab = build_vocab(filtered_entries, "company")
    model_vocab = build_vocab(filtered_entries, "model")
    
    # Get label names for reports
    company_names = sorted(company_vocab.keys(), key=lambda x: company_vocab[x])
    model_names = sorted(model_vocab.keys(), key=lambda x: model_vocab[x])
    
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
    train_dataset = ImplantDataset(args.train_json, company_vocab, model_vocab,
                                 train_transform, image_root=args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantDataset(args.val_json, company_vocab, model_vocab,
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

    model = HierarchicalModelClassifier(
        model_map[args.model_size], 
        len(company_vocab), 
        len(model_vocab),
        embed_dim=args.embed_dim,
        use_attention=args.use_attention
    ).to(device)
    
    # Load pretrained company model if provided
    if args.company_checkpoint:
        print(f"📥 Loading pretrained company model from {args.company_checkpoint}")
        checkpoint = torch.load(args.company_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Format 1: Dictionary with 'model_state_dict' key
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and not any(k.startswith('blocks.') for k in checkpoint.keys()):
            # Format 2: Dictionary but not a state dict (has metadata)
            state_dict = checkpoint
        else:
            # Format 3: Raw state dict (OrderedDict)
            state_dict = checkpoint
        
        # Map the ViT backbone weights to our CompanyEmbeddingModel
        # The checkpoint has standard ViT layers, we need to map them to our model structure
        mapped_state_dict = {}
        
        for key, value in state_dict.items():
            # Map backbone layers
            if key.startswith(('cls_token', 'pos_embed', 'patch_embed', 'blocks', 'norm')):
                # These go into the backbone
                mapped_state_dict[f'backbone.{key}'] = value
            elif key == 'head.weight':
                # The original head.weight [48, 768] needs to map to our classifier structure
                # We'll use it to initialize the first layer of our classifier
                # Our first layer is Linear(768, 512), so we'll need to handle dimension mismatch
                print(f"⚠️  Original head layer has shape {value.shape}, skipping direct mapping")
            elif key == 'head.bias':
                # Skip the original head bias
                pass
            else:
                # Try to map as-is
                mapped_state_dict[key] = value
        
        # Load the mapped weights, allowing missing keys for the custom classifier layers
        missing_keys, unexpected_keys = model.company_model.load_state_dict(mapped_state_dict, strict=False)
        
        print(f"✅ Loaded pretrained weights")
        if missing_keys:
            print(f"ℹ️  Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            # Show first few missing keys
            for key in missing_keys[:5]:
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    # Optimizer and loss functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    company_criterion = nn.CrossEntropyLoss()
    model_criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    if args.pretrain_company_epochs > 0:
        print(f"📌 First {args.pretrain_company_epochs} epochs: company model only")
        print(f"📌 Unfreezing company model at epoch {args.unfreeze_epoch}")
    
    best_model_acc = 0.0
    best_combined_acc = 0.0
    training_history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Unfreeze company model gradually
        if epoch == args.unfreeze_epoch:
            model.unfreeze_company_model(unfreeze_ratio=0.5)
        elif epoch == args.unfreeze_epoch + 10:
            model.unfreeze_company_model(unfreeze_ratio=1.0)
        
        # Training phase
        model.train()
        running_company_loss = 0.0
        running_model_loss = 0.0
        company_correct = 0
        model_correct = 0
        total = 0
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, company_labels, model_labels in pbar:
            images = images.to(device)
            company_labels = company_labels.to(device)
            model_labels = model_labels.to(device)
            
            company_logits, model_logits = model(images)
            
            # Calculate losses
            company_loss = company_criterion(company_logits, company_labels)
            model_loss = model_criterion(model_logits, model_labels)
            
            # Combined loss (weight model loss higher after pretraining)
            if epoch < args.pretrain_company_epochs:
                # Focus on company classification initially
                total_loss = company_loss + 0.1 * model_loss
            else:
                # Balance both tasks
                total_loss = 0.5 * company_loss + model_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            running_company_loss += company_loss.item()
            running_model_loss += model_loss.item()
            
            company_pred = torch.argmax(company_logits, dim=1)
            model_pred = torch.argmax(model_logits, dim=1)
            
            company_correct += (company_pred == company_labels).sum().item()
            model_correct += (model_pred == model_labels).sum().item()
            total += company_labels.size(0)
            
            pbar.set_postfix({
                'c_loss': running_company_loss / (pbar.n + 1),
                'm_loss': running_model_loss / (pbar.n + 1),
                'c_acc': company_correct / total,
                'm_acc': model_correct / total
            })

        scheduler.step()
        
        train_company_acc = company_correct / total
        train_model_acc = model_correct / total
        
        print(f"🔼 Train - Company Loss: {running_company_loss / len(train_dataloader):.4f}, "
              f"Acc: {train_company_acc:.4f}")
        print(f"         Model Loss: {running_model_loss / len(train_dataloader):.4f}, "
              f"Acc: {train_model_acc:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_results = evaluate_model(
                model, val_dataloader, company_criterion, model_criterion, 
                device, company_names, model_names
            )
            
            print(f"🔽 Val - Company Loss: {val_results['company_loss']:.4f}, "
                  f"Acc: {val_results['company_acc']:.4f}")
            print(f"       Model Loss: {val_results['model_loss']:.4f}, "
                  f"Acc: {val_results['model_acc']:.4f}")
            
            # Save best model based on model accuracy (our primary task)
            if val_results['model_acc'] > best_model_acc:
                best_model_acc = val_results['model_acc']
                best_combined_acc = (val_results['company_acc'] + val_results['model_acc']) / 2
                print(f"💾 New best model accuracy: {best_model_acc:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_model_acc': best_model_acc,
                    'best_combined_acc': best_combined_acc,
                    'company_vocab': company_vocab,
                    'model_vocab': model_vocab,
                    'args': vars(args),
                    'val_results': val_results
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_company_loss': running_company_loss / len(train_dataloader),
                'train_model_loss': running_model_loss / len(train_dataloader),
                'train_company_acc': train_company_acc,
                'train_model_acc': train_model_acc,
                'val_results': val_results,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Save final model
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'company_vocab': company_vocab,
            'model_vocab': model_vocab,
            'args': vars(args)
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best model accuracy: {best_model_acc:.4f}")
    print(f"📊 Best combined accuracy: {best_combined_acc:.4f}")

if __name__ == '__main__':
    main()
