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
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
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
    def __init__(self, json_path, company_vocab, model_vocab, transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.company_vocab = company_vocab
        self.model_vocab = model_vocab
        self.transform = transform
        self.image_root = image_root
        
        print(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        
        # Show data distribution
        companies = [e.get("company") for e in self.entries if e.get("company")]
        models = [e.get("model") for e in self.entries if e.get("model")]
        print(f"📊 Unique companies: {len(set(companies))}")
        print(f"📊 Unique models: {len(set(models))}")

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

        # Get labels
        company_label = self.company_vocab.get(entry.get("company"), -1)
        model_label = self.model_vocab.get(entry.get("model"), -1)
        
        return image, company_label, model_label

class AttentionModule(nn.Module):
    """Attention module to weight model features based on company predictions"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, features, company_features):
        # features: [batch, feature_dim]
        # company_features: [batch, feature_dim]
        
        q = self.query(features)
        k = self.key(company_features)
        v = self.value(company_features)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)
        output = self.output(attended)
        
        return output + features  # Residual connection

class HierarchicalClassifier(nn.Module):
    def __init__(self, base_model_name, num_companies, num_models):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        # Company classification head
        self.company_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_companies)
        )
        
        # Feature projection for company features
        self.company_feature_proj = nn.Linear(num_companies, dim)
        
        # Attention module
        self.attention = AttentionModule(dim)
        
        # Model classification head (conditioned on company)
        self.model_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_models)
        )
        
        print(f"🏗️  Hierarchical Classifier Architecture:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {dim}")
        print(f"   Company classes: {num_companies}")
        print(f"   Model classes: {num_models}")
        print(f"   Using attention mechanism for hierarchical structure")

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Company prediction
        company_logits = self.company_head(features)
        company_probs = F.softmax(company_logits, dim=1)
        
        # Get company confidence (max probability)
        company_confidence, _ = torch.max(company_probs, dim=1, keepdim=True)
        
        # Project company predictions to feature space
        company_features = self.company_feature_proj(company_probs)
        
        # Apply attention based on company predictions
        attended_features = self.attention(features, company_features)
        
        # Model prediction (conditioned on company)
        model_logits = self.model_head(attended_features)
        
        return company_logits, model_logits, company_confidence

def build_vocab(entries, field):
    """Build vocabulary for a specific field"""
    values = sorted(set(e[field] for e in entries if e.get(field) is not None))
    print(f"📋 {field} vocabulary: {len(values)} unique values")
    if len(values) <= 20:
        print(f"   {field}s: {values}")
    else:
        print(f"   Sample {field}s: {values[:10]} ... {values[-5:]}")
    return {k: i for i, k in enumerate(values)}

def inverse_vocab(vocab):
    return {i: k for k, i in vocab.items()}

def evaluate_model(model, dataloader, company_criterion, model_criterion, device, 
                  company_vocab, model_vocab, hierarchy_weight):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    company_total_loss = 0.0
    model_total_loss = 0.0
    
    company_predictions = []
    company_targets = []
    model_predictions = []
    model_targets = []
    
    with torch.no_grad():
        for images, company_labels, model_labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            company_labels = company_labels.to(device)
            model_labels = model_labels.to(device)
            
            # Forward pass
            company_logits, model_logits, company_confidence = model(images)
            
            # Company loss
            company_mask = company_labels != -1
            if company_mask.sum() > 0:
                company_loss = company_criterion(company_logits[company_mask], company_labels[company_mask])
                company_total_loss += company_loss.item()
                
                company_pred = torch.argmax(company_logits[company_mask], dim=1)
                company_predictions.extend(company_pred.cpu().numpy())
                company_targets.extend(company_labels[company_mask].cpu().numpy())
            
            # Model loss (weighted by company confidence)
            model_mask = model_labels != -1
            if model_mask.sum() > 0:
                # Apply confidence weighting
                confidence_weights = company_confidence[model_mask].squeeze()
                model_loss_unreduced = F.cross_entropy(
                    model_logits[model_mask], 
                    model_labels[model_mask], 
                    reduction='none'
                )
                model_loss = (confidence_weights * model_loss_unreduced).mean()
                model_total_loss += model_loss.item()
                
                model_pred = torch.argmax(model_logits[model_mask], dim=1)
                model_predictions.extend(model_pred.cpu().numpy())
                model_targets.extend(model_labels[model_mask].cpu().numpy())
            
            # Combined loss
            if company_mask.sum() > 0 and model_mask.sum() > 0:
                total_loss += company_loss.item() + hierarchy_weight * model_loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    company_accuracy = accuracy_score(company_targets, company_predictions) if company_targets else 0
    model_accuracy = accuracy_score(model_targets, model_predictions) if model_targets else 0
    
    return {
        'loss': avg_loss,
        'company_loss': company_total_loss / len(dataloader),
        'model_loss': model_total_loss / len(dataloader),
        'company_accuracy': company_accuracy,
        'model_accuracy': model_accuracy
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train Hierarchical Company-Model Classifier')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_hierarchical')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--hierarchy_weight', type=float, default=0.5,
                        help='Weight for model loss in hierarchical structure')
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                        help='Minimum company confidence for full model loss weight')
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')

    print("🔧 Hierarchical Company-Model Classification Training")
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
    
    print(f"\n📚 Building vocabularies from {len(all_entries)} total entries")
    company_vocab = build_vocab(all_entries, 'company')
    model_vocab = build_vocab(all_entries, 'model')
    
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

    model = HierarchicalClassifier(
        model_map[args.model_size], 
        len(company_vocab), 
        len(model_vocab)
    ).to(device)
    
    # Optimizer and losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    company_criterion = nn.CrossEntropyLoss()
    model_criterion = nn.CrossEntropyLoss(reduction='none')  # For confidence weighting
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print(f"   Hierarchy weight: {args.hierarchy_weight}")
    print(f"   Confidence threshold: {args.confidence_threshold}")
    
    best_val_accuracy = 0.0
    best_company_accuracy = 0.0
    best_model_accuracy = 0.0
    training_history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        company_correct = 0
        model_correct = 0
        company_total = 0
        model_total = 0
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, company_labels, model_labels in pbar:
            images = images.to(device)
            company_labels = company_labels.to(device)
            model_labels = model_labels.to(device)
            
            # Forward pass
            company_logits, model_logits, company_confidence = model(images)
            
            # Company loss
            company_mask = company_labels != -1
            company_loss = torch.tensor(0.0).to(device)
            if company_mask.sum() > 0:
                company_loss = company_criterion(company_logits[company_mask], company_labels[company_mask])
                company_pred = torch.argmax(company_logits[company_mask], dim=1)
                company_correct += (company_pred == company_labels[company_mask]).sum().item()
                company_total += company_mask.sum().item()
            
            # Model loss (weighted by company confidence)
            model_mask = model_labels != -1
            model_loss = torch.tensor(0.0).to(device)
            if model_mask.sum() > 0:
                # Apply confidence weighting
                confidence_weights = company_confidence[model_mask].squeeze()
                # Apply threshold
                confidence_weights = torch.where(
                    confidence_weights > args.confidence_threshold,
                    confidence_weights,
                    confidence_weights * 0.5  # Reduce weight for low confidence
                )
                model_loss_unreduced = model_criterion(model_logits[model_mask], model_labels[model_mask])
                model_loss = (confidence_weights * model_loss_unreduced).mean()
                
                model_pred = torch.argmax(model_logits[model_mask], dim=1)
                model_correct += (model_pred == model_labels[model_mask]).sum().item()
                model_total += model_mask.sum().item()
            
            # Combined loss
            loss = company_loss + args.hierarchy_weight * model_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'comp_acc': company_correct / company_total if company_total > 0 else 0,
                'model_acc': model_correct / model_total if model_total > 0 else 0
            })

        scheduler.step()
        
        train_company_accuracy = company_correct / company_total if company_total > 0 else 0
        train_model_accuracy = model_correct / model_total if model_total > 0 else 0
        train_loss = running_loss / len(train_dataloader)
        print(f"🔼 Train - Loss: {train_loss:.4f}, Company Acc: {train_company_accuracy:.4f}, Model Acc: {train_model_accuracy:.4f}")
        
        # Report training metrics for HPO
        report_metric('train_loss', train_loss, epoch + 1)
        report_metric('train_company_accuracy', train_company_accuracy, epoch + 1)
        report_metric('train_model_accuracy', train_model_accuracy, epoch + 1)
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_metrics = evaluate_model(
                model, val_dataloader, company_criterion, model_criterion, 
                device, company_vocab, model_vocab, args.hierarchy_weight
            )
            
            print(f"🔽 Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Company Acc: {val_metrics['company_accuracy']:.4f}, "
                  f"Model Acc: {val_metrics['model_accuracy']:.4f}")
            
            # Report validation metrics for HPO
            report_metric('val_loss', val_metrics['loss'], epoch + 1)
            report_metric('val_company_accuracy', val_metrics['company_accuracy'], epoch + 1)
            report_metric('val_model_accuracy', val_metrics['model_accuracy'], epoch + 1)
            
            # Combined accuracy for best model selection
            combined_accuracy = (val_metrics['company_accuracy'] + val_metrics['model_accuracy']) / 2
            
            # Save best model
            if combined_accuracy > best_val_accuracy:
                best_val_accuracy = combined_accuracy
                best_company_accuracy = val_metrics['company_accuracy']
                best_model_accuracy = val_metrics['model_accuracy']
                print(f"💾 New best combined accuracy: {best_val_accuracy:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_combined_accuracy': best_val_accuracy,
                    'best_company_accuracy': best_company_accuracy,
                    'best_model_accuracy': best_model_accuracy,
                    'company_vocab': company_vocab,
                    'model_vocab': model_vocab,
                    'args': vars(args)
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_company_accuracy': train_company_accuracy,
                'train_model_accuracy': train_model_accuracy,
                'val_loss': val_metrics['loss'],
                'val_company_accuracy': val_metrics['company_accuracy'],
                'val_model_accuracy': val_metrics['model_accuracy'],
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
    
    # Save vocabularies and training history
    with open(os.path.join(args.output_dir, "company_vocabulary.json"), "w") as f:
        json.dump(inverse_vocab(company_vocab), f, indent=2)
    
    with open(os.path.join(args.output_dir, "model_vocabulary.json"), "w") as f:
        json.dump(inverse_vocab(model_vocab), f, indent=2)
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation accuracies:")
    print(f"   Combined: {best_val_accuracy:.4f}")
    print(f"   Company: {best_company_accuracy:.4f}")
    print(f"   Model: {best_model_accuracy:.4f}")
    
    # Report final best metrics for HPO
    report_metric('best_combined_accuracy', best_val_accuracy)
    report_metric('best_company_accuracy', best_company_accuracy)
    report_metric('best_model_accuracy', best_model_accuracy)

if __name__ == '__main__':
    main()
