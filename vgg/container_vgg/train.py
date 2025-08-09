#!/usr/bin/env python3
"""
VGG training script for implant company and model-type identification
Self-contained with defaults - no external hyperparameters needed
"""

import argparse
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sagemaker_training import environment
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImplantDataset(Dataset):
    """Dataset for implant company and model-type classification"""
    
    def __init__(self, json_path: str, transform=None, image_root: Optional[str] = None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.transform = transform
        self.image_root = image_root
        
        # Create label mappings
        self.company_to_idx = {}
        self.model_to_idx = {}
        
        companies = set()
        models = set()
        
        for entry in self.entries:
            if 'company' in entry:
                companies.add(entry['company'])
            if 'model' in entry:
                models.add(entry['model'])
        
        # Sort for consistent ordering
        self.company_to_idx = {comp: idx for idx, comp in enumerate(sorted(companies))}
        self.model_to_idx = {model: idx for idx, model in enumerate(sorted(models))}
        
        self.idx_to_company = {idx: comp for comp, idx in self.company_to_idx.items()}
        self.idx_to_model = {idx: model for model, idx in self.model_to_idx.items()}
        
        logger.info(f"📊 Loaded {len(self.entries)} entries from {json_path}")
        logger.info(f"📊 Found {len(self.company_to_idx)} unique companies")
        logger.info(f"📊 Found {len(self.model_to_idx)} unique models")
        
        # Show data distribution
        self._show_distribution()
    
    def _show_distribution(self):
        """Show distribution of companies and models"""
        company_counts = {}
        model_counts = {}
        
        for entry in self.entries:
            company = entry.get('company', 'Unknown')
            model = entry.get('model', 'Unknown')
            
            company_counts[company] = company_counts.get(company, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Show top 5 companies
        top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("📊 Top 5 companies:")
        for comp, count in top_companies:
            logger.info(f"   {comp}: {count} samples")
        
        # Show top 5 models
        top_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("📊 Top 5 models:")
        for model, count in top_models:
            logger.info(f"   {model}: {count} samples")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Handle image path
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
            logger.error(f"❌ Error loading image {image_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
        
        # Get labels
        company = entry.get('company', 'Unknown')
        model = entry.get('model', 'Unknown')
        
        company_idx = self.company_to_idx.get(company, -1)
        model_idx = self.model_to_idx.get(model, -1)
        
        # Also return optional regression targets if available
        length = float(entry.get('length', -1)) if entry.get('length') is not None else -1.0
        diameter = float(entry.get('diameter', -1)) if entry.get('diameter') is not None else -1.0
        
        return image, {
            'company': company_idx,
            'model': model_idx,
            'length': length,
            'diameter': diameter
        }

class VGGMultiTaskModel(nn.Module):
    """VGG-based model for multi-task learning"""
    
    def __init__(self, num_companies: int, num_models: int, vgg_variant: str = 'vgg16', 
                 pretrained: bool = True, include_regression: bool = False):
        super().__init__()
        
        # Load pre-trained VGG
        if vgg_variant == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
        elif vgg_variant == 'vgg19':
            self.backbone = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown VGG variant: {vgg_variant}")
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[0].in_features
        
        # Replace the classifier with our custom heads
        self.backbone.classifier = nn.Identity()
        
        # Classification heads
        self.company_head = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_companies)
        )
        
        self.model_head = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_models)
        )
        
        # Optional regression heads
        self.include_regression = include_regression
        if include_regression:
            self.length_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            self.diameter_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
        
        logger.info(f"🏗️  Model Architecture:")
        logger.info(f"   Backbone: {vgg_variant} (pretrained={pretrained})")
        logger.info(f"   Features: {num_features}")
        logger.info(f"   Company classes: {num_companies}")
        logger.info(f"   Model classes: {num_models}")
        if include_regression:
            logger.info(f"   Regression: Length & Diameter enabled")
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        outputs = {
            'company': self.company_head(features),
            'model': self.model_head(features)
        }
        
        if self.include_regression:
            outputs['length'] = self.length_head(features).squeeze(-1)
            outputs['diameter'] = self.diameter_head(features).squeeze(-1)
        
        return outputs

def evaluate_model(model, dataloader, device, dataset):
    """Evaluate model on validation set"""
    model.eval()
    
    all_preds = {'company': [], 'model': []}
    all_labels = {'company': [], 'model': []}
    total_loss = 0.0
    
    # For regression metrics if enabled
    if model.include_regression:
        all_reg_preds = {'length': [], 'diameter': []}
        all_reg_labels = {'length': [], 'diameter': []}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            # Classification predictions
            for task in ['company', 'model']:
                preds = outputs[task].argmax(dim=1)
                targets = labels[task].to(device)
                
                # Filter valid labels (not -1)
                valid_mask = targets != -1
                if valid_mask.sum() > 0:
                    all_preds[task].extend(preds[valid_mask].cpu().numpy())
                    all_labels[task].extend(targets[valid_mask].cpu().numpy())
            
            # Regression predictions if enabled
            if model.include_regression:
                for task in ['length', 'diameter']:
                    targets = labels[task].to(device)
                    valid_mask = targets != -1
                    
                    if valid_mask.sum() > 0:
                        all_reg_preds[task].extend(outputs[task][valid_mask].cpu().numpy())
                        all_reg_labels[task].extend(targets[valid_mask].cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    
    # Classification metrics
    for task in ['company', 'model']:
        if len(all_labels[task]) > 0:
            acc = accuracy_score(all_labels[task], all_preds[task])
            metrics[f"{task}_accuracy"] = acc
            
            # Get label mapping for readable report
            if task == 'company':
                target_names = [dataset.idx_to_company[i] for i in sorted(dataset.company_to_idx.values())]
            else:
                target_names = [dataset.idx_to_model[i] for i in sorted(dataset.model_to_idx.values())]
            
            # Only include labels that appear in the data
            unique_labels = sorted(set(all_labels[task]))
            filtered_target_names = [target_names[i] for i in unique_labels if i < len(target_names)]
            
            report = classification_report(
                all_labels[task], all_preds[task], 
                labels=unique_labels,
                target_names=filtered_target_names,
                output_dict=True
            )
            metrics[f"{task}_report"] = report
    
    # Regression metrics if enabled
    if model.include_regression:
        for task in ['length', 'diameter']:
            if len(all_reg_labels[task]) > 0:
                mae = np.mean(np.abs(np.array(all_reg_preds[task]) - np.array(all_reg_labels[task])))
                metrics[f"{task}_mae"] = mae
    
    return metrics

def print_metrics(metrics, epoch):
    """Print evaluation metrics"""
    logger.info(f"\n📊 Evaluation Results - Epoch {epoch}")
    logger.info("=" * 60)
    
    # Classification metrics
    for task in ['company', 'model']:
        if f"{task}_accuracy" in metrics:
            logger.info(f"\n🎯 {task.upper()} Classification:")
            logger.info(f"  Overall Accuracy: {metrics[f'{task}_accuracy']:.4f}")
            
            if f"{task}_report" in metrics:
                report = metrics[f"{task}_report"]
                # Show top classes by F1-score
                class_scores = [(cls, data['f1-score']) for cls, data in report.items() 
                               if isinstance(data, dict) and 'f1-score' in data]
                class_scores.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"  Top 5 classes by F1-score:")
                for cls, f1 in class_scores[:5]:
                    logger.info(f"    {cls}: {f1:.3f}")
    
    # Regression metrics
    for task in ['length', 'diameter']:
        if f"{task}_mae" in metrics:
            logger.info(f"\n📏 {task.upper()} Regression:")
            logger.info(f"  MAE: {metrics[f'{task}_mae']:.3f}mm")

def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG for Implant Classification')
    
    # Data paths
    parser.add_argument('--train-json', type=str, required=True,
                        help='Path to training JSON file')
    parser.add_argument('--val-json', type=str, default=None,
                        help='Path to validation JSON file')
    parser.add_argument('--data-dir', type=str, 
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'),
                        help='Input data directory')
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
                        help='Output model directory')
    parser.add_argument('--image-root', type=str, default='train_augmented',
                        help='Root directory for images')
    
    # Model configuration
    parser.add_argument('--vgg-variant', type=str, default='vgg16',
                        choices=['vgg16', 'vgg19'],
                        help='VGG variant to use')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained weights')
    parser.add_argument('--include-regression', type=bool, default=False,
                        help='Include length/diameter regression heads')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Other options
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--save-best-only', type=bool, default=True,
                        help='Only save the best model')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("🦷 VGG Implant Classification Training")
    logger.info("=" * 60)
    logger.info("🎯 Training for:")
    logger.info("   • Company classification")
    logger.info("   • Model-type classification")
    if args.include_regression:
        logger.info("   • Length regression (mm)")
        logger.info("   • Diameter regression (mm)")
    logger.info("=" * 60)
    
    # Log configuration
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using {device} device")
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"🚀 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
    train_dataset = ImplantDataset(args.train_json, train_transform, args.image_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_loader = None
    if args.val_json and os.path.exists(args.val_json):
        val_dataset = ImplantDataset(args.val_json, val_transform, args.image_root)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)
        logger.info(f"✅ Validation data: {len(val_dataset)} samples")
    
    # Create model
    num_companies = len(train_dataset.company_to_idx)
    num_models = len(train_dataset.model_to_idx)
    
    model = VGGMultiTaskModel(
        num_companies=num_companies,
        num_models=num_models,
        vgg_variant=args.vgg_variant,
        pretrained=args.pretrained,
        include_regression=args.include_regression
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_mse = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    logger.info(f"\n🚀 Starting training for {args.epochs} epochs")
    logger.info(f"📊 Training samples: {len(train_dataset)}")
    
    best_val_acc = 0.0
    training_history = []
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save label mappings
    label_mappings = {
        'company_to_idx': train_dataset.company_to_idx,
        'model_to_idx': train_dataset.model_to_idx,
        'idx_to_company': train_dataset.idx_to_company,
        'idx_to_model': train_dataset.idx_to_model
    }
    
    with open(os.path.join(args.model_dir, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    for epoch in range(args.epochs):
        logger.info(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {optimizer.param_groups[0]["lr"]:.2e})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = {'company': 0, 'model': 0}
        running_total = {'company': 0, 'model': 0}
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = 0.0
            
            # Classification losses
            for task in ['company', 'model']:
                targets = labels[task].to(device)
                valid_mask = targets != -1
                if valid_mask.sum() > 0:
                    task_loss = criterion_ce(outputs[task], targets)
                    loss += task_loss
                    
                    # Track accuracy
                    preds = outputs[task].argmax(dim=1)
                    running_correct[task] += (preds[valid_mask] == targets[valid_mask]).sum().item()
                    running_total[task] += valid_mask.sum().item()
            
            # Regression losses if enabled
            if model.include_regression:
                for task in ['length', 'diameter']:
                    targets = labels[task].to(device)
                    valid_mask = targets != -1
                    if valid_mask.sum() > 0:
                        task_loss = criterion_mse(outputs[task][valid_mask], targets[valid_mask])
                        loss += task_loss * 0.1  # Weight regression loss lower
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            train_acc_company = running_correct['company'] / max(running_total['company'], 1)
            train_acc_model = running_correct['model'] / max(running_total['model'], 1)
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'comp_acc': train_acc_company,
                'model_acc': train_acc_model
            })
        
        # Training metrics
        train_metrics = {
            'loss': running_loss / len(train_loader),
            'company_acc': train_acc_company,
            'model_acc': train_acc_model,
            'avg_acc': (train_acc_company + train_acc_model) / 2
        }
        
        logger.info(f"🔼 Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Company Acc: {train_metrics['company_acc']:.4f}, "
                   f"Model Acc: {train_metrics['model_acc']:.4f}")
        
        # Validation phase
        if val_loader and (epoch + 1) % args.validate_every == 0:
            val_metrics = evaluate_model(model, val_loader, device, val_dataset)
            
            val_company_acc = val_metrics.get('company_accuracy', 0)
            val_model_acc = val_metrics.get('model_accuracy', 0)
            val_avg_acc = (val_company_acc + val_model_acc) / 2
            
            logger.info(f"🔽 Val - Company Acc: {val_company_acc:.4f}, "
                       f"Model Acc: {val_model_acc:.4f}, "
                       f"Avg Acc: {val_avg_acc:.4f}")
            
            # Print detailed metrics
            print_metrics(val_metrics, epoch + 1)
            
            # Update learning rate
            scheduler.step(val_avg_acc)
            
            # Save best model
            if val_avg_acc > best_val_acc:
                best_val_acc = val_avg_acc
                logger.info(f"💾 New best average accuracy: {best_val_acc:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': best_val_acc,
                    'company_acc': val_company_acc,
                    'model_acc': val_model_acc,
                    'args': vars(args),
                    'num_companies': num_companies,
                    'num_models': num_models
                }, os.path.join(args.model_dir, 'best_model.pth'))
            
            # Track history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_company_acc': train_metrics['company_acc'],
                'train_model_acc': train_metrics['model_acc'],
                'val_company_acc': val_company_acc,
                'val_model_acc': val_model_acc,
                'val_avg_acc': val_avg_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    # Save final model
    if not args.save_best_only:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.epochs,
            'args': vars(args),
            'num_companies': num_companies,
            'num_models': num_models
        }, os.path.join(args.model_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(args.model_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training info
    training_info = {
        'vgg_variant': args.vgg_variant,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_loader else 0,
        'num_companies': num_companies,
        'num_models': num_models,
        'best_val_acc': best_val_acc,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    with open(os.path.join(args.model_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"📁 Models saved to: {args.model_dir}")
    logger.info(f"📊 Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    # List files for debugging in SageMaker
    if os.path.exists("/opt/ml/"):
        logger.info("\n=== 🔍 SAGEMAKER INSTANCE STRUCTURE ===")
        dirs_to_check = ["/opt/ml/input", "/opt/ml/input/data"]
        for d in dirs_to_check:
            if os.path.exists(d):
                logger.info(f"\n📁 {d}")
                try:
                    items = os.listdir(d)[:10]  # Show first 10 items
                    for item in items:
                        logger.info(f"   {item}")
                except Exception as e:
                    logger.info(f"   Error listing: {e}")
    
    main()
