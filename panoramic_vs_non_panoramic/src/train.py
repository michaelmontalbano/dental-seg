#!/usr/bin/env python3
"""
Training script for panoramic vs non-panoramic dental X-ray classification using train.json and val.json
Classes: panoramic, non_panoramic (binary classification)
With overfitting detection and realistic expectations
"""

import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import logging
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
from tqdm import tqdm

# Import dataset module
from dataset import create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanoramicClassifier(nn.Module):
    """Binary panoramic vs non-panoramic dental X-ray classifier with proper regularization"""
    
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True, dropout_rate=0.5):
        super(PanoramicClassifier, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Add regularization layers
            self.backbone.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Keep original dropout
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
            
        elif model_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],
                self.backbone.classifier[1],
                self.backbone.classifier[2],
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)

def calculate_class_weights(datasets):
    """Calculate class weights for imbalanced data"""
    if 'train' not in datasets:
        return None
    
    train_dataset = datasets['train']
    
    # Count samples per class
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for class_name, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_idx = ['non_panoramic', 'panoramic'].index(class_name)
        class_weights[class_idx] = weight
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    # Convert to tensor
    weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float)
    return weights

def detect_overfitting(train_metrics, val_metrics, threshold=0.10):
    """Detect overfitting by comparing train vs validation metrics"""
    warnings = []
    
    # Check accuracy gap
    train_acc = train_metrics.get('accuracy', 0)
    val_acc = val_metrics.get('accuracy', 0)
    acc_gap = train_acc - val_acc
    
    if acc_gap > threshold:
        warnings.append(f"Accuracy gap: {acc_gap:.3f} (train - val)")
    
    # Check for suspiciously high accuracies (adjusted for binary classification)
    if val_acc > 0.85:
        warnings.append(f"Suspiciously high validation accuracy: {val_acc:.3f}")
    
    if train_acc > 0.90:
        warnings.append(f"Suspiciously high training accuracy: {train_acc:.3f}")
    
    # Early warning for rapid accuracy jumps
    if val_acc > 0.75 and train_acc > 0.85:
        warnings.append(f"Rapid convergence suggests task may be too easy")
    
    # Check per-class gaps
    train_classes = train_metrics.get('per_class_metrics', {})
    val_classes = val_metrics.get('per_class_metrics', {})
    
    for class_name in ['non_panoramic', 'panoramic']:
        if class_name in train_classes and class_name in val_classes:
            train_f1 = train_classes[class_name].get('f1-score', 0)
            val_f1 = val_classes[class_name].get('f1-score', 0)
            f1_gap = train_f1 - val_f1
            
            if f1_gap > threshold:
                warnings.append(f"{class_name} F1 gap: {f1_gap:.3f}")
    
    return warnings

def evaluate_epoch_metrics(model, dataloader, device, class_names, epoch, phase):
    """Evaluate detailed metrics with realistic bounds checking"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate per-class metrics
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics every 5 epochs or if suspicious
    should_log = (epoch % 5 == 0) or (accuracy > 0.80) or (phase == 'val' and accuracy > 0.75)
    
    if should_log:
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1} - {phase.upper()} METRICS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name}: P={metrics['precision']:.3f} "
                      f"R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} "
                      f"N={metrics['support']}")
        
        # Warnings for suspicious results (adjusted for binary classification)
        if accuracy > 0.85:
            print("⚠️  HIGH ACCURACY - Check for overfitting or data leakage")
        
        if len(all_labels) < 50:
            print(f"⚠️  SMALL {phase.upper()} SET ({len(all_labels)}) - Results may be unreliable")
        
        print(f"{'='*50}")
    
    return {
        'accuracy': accuracy,
        'per_class_metrics': {class_name: report.get(class_name, {}) for class_name in class_names},
        'macro_avg': report.get('macro avg', {}),
        'weighted_avg': report.get('weighted avg', {}),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }

def train_model_with_overfitting_detection(model, dataloaders, criterion, optimizer, scheduler, 
                                         num_epochs=25, device='cuda', patience=10):
    """Training with comprehensive overfitting detection"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    
    class_names = ['non_panoramic', 'panoramic']
    
    # Track training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_metrics': [], 'val_metrics': [],
        'overfitting_warnings': []
    }
    
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 40)
        
        epoch_train_metrics = None
        epoch_val_metrics = None
        
        # Training and validation phases
        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue
                
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Add progress bar
            pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()} Epoch {epoch+1}')
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{running_corrects.double() / ((pbar.n + 1) * inputs.size(0)):.4f}'
                })
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Update learning rate scheduler
            if phase == 'val' and scheduler:
                scheduler.step(epoch_loss)
            
            # Store basic metrics
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Detailed evaluation
            detailed_metrics = evaluate_epoch_metrics(
                model, dataloaders[phase], device, class_names, epoch, phase
            )
            history[f'{phase}_metrics'].append(detailed_metrics)
            
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log for monitoring
            print(f"[{epoch}] {phase}_loss: {epoch_loss:.4f}")
            print(f"[{epoch}] {phase}_accuracy: {epoch_acc:.4f}")
            
            # Store for overfitting detection
            if phase == 'train':
                epoch_train_metrics = detailed_metrics
            else:
                epoch_val_metrics = detailed_metrics
            
            # Save best model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            elif phase == 'val':
                epochs_without_improvement += 1
        
        # OVERFITTING DETECTION
        if epoch_train_metrics and epoch_val_metrics:
            warnings = detect_overfitting(epoch_train_metrics, epoch_val_metrics)
            if warnings:
                logger.warning(f"🚨 EPOCH {epoch + 1} OVERFITTING DETECTED:")
                for warning in warnings:
                    logger.warning(f"   • {warning}")
                history['overfitting_warnings'].append({
                    'epoch': epoch + 1,
                    'warnings': warnings
                })
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break
        
        print()
    
    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_val_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Final analysis
    total_overfitting_epochs = len(history['overfitting_warnings'])
    if total_overfitting_epochs > 0:
        logger.warning(f"🚨 TRAINING SUMMARY:")
        logger.warning(f"   Overfitting detected in {total_overfitting_epochs} epochs")
        
        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        
        logger.warning(f"   Final train accuracy: {final_train_acc:.3f}")
        logger.warning(f"   Final val accuracy: {final_val_acc:.3f}")
        logger.warning(f"   Accuracy gap: {final_train_acc - final_val_acc:.3f}")
        
        if best_val_acc > 0.85:
            logger.warning("⚠️  Validation accuracy > 85% - investigate task difficulty")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Panoramic vs non-panoramic dental X-ray classifier using train.json and val.json')
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3_large'])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--remove_corner_text', type=bool, default=True,
                        help='Remove corner text to prevent data leakage')
    
    # Data paths
    parser.add_argument('--train_json', type=str, default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/train.json',
                        help='Path to train.json file')
    parser.add_argument('--val_json', type=str, default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/val.json',
                        help='Path to val.json file')
    parser.add_argument('--image_dir', type=str, default='s3://codentist-general/datasets/panoramic_vs_non_panoramic/images',
                        help='Directory containing image files')
    parser.add_argument('--model_dir', type=str, default='../model',
                        help='Directory to save model')
    
    # SageMaker paths (environment variables)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', None),
                        help='SageMaker train channel (for train.json)')
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', None),
                        help='SageMaker val channel (for val.json)')
    parser.add_argument('--images', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', None),
                        help='SageMaker images channel')

    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'PyTorch version: {torch.__version__}')
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Determine paths (SageMaker vs local)
    if args.train and args.val and args.images:
        # SageMaker mode - use input channels
        train_json_path = os.path.join(args.train, 'train.json')
        val_json_path = os.path.join(args.val, 'val.json')
        image_dir_path = args.images
        logger.info("🚀 Running in SageMaker mode")
        logger.info(f"Train JSON: {train_json_path}")
        logger.info(f"Val JSON: {val_json_path}")
        logger.info(f"Images: {image_dir_path}")
    else:
        # Local mode - use provided paths
        train_json_path = args.train_json
        val_json_path = args.val_json
        image_dir_path = args.image_dir
        logger.info("🏠 Running in local mode")
        logger.info(f"Train JSON: {train_json_path}")
        logger.info(f"Val JSON: {val_json_path}")
        logger.info(f"Images: {image_dir_path}")
    
    # Update model_dir for SageMaker
    if 'SM_MODEL_DIR' in os.environ:
        args.model_dir = os.environ['SM_MODEL_DIR']
        logger.info(f"Using SageMaker model directory: {args.model_dir}")
    
    # Create data loaders
    logger.info("🎯 Creating dataloaders from JSON files...")
    datasets, dataloaders = create_dataloaders(
        train_json=train_json_path,
        val_json=val_json_path,
        image_dir=image_dir_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        center_crop_corners=args.remove_corner_text
    )
    
    if 'train' not in dataloaders:
        raise ValueError("Training data not found!")
    
    logger.info("✅ Dataloaders created successfully")
    
    # Create model with regularization
    model = PanoramicClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)
    
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                          lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    
    # Use ReduceLROnPlateau for better convergence
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Train model with overfitting detection
    logger.info("🚀 Starting training with overfitting monitoring...")
    model, history = train_model_with_overfitting_detection(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience
    )
    
    # Final evaluation
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save model and metadata
    model_artifacts = {
        'model_state_dict': model.state_dict(),
        'model_name': args.model_name,
        'class_names': ['non_panoramic', 'panoramic'],
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'final_accuracy': final_val_acc,
        'training_history': history,
        'overfitting_detected': len(history['overfitting_warnings']) > 0,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout_rate,
            'epochs': args.epochs,
            'model_name': args.model_name,
            'image_size': args.image_size,
            'remove_corner_text': args.remove_corner_text
        }
    }
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model_artifacts, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training metrics for analysis
    metrics_path = os.path.join(args.model_dir, 'training_metrics.json')
    
    # Convert numpy types for JSON serialization
    json_history = {}
    for key, value in history.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (np.integer, np.floating)):
                json_history[key] = [float(v) for v in value]
            else:
                json_history[key] = value
        else:
            json_history[key] = value
    
    metrics = {
        'final_val_accuracy': final_val_acc,
        'final_train_accuracy': final_train_acc,
        'accuracy_gap': final_train_acc - final_val_acc,
        'training_history': json_history,
        'overfitting_summary': {
            'epochs_with_overfitting': len(history['overfitting_warnings']),
            'overfitting_detected': len(history['overfitting_warnings']) > 0
        },
        'hyperparameters': model_artifacts['hyperparameters']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")
    
    # Log final results
    print(f"final_accuracy: {final_val_acc:.4f}")
    print(f"final_train_accuracy: {final_train_acc:.4f}")
    print(f"accuracy_gap: {final_train_acc - final_val_acc:.4f}")
    print(f"overfitting_detected: {len(history['overfitting_warnings']) > 0}")
    
    # Final summary
    logger.info("🎯 TRAINING COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
    logger.info(f"Accuracy gap: {final_train_acc - final_val_acc:.4f}")
    
    # Realistic expectations feedback (adjusted for binary classification)
    if 0.60 <= final_val_acc <= 0.80:
        logger.info("✅ Validation accuracy in conservative range (60-80%)")
        logger.info("   This is expected with aggressive regularization")
    elif final_val_acc > 0.85:
        logger.warning("🚨 Validation accuracy > 85% - investigate task difficulty")
        logger.warning("   This may indicate task is easier than expected or data issues")
    elif final_val_acc < 0.55:
        logger.warning("⚠️  Validation accuracy < 55% is below reasonable baseline for binary classification")
        logger.warning("   Consider collecting more data or adjusting hyperparameters")
    
    if len(history['overfitting_warnings']) > 3:
        logger.warning("🚨 Frequent overfitting detected!")
        logger.warning("   Consider more regularization or data augmentation")
    
    logger.info("Model training completed successfully!")

if __name__ == '__main__':
    main()
