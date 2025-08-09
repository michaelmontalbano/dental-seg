"""
Utility functions for VGG training
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

def save_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         labels: List[str], output_path: str, 
                         title: str = "Confusion Matrix"):
    """Save confusion matrix as an image"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(min(20, len(labels)), min(20, len(labels))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")

def load_model_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load a model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint

def plot_training_history(history_path: str, output_dir: str):
    """Plot training history from JSON file"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        logger.warning("No training history found")
        return
    
    epochs = [h['epoch'] for h in history]
    
    # Plot accuracies
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Company accuracy
    axes[0, 0].plot(epochs, [h['train_company_acc'] for h in history], 
                    label='Train', marker='o')
    if 'val_company_acc' in history[0]:
        axes[0, 0].plot(epochs, [h['val_company_acc'] for h in history], 
                        label='Val', marker='s')
    axes[0, 0].set_title('Company Classification Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Model accuracy
    axes[0, 1].plot(epochs, [h['train_model_acc'] for h in history], 
                    label='Train', marker='o')
    if 'val_model_acc' in history[0]:
        axes[0, 1].plot(epochs, [h['val_model_acc'] for h in history], 
                        label='Val', marker='s')
    axes[0, 1].set_title('Model Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(epochs, [h['train_loss'] for h in history], 
                    label='Train Loss', marker='o')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(epochs, [h['learning_rate'] for h in history], 
                    marker='o', color='green')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {output_path}")

def get_dataset_stats(dataset) -> Dict:
    """Get statistics about the dataset"""
    stats = {
        'total_samples': len(dataset),
        'num_companies': len(dataset.company_to_idx),
        'num_models': len(dataset.model_to_idx),
        'companies': list(dataset.company_to_idx.keys()),
        'models': list(dataset.model_to_idx.keys())
    }
    
    # Count samples per company
    company_counts = {}
    model_counts = {}
    
    for entry in dataset.entries:
        company = entry.get('company', 'Unknown')
        model = entry.get('model', 'Unknown')
        
        company_counts[company] = company_counts.get(company, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1
    
    stats['company_distribution'] = company_counts
    stats['model_distribution'] = model_counts
    
    return stats

def save_predictions(predictions: Dict[str, List], 
                    labels: Dict[str, List],
                    output_path: str):
    """Save predictions and labels to a JSON file"""
    results = {
        'predictions': predictions,
        'labels': labels
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved predictions to {output_path}")

def calculate_class_weights(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate class weights for imbalanced datasets"""
    company_counts = torch.zeros(len(dataset.company_to_idx))
    model_counts = torch.zeros(len(dataset.model_to_idx))
    
    for entry in dataset.entries:
        company_idx = dataset.company_to_idx.get(entry.get('company', 'Unknown'), -1)
        model_idx = dataset.model_to_idx.get(entry.get('model', 'Unknown'), -1)
        
        if company_idx != -1:
            company_counts[company_idx] += 1
        if model_idx != -1:
            model_counts[model_idx] += 1
    
    # Calculate weights (inverse frequency)
    company_weights = 1.0 / (company_counts + 1e-6)
    model_weights = 1.0 / (model_counts + 1e-6)
    
    # Normalize
    company_weights = company_weights / company_weights.sum() * len(company_weights)
    model_weights = model_weights / model_weights.sum() * len(model_weights)
    
    return company_weights, model_weights

def export_model_for_inference(model_path: str, output_path: str, 
                              label_mappings_path: str):
    """Export model for inference with all necessary files"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    import shutil
    shutil.copy2(model_path, output_dir / 'model.pth')
    
    # Copy label mappings
    shutil.copy2(label_mappings_path, output_dir / 'label_mappings.json')
    
    # Create inference config
    inference_config = {
        'model_type': 'vgg_multitask',
        'input_size': [224, 224],
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'tasks': ['company', 'model']
    }
    
    with open(output_dir / 'inference_config.json', 'w') as f:
        json.dump(inference_config, f, indent=2)
    
    logger.info(f"Exported model for inference to {output_dir}")
