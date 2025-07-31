#!/usr/bin/env python3
"""
Loss functions for class imbalanced medical image segmentation
src/losses.py
transunet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks"""
    def __init__(self, smooth=1e-7, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        # Create mask for valid pixels (not ignore_index)
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        
        dice_scores = []
        for class_idx in range(num_classes):
            pred_class = pred[:, class_idx]
            target_class = (target == class_idx).float()
            
            # Apply valid mask
            pred_class = pred_class * valid_mask.float()
            target_class = target_class * valid_mask.float()
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return 1.0 - torch.stack(dice_scores).mean()

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Handle ignore_index in alpha weighting
                valid_mask = (target != self.ignore_index)
                alpha_t = torch.ones_like(target, dtype=torch.float, device=target.device)
                for class_idx in range(len(self.alpha)):
                    class_mask = (target == class_idx) & valid_mask
                    alpha_t[class_mask] = self.alpha[class_idx]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss - proven effective for imbalanced segmentation"""
    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.5, focal_weight=0.5, 
                 smooth=1e-7, ignore_index=-100):
        super(DiceFocalLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal

class TverskyLoss(nn.Module):
    """Tversky Loss - good for imbalanced datasets"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7, ignore_index=-100):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        # Create mask for valid pixels
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        
        tversky_scores = []
        for class_idx in range(num_classes):
            pred_class = pred[:, class_idx]
            target_class = (target == class_idx).float()
            
            # Apply valid mask
            pred_class = pred_class * valid_mask.float()
            target_class = target_class * valid_mask.float()
            
            TP = (pred_class * target_class).sum()
            FP = (pred_class * (1 - target_class)).sum()
            FN = ((1 - pred_class) * target_class).sum()
            
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky_scores.append(tversky)
        
        return 1.0 - torch.stack(tversky_scores).mean()

def get_loss_for_imbalance(num_classes, class_distribution=None):
    """
    Get recommended loss function based on class distribution
    
    Args:
        num_classes: Number of classes including background
        class_distribution: List of class frequencies
    """
    
    if class_distribution is not None:
        # Calculate imbalance ratio
        max_freq = max(class_distribution)
        min_freq = min([f for f in class_distribution if f > 0])
        imbalance_ratio = max_freq / min_freq
        
        print(f"Class imbalance ratio: {imbalance_ratio:.1f}")
        
        if imbalance_ratio > 10:
            print("High imbalance detected - using DiceFocalLoss with class weights")
            
            # Calculate inverse frequency weights
            total = sum(class_distribution)
            alpha_weights = []
            for freq in class_distribution:
                if freq > 0:
                    weight = total / (num_classes * freq)
                else:
                    weight = 1.0
                alpha_weights.append(weight)
            
            # Normalize weights
            alpha_weights = torch.tensor(alpha_weights, dtype=torch.float32)
            alpha_weights = alpha_weights / alpha_weights.mean()
            
            print(f"Class weights: {alpha_weights.tolist()}")
            
            return DiceFocalLoss(
                alpha=alpha_weights,
                gamma=2.0,
                dice_weight=0.6,
                focal_weight=0.4
            )
        else:
            print("Moderate imbalance detected - using TverskyLoss")
            return TverskyLoss(alpha=0.7, beta=0.3)
    else:
        print("No class distribution provided - using DiceFocalLoss (safe default)")
        return DiceFocalLoss(gamma=2.0)