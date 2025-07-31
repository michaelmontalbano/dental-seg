# Strategies to Improve Apex Detection Performance

## Current Performance
- Apex F1: 73.68% (vs 90%+ for other landmarks)
- Other landmarks: cej mesial (92.09%), cej distal (95.82%), ac mesial (92.60%), ac distal (94.55%)

## Recommended Improvements

### 1. **Weighted Loss Function**
Modify the MultiTaskLoss to give more weight to apex classification:

```python
# In MultiTaskLoss.__init__
self.class_weights = torch.ones(num_classes)
apex_idx = class_names.index('apex')
self.class_weights[apex_idx] = 2.0  # Double weight for apex

# In forward method
weighted_bce = nn.BCELoss(weight=self.class_weights[i])
```

### 2. **Focal Loss for Class Imbalance**
Replace BCE with Focal Loss for better handling of hard examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()
```

### 3. **Data Augmentation for Apex**
Add specific augmentations that help with apex visibility:

```python
# In dataset transform
if 'apex' in crop_annotations:
    # Apply contrast enhancement
    transforms.ColorJitter(brightness=0.3, contrast=0.4)
    # Apply slight rotation to see different angles
    transforms.RandomRotation(degrees=10)
```

### 4. **Multi-Scale Feature Extraction**
Add apex to SMALL_CLASSES for multi-scale processing:

```python
SMALL_CLASSES = {
    "margin", "calculus", "apex"  # Add apex here
}
```

### 5. **Auxiliary Apex Head**
Create a dedicated head for apex detection:

```python
class ViTMultiTask(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        # Add dedicated apex head
        self.apex_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary apex detection
        )
```

### 6. **Training Strategy Modifications**

```python
# In sagemaker_launcher.py, add these hyperparameters:
"apex_weight": 2.0,  # Increase apex loss weight
"apex_focal_loss": True,  # Use focal loss for apex
"apex_augmentation": True,  # Extra augmentation for apex samples
```

### 7. **Post-Processing Improvements**
- Use ensemble predictions from multiple crops
- Apply confidence threshold adjustment specifically for apex
- Use neighboring teeth context for apex validation

### 8. **Data Analysis**
Create a script to analyze apex annotations:

```python
# Check apex annotation distribution
apex_count = 0
total_crops = 0
for crop in dataset.crop_data:
    total_crops += 1
    for ann in crop['annotations']:
        if ann['category_name'] == 'apex':
            apex_count += 1
            
print(f"Apex presence: {apex_count}/{total_crops} = {apex_count/total_crops:.2%}")
```

## Implementation Priority
1. **Immediate**: Add apex to SMALL_CLASSES (easiest fix)
2. **Short-term**: Implement weighted loss and focal loss
3. **Medium-term**: Add dedicated apex head and augmentations
4. **Long-term**: Collect more apex annotations if data is imbalanced
