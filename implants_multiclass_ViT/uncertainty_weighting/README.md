# Uncertainty-Based Task Weighting for Multi-Task Learning

This directory contains an implementation of unified multi-task learning with uncertainty-based task weighting, following the approach from "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., 2018).

## Key Features

- **Automatic Task Balancing**: Learns optimal weights for each task during training
- **Uncertainty Modeling**: Each task has a learnable uncertainty parameter
- **Gradient Normalization**: Prevents one task from dominating others
- **Dynamic Adjustment**: Task weights adapt based on relative performance

## Mathematical Foundation

The loss for each task is weighted by its uncertainty:
```
L_total = Σ_i (1/2σ_i²) * L_i + log(σ_i)
```

Where:
- `L_i` is the loss for task i
- `σ_i` is the learnable uncertainty parameter for task i
- The `log(σ_i)` term prevents σ from growing unbounded

## Architecture

```
Image -> ViT Backbone -> Shared Features
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
      Company Head      Model Head       Diameter Head     Length Head
            |                 |                 |                |
            v                 v                 v                v
     Company Loss      Model Loss      Diameter Loss     Length Loss
            |                 |                 |                |
            v                 v                 v                v
          σ_c²              σ_m²              σ_d²             σ_l²
            |                 |                 |                |
            +--------+--------+--------+--------+
                     |
                     v
            Weighted Total Loss
```

## Benefits

- **No Manual Tuning**: Automatically finds optimal task weights
- **Principled Approach**: Based on homoscedastic uncertainty
- **Adaptive**: Adjusts to task difficulty throughout training
- **Interpretable**: Uncertainty values indicate task difficulty

## Implementation Details

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super().__init__()
        # Initialize log variances (log(σ²))
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        # losses: dict with keys ['company', 'model', 'diameter', 'length']
        weighted_losses = []
        for i, (task, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)
```

## Gradient Normalization

To prevent tasks with larger gradients from dominating:
```python
# Normalize gradients by their L2 norm
for task_head in [company_head, model_head, diameter_head, length_head]:
    grad_norm = torch.nn.utils.clip_grad_norm_(task_head.parameters(), max_norm=1.0)
```

## Usage

```bash
python train_unified_uncertainty.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --use_uncertainty_weighting true \
    --gradient_normalization true \
    --initial_log_vars "0,0,0,0"
```

## Monitoring

During training, track:
- Individual task losses
- Learned uncertainty values (σ²)
- Effective task weights (1/2σ²)
- Task-specific learning progress

## Expected Behavior

- Tasks that are harder to learn will have higher σ² (lower weight)
- Tasks that converge quickly will have lower σ² (higher weight)
- The model automatically balances effort across all tasks
- Final σ² values indicate relative task difficulty
