import argparse
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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

def report_metric(metric_name, value, epoch=None):
    """Report metric to SageMaker and CloudWatch"""
    if epoch is not None:
        # Format for SageMaker hyperparameter tuning
        print(f"[{epoch}] {metric_name} = {value:.6f};")
        logger.info(f"Epoch {epoch}: {metric_name} = {value:.6f}")
    else:
        print(f"{metric_name} = {value:.6f};")
        logger.info(f"{metric_name} = {value:.6f}")

class ImplantRegressionDataset(Dataset):
    def __init__(self, json_path, company_vocab, model_vocab, task='both', transform=None, image_root=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.company_vocab = company_vocab
        self.model_vocab = model_vocab
        self.task = task  # 'length', 'diameter', or 'both'
        self.transform = transform
        self.image_root = image_root
        
        # Filter entries based on task with robust validation
        valid_entries = []
        skipped_count = 0
        for i, e in enumerate(self.entries):
            if e.get("company") and e.get("model"):
                # Validate length
                length_valid = False
                if e.get("length") is not None:
                    try:
                        # Try to convert to float - if it fails, it's not valid
                        length_val = float(e["length"])
                        if not np.isnan(length_val) and not np.isinf(length_val):
                            length_valid = True
                    except (ValueError, TypeError):
                        pass
                
                # Validate diameter
                diameter_valid = False
                if e.get("diameter") is not None:
                    try:
                        # Try to convert to float - if it fails, it's not valid
                        diameter_val = float(e["diameter"])
                        if not np.isnan(diameter_val) and not np.isinf(diameter_val):
                            diameter_valid = True
                    except (ValueError, TypeError):
                        pass
                
                # Add to valid entries based on task
                if task == 'length' and length_valid:
                    valid_entries.append(e)
                elif task == 'diameter' and diameter_valid:
                    valid_entries.append(e)
                elif task == 'both' and length_valid and diameter_valid:
                    valid_entries.append(e)
                else:
                    skipped_count += 1
                    if skipped_count <= 5:  # Log first few skipped entries
                        print(f"⚠️  Skipped entry {i}: length={e.get('length', 'None')}, diameter={e.get('diameter', 'None')}")
        
        self.entries = valid_entries
        print(f"📊 Loaded {len(self.entries)} entries for {task} task from {json_path}")
        
        # Compute statistics
        if task in ['length', 'both']:
            lengths = []
            for e in self.entries:
                try:
                    if e.get("length") is not None and str(e.get("length", "")).strip() != "":
                        lengths.append(float(e["length"]))
                except (ValueError, TypeError):
                    pass  # Skip invalid values
            
            if lengths:
                self.length_mean = np.mean(lengths)
                self.length_std = np.std(lengths)
                print(f"📏 Length stats: mean={self.length_mean:.2f}mm, std={self.length_std:.2f}mm")
            else:
                self.length_mean = 0.0
                self.length_std = 1.0
                print(f"⚠️  No valid length values found, using defaults")
        
        if task in ['diameter', 'both']:
            diameters = []
            for e in self.entries:
                try:
                    if e.get("diameter") is not None and str(e.get("diameter", "")).strip() != "":
                        diameters.append(float(e["diameter"]))
                except (ValueError, TypeError):
                    pass  # Skip invalid values
            
            if diameters:
                self.diameter_mean = np.mean(diameters)
                self.diameter_std = np.std(diameters)
                print(f"📏 Diameter stats: mean={self.diameter_mean:.2f}mm, std={self.diameter_std:.2f}mm")
            else:
                self.diameter_mean = 0.0
                self.diameter_std = 1.0
                print(f"⚠️  No valid diameter values found, using defaults")

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
            print(f"❌ Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)

        company_label = self.company_vocab.get(entry["company"], -1)
        model_label = self.model_vocab.get(entry["model"], -1)
        
        # Get targets based on task with robust error handling
        try:
            if self.task == 'length':
                target = torch.tensor([float(entry["length"])], dtype=torch.float32)
            elif self.task == 'diameter':
                target = torch.tensor([float(entry["diameter"])], dtype=torch.float32)
            else:  # both
                target = torch.tensor([float(entry["length"]), float(entry["diameter"])], dtype=torch.float32)
        except (ValueError, TypeError) as e:
            # This should not happen if filtering is working correctly
            print(f"⚠️  Error converting values for entry {idx}: {e}")
            print(f"    Entry: {entry}")
            # Return zeros as fallback
            if self.task == 'both':
                target = torch.tensor([0.0, 0.0], dtype=torch.float32)
            else:
                target = torch.tensor([0.0], dtype=torch.float32)
            
        return image, company_label, model_label, target

class EmbeddingModel(nn.Module):
    """Base embedding model used for company/model embeddings - can be frozen/unfrozen"""
    def __init__(self, base_model_name, num_classes, embed_dim=256, model_type='company'):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.embed_dim = embed_dim
        self.model_type = model_type
        
        # Classification branch (will be loaded from checkpoint)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Start with frozen parameters
        self.freeze()
        
        print(f"🏗️  {model_type.capitalize()} Embedding Model:")
        print(f"   Backbone: {base_model_name}")
        print(f"   Features: {self.feat_dim}")
        print(f"   Embedding dim: {embed_dim}")
        print(f"   Initially: FROZEN")

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return embeddings, features
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self, ratio=1.0):
        """Gradually unfreeze parameters
        ratio: fraction of parameters to unfreeze (from top layers down)
        """
        all_params = list(self.parameters())
        n_params = len(all_params)
        n_unfreeze = int(n_params * ratio)
        
        # Unfreeze the last n_unfreeze parameters (top layers)
        for i, param in enumerate(all_params):
            if i >= n_params - n_unfreeze:
                param.requires_grad = True
        
        # Count unfrozen parameters
        unfrozen = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"📌 Unfroze {unfrozen}/{n_params} {self.model_type} model parameters")

class DualEmbeddingRegression(nn.Module):
    def __init__(self, base_model_name, num_companies, num_models, 
                 company_embed_dim=256, model_embed_dim=256, task='both', 
                 use_attention=True, attention_type='dual'):
        super().__init__()
        
        # Frozen embedding models
        self.company_model = EmbeddingModel(base_model_name, num_companies, company_embed_dim, 'company')
        self.model_model = EmbeddingModel(base_model_name, num_models, model_embed_dim, 'model')
        
        self.use_attention = use_attention
        self.attention_type = attention_type  # 'dual', 'triple', or 'hierarchical'
        self.task = task
        feat_dim = self.company_model.feat_dim
        
        if use_attention:
            if attention_type == 'dual':
                # Simple attention for company vs model embeddings
                self.attention = nn.Sequential(
                    nn.Linear(company_embed_dim + model_embed_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),  # 2 weights: company, model
                    nn.Softmax(dim=1)
                )
                attention_output_dim = company_embed_dim + model_embed_dim
            elif attention_type == 'triple':
                # Attention for features, company, and model embeddings
                self.attention = nn.Sequential(
                    nn.Linear(feat_dim + company_embed_dim + model_embed_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),  # 3 weights: features, company, model
                    nn.Softmax(dim=1)
                )
                attention_output_dim = feat_dim + company_embed_dim + model_embed_dim
            else:  # hierarchical
                # Two-level attention: first combine embeddings, then with features
                self.embed_attention = nn.Sequential(
                    nn.Linear(company_embed_dim + model_embed_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                    nn.Softmax(dim=1)
                )
                self.feature_attention = nn.Sequential(
                    nn.Linear(feat_dim + company_embed_dim + model_embed_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2),  # features vs combined embeddings
                    nn.Softmax(dim=1)
                )
                attention_output_dim = feat_dim + company_embed_dim + model_embed_dim
        else:
            attention_output_dim = feat_dim + company_embed_dim + model_embed_dim
        
        # Regression head
        # Determine output dimension
        if task == 'length':
            output_dim = 1
        elif task == 'diameter':
            output_dim = 1
        else:  # both
            output_dim = 2
        
        self.regression_head = nn.Sequential(
            nn.Linear(attention_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        print(f"🏗️  Dual Embedding Regression Model:")
        print(f"   Task: {task}")
        print(f"   Company embeddings: {company_embed_dim}")
        print(f"   Model embeddings: {model_embed_dim}")
        print(f"   Output dimension: {output_dim}")
        print(f"   Use attention: {use_attention}")
        print(f"   Attention type: {attention_type}")

    def forward(self, x):
        # Get embeddings and features from both models
        # Check if models are frozen
        company_frozen = not any(p.requires_grad for p in self.company_model.parameters())
        model_frozen = not any(p.requires_grad for p in self.model_model.parameters())
        
        if company_frozen:
            with torch.no_grad():
                company_embed, company_features = self.company_model(x)
        else:
            company_embed, company_features = self.company_model(x)
            
        if model_frozen:
            with torch.no_grad():
                model_embed, model_features = self.model_model(x)
        else:
            model_embed, model_features = self.model_model(x)
        
        # Use features from company model (they should be similar)
        features = company_features
        
        if self.use_attention:
            if self.attention_type == 'dual':
                # Attention on embeddings only
                embed_combined = torch.cat([company_embed, model_embed], dim=1)
                attention_weights = self.attention(embed_combined)
                
                weighted_company = company_embed * attention_weights[:, 0:1]
                weighted_model = model_embed * attention_weights[:, 1:2]
                combined = torch.cat([weighted_company, weighted_model], dim=1)
                
            elif self.attention_type == 'triple':
                # Attention on all three: features, company, model
                all_combined = torch.cat([features, company_embed, model_embed], dim=1)
                attention_weights = self.attention(all_combined)
                
                weighted_features = features * attention_weights[:, 0:1]
                weighted_company = company_embed * attention_weights[:, 1:2]
                weighted_model = model_embed * attention_weights[:, 2:3]
                combined = torch.cat([weighted_features, weighted_company, weighted_model], dim=1)
                
            else:  # hierarchical
                # First level: attention between embeddings
                embed_combined = torch.cat([company_embed, model_embed], dim=1)
                embed_attention_weights = self.embed_attention(embed_combined)
                
                weighted_company = company_embed * embed_attention_weights[:, 0:1]
                weighted_model = model_embed * embed_attention_weights[:, 1:2]
                combined_embeddings = torch.cat([weighted_company, weighted_model], dim=1)
                
                # Second level: attention between features and combined embeddings
                all_combined = torch.cat([features, combined_embeddings], dim=1)
                feature_attention_weights = self.feature_attention(all_combined)
                
                # Split the weight for combined embeddings proportionally
                embed_weight = feature_attention_weights[:, 1:2]
                weighted_features = features * feature_attention_weights[:, 0:1]
                weighted_embeddings = combined_embeddings * embed_weight
                
                combined = torch.cat([weighted_features, weighted_embeddings], dim=1)
        else:
            # Simple concatenation
            combined = torch.cat([features, company_embed, model_embed], dim=1)
        
        # Regression prediction
        predictions = self.regression_head(combined)
        
        return predictions

def evaluate_regression(model, dataloader, criterion, device, task, stats=None):
    """Evaluate regression model"""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, _, _, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    results = {'loss': avg_loss}
    
    if task == 'both':
        # Length metrics
        length_mae = mean_absolute_error(all_targets[:, 0], all_predictions[:, 0])
        length_rmse = np.sqrt(mean_squared_error(all_targets[:, 0], all_predictions[:, 0]))
        length_r2 = r2_score(all_targets[:, 0], all_predictions[:, 0])
        
        # Diameter metrics
        diameter_mae = mean_absolute_error(all_targets[:, 1], all_predictions[:, 1])
        diameter_rmse = np.sqrt(mean_squared_error(all_targets[:, 1], all_predictions[:, 1]))
        diameter_r2 = r2_score(all_targets[:, 1], all_predictions[:, 1])
        
        results.update({
            'length_mae': length_mae,
            'length_rmse': length_rmse,
            'length_r2': length_r2,
            'diameter_mae': diameter_mae,
            'diameter_rmse': diameter_rmse,
            'diameter_r2': diameter_r2
        })
    else:
        # Single task metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        
        results.update({
            f'{task}_mae': mae,
            f'{task}_rmse': rmse,
            f'{task}_r2': r2
        })
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Train Regression Model with Dual Embeddings')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base', 
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--task', type=str, default='both',
                        choices=['length', 'diameter', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='outputs_dual_regression')
    parser.add_argument('--image_root', type=str, default='aug_dataset')
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--save_best_only', type=str, default='true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--company_embed_dim', type=int, default=256)
    parser.add_argument('--model_embed_dim', type=int, default=256)
    parser.add_argument('--use_attention', type=str, default='true')
    parser.add_argument('--attention_type', type=str, default='triple',
                        choices=['dual', 'triple', 'hierarchical'])
    parser.add_argument('--company_checkpoint', type=str, required=True,
                        help='Path to pretrained company model checkpoint')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to pretrained model-type model checkpoint')
    parser.add_argument('--unfreeze_epoch', type=int, default=20,
                        help='Epoch to start unfreezing embedding models')
    parser.add_argument('--full_unfreeze_epoch', type=int, default=40,
                        help='Epoch to fully unfreeze embedding models')
    parser.add_argument('--company_lr_multiplier', type=float, default=0.1,
                        help='Learning rate multiplier for company model')
    parser.add_argument('--model_lr_multiplier', type=float, default=0.1,
                        help='Learning rate multiplier for model-type model')
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_best_only = args.save_best_only.lower() in ('true', '1', 'yes')
    args.use_attention = args.use_attention.lower() in ('true', '1', 'yes')

    print(f"🎯 Regression with Dual Embeddings - {args.task.upper()}")
    print("=" * 60)
    print("=== TRAINING CONFIGURATION ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using {device} device")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # Load data to build vocabularies
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
    
    companies = sorted(set(e.get("company") for e in all_entries if e.get("company")))
    company_vocab = {c: i for i, c in enumerate(companies)}
    print(f"📋 Company vocabulary: {len(company_vocab)} unique companies")
    
    models = sorted(set(e.get("model") for e in all_entries if e.get("model")))
    model_vocab = {m: i for i, m in enumerate(models)}
    print(f"📋 Model vocabulary: {len(model_vocab)} unique models")
    
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
    train_dataset = ImplantRegressionDataset(
        args.train_json, company_vocab, model_vocab, args.task,
        train_transform, image_root=args.image_root
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = None
    if val_entries:
        val_dataset = ImplantRegressionDataset(
            args.val_json, company_vocab, model_vocab, args.task,
            val_transform, image_root=args.image_root
        )
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model_map = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224', 
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }

    model = DualEmbeddingRegression(
        model_map[args.model_size], 
        len(company_vocab),
        len(model_vocab),
        company_embed_dim=args.company_embed_dim,
        model_embed_dim=args.model_embed_dim,
        task=args.task,
        use_attention=args.use_attention,
        attention_type=args.attention_type
    ).to(device)
    
    # Load pretrained company model
    print(f"📥 Loading pretrained company model from {args.company_checkpoint}")
    company_checkpoint = torch.load(args.company_checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(company_checkpoint, dict) and 'model_state_dict' in company_checkpoint:
        company_state_dict = company_checkpoint['model_state_dict']
    else:
        company_state_dict = company_checkpoint
    
    # Map the weights to our model structure
    mapped_company_state_dict = {}
    for key, value in company_state_dict.items():
        if key.startswith('backbone.'):
            mapped_company_state_dict[key] = value
        elif key.startswith('classifier.'):
            mapped_company_state_dict[f'head.{key}'] = value
        elif key.startswith(('cls_token', 'pos_embed', 'patch_embed', 'blocks', 'norm')):
            mapped_company_state_dict[f'backbone.{key}'] = value
    
    # Load the mapped weights
    missing_keys, unexpected_keys = model.company_model.load_state_dict(mapped_company_state_dict, strict=False)
    print(f"✅ Loaded pretrained company weights")
    if missing_keys:
        print(f"ℹ️  Missing keys: {len(missing_keys)} (expected for custom classifier layers)")
    
    # Load pretrained model-type model
    print(f"📥 Loading pretrained model-type model from {args.model_checkpoint}")
    model_checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(model_checkpoint, dict) and 'model_state_dict' in model_checkpoint:
        model_state_dict = model_checkpoint['model_state_dict']
    else:
        model_state_dict = model_checkpoint
    
    # Map the weights to our model structure
    mapped_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('backbone.'):
            mapped_model_state_dict[key] = value
        elif key.startswith('classifier.'):
            mapped_model_state_dict[f'head.{key}'] = value
        elif key.startswith(('cls_token', 'pos_embed', 'patch_embed', 'blocks', 'norm')):
            mapped_model_state_dict[f'backbone.{key}'] = value
    
    # Load the mapped weights
    missing_keys, unexpected_keys = model.model_model.load_state_dict(mapped_model_state_dict, strict=False)
    print(f"✅ Loaded pretrained model-type weights")
    if missing_keys:
        print(f"ℹ️  Missing keys: {len(missing_keys)} (expected for custom classifier layers)")
    
    # Only train the regression head (embedding models are frozen)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n🚀 Starting training for {args.epochs} epochs")
    
    best_val_loss = float('inf')
    training_history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset statistics for denormalization
    stats = {
        'length_mean': train_dataset.length_mean if hasattr(train_dataset, 'length_mean') else 0,
        'length_std': train_dataset.length_std if hasattr(train_dataset, 'length_std') else 1,
        'diameter_mean': train_dataset.diameter_mean if hasattr(train_dataset, 'diameter_mean') else 0,
        'diameter_std': train_dataset.diameter_std if hasattr(train_dataset, 'diameter_std') else 1
    }
    
    for epoch in range(args.epochs):
        print(f'\n📍 Epoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})')
        
        # Handle unfreezing
        if epoch + 1 == args.unfreeze_epoch:
            # Start gradual unfreezing (50% of parameters)
            model.company_model.unfreeze(ratio=0.5)
            model.model_model.unfreeze(ratio=0.5)
            # Recreate optimizer with different learning rates
            param_groups = [
                {'params': model.regression_head.parameters(), 'lr': args.learning_rate},
                {'params': model.company_model.parameters(), 'lr': args.learning_rate * args.company_lr_multiplier},
                {'params': model.model_model.parameters(), 'lr': args.learning_rate * args.model_lr_multiplier}
            ]
            if args.use_attention:
                if args.attention_type == 'hierarchical':
                    param_groups.extend([
                        {'params': model.embed_attention.parameters(), 'lr': args.learning_rate},
                        {'params': model.feature_attention.parameters(), 'lr': args.learning_rate}
                    ])
                else:
                    param_groups.append({'params': model.attention.parameters(), 'lr': args.learning_rate})
            
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            # Recreate scheduler
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # Fast-forward scheduler to current epoch
            for _ in range(epoch):
                scheduler.step()
            # Update trainable parameters count
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
            
        elif epoch + 1 == args.full_unfreeze_epoch:
            # Fully unfreeze embedding models
            model.company_model.unfreeze(ratio=1.0)
            model.model_model.unfreeze(ratio=1.0)
            # Update trainable parameters count
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc="Training")
        for images, _, _, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        scheduler.step()
        
        avg_train_loss = running_loss / len(train_dataloader)
        print(f"🔼 Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_dataloader and (epoch + 1) % args.validate_every == 0:
            val_results = evaluate_regression(
                model, val_dataloader, criterion, device, args.task, stats
            )
            
            print(f"🔽 Val Loss: {val_results['loss']:.4f}")
            
            # Report metrics for SageMaker
            report_metric('val_loss', val_results['loss'], epoch + 1)
            
            if args.task == 'both':
                print(f"   Length - MAE: {val_results['length_mae']:.2f}mm, "
                      f"RMSE: {val_results['length_rmse']:.2f}mm, R²: {val_results['length_r2']:.3f}")
                print(f"   Diameter - MAE: {val_results['diameter_mae']:.2f}mm, "
                      f"RMSE: {val_results['diameter_rmse']:.2f}mm, R²: {val_results['diameter_r2']:.3f}")
                
                # Report individual metrics
                report_metric('length_mae', val_results['length_mae'], epoch + 1)
                report_metric('diameter_mae', val_results['diameter_mae'], epoch + 1)
                report_metric('avg_mae', (val_results['length_mae'] + val_results['diameter_mae']) / 2, epoch + 1)
            else:
                print(f"   {args.task.capitalize()} - MAE: {val_results[f'{args.task}_mae']:.2f}mm, "
                      f"RMSE: {val_results[f'{args.task}_rmse']:.2f}mm, R²: {val_results[f'{args.task}_r2']:.3f}")
                
                # Report single task metric
                report_metric(f'{args.task}_mae', val_results[f'{args.task}_mae'], epoch + 1)
            
            # Save best model
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                print(f"💾 New best validation loss: {best_val_loss:.4f}! Saving model...")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'company_vocab': company_vocab,
                    'model_vocab': model_vocab,
                    'args': vars(args),
                    'val_results': val_results,
                    'stats': stats
                }, os.path.join(args.output_dir, "best_model.pth"))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
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
            'args': vars(args),
            'stats': stats
        }, os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history with proper serialization
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(convert_to_serializable(training_history), f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
