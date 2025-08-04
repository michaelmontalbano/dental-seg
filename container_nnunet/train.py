#!/usr/bin/env python3
import argparse
import os
import torch
from MedMamba import MedMamba  # make sure this file is in the same directory
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import cv2
import numpy as np
from typing import Optional, Tuple, Union, List, Dict
import json
from pathlib import Path
from collections import defaultdict, Counter



class DentalAugmentations:
    """Dental-specific augmentation techniques for X-ray analysis"""
    
    @staticmethod
    def create_dental_transform(
        image_size: int = 224,
        is_training: bool = True,
        intensity: str = 'medium'  # 'light', 'medium', 'strong'
    ) -> A.Compose:
        """
        Create comprehensive dental X-ray augmentation pipeline
        
        Args:
            image_size: Target image size
            is_training: Whether this is for training (apply augmentations) or validation
            intensity: Augmentation intensity level
        """
        
        # Intensity-based parameters
        intensity_params = {
            'light': {
                'rotation_limit': 10,
                'brightness_limit': 0.1,
                'contrast_limit': 0.15,
                'gamma_limit': (85, 115),
                'noise_var': (5, 15),
                'blur_limit': 3,
                'quality_lower': 85
            },
            'medium': {
                'rotation_limit': 15,
                'brightness_limit': 0.15,
                'contrast_limit': 0.25,
                'gamma_limit': (80, 120),
                'noise_var': (5, 25),
                'blur_limit': 5,
                'quality_lower': 80
            },
            'strong': {
                'rotation_limit': 25,
                'brightness_limit': 0.2,
                'contrast_limit': 0.3,
                'gamma_limit': (75, 125),
                'noise_var': (10, 35),
                'blur_limit': 7,
                'quality_lower': 70
            }
        }
        
        params = intensity_params[intensity]
        
        if not is_training:
            # Validation: only resize and normalize
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Training augmentations
        augmentations = [
            # Geometric transformations (clinically plausible)
            A.Rotate(
                limit=params['rotation_limit'], 
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=params['rotation_limit'],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            
            # Perspective and distortion (subtle)
            A.OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.05,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=0.3
            ),
            
            # Intensity and contrast (dental X-ray specific)
            A.OneOf([
                A.CLAHE(
                    clip_limit=4.0,
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=params['brightness_limit'],
                    contrast_limit=params['contrast_limit'],
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=params['gamma_limit'],
                    p=1.0
                )
            ], p=0.8),
            
            # Histogram equalization variants
            A.OneOf([
                A.Equalize(p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            ], p=0.4),
            
            # Noise simulation (imaging artifacts)
            A.OneOf([
                A.GaussNoise(
                    var_limit=params['noise_var'],
                    p=1.0
                ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05),
                    p=1.0
                )
            ], p=0.4),
            
            # Blur and sharpening (image quality variations)
            A.OneOf([
                A.MotionBlur(blur_limit=params['blur_limit'], p=1.0),
                A.GaussianBlur(blur_limit=params['blur_limit'], p=1.0),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0)
            ], p=0.3),
            
            # Compression artifacts (digital imaging)
            A.ImageCompression(
                quality_lower=params['quality_lower'],
                quality_upper=100,
                p=0.2
            ),
            
            # Cutout/Erasing (simulate missing regions)
            A.CoarseDropout(
                max_holes=3,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),
            
            # Final resize and normalization
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return A.Compose(augmentations)
    
    @staticmethod
    def create_scatter_simulation(
        scatter_intensity: float = 0.3,
        scatter_pattern: str = 'gaussian'  # 'gaussian', 'uniform', 'poisson'
    ) -> A.Compose:
        """
        Simulate X-ray scatter artifacts (specific to radiographic imaging)
        """
        def scatter_transform(image, **kwargs):
            h, w = image.shape[:2]
            
            if scatter_pattern == 'gaussian':
                # Gaussian scatter pattern
                scatter = np.random.normal(0, scatter_intensity * 255, (h, w))
            elif scatter_pattern == 'uniform':
                # Uniform scatter
                scatter = np.random.uniform(-scatter_intensity * 255, 
                                          scatter_intensity * 255, (h, w))
            else:  # poisson
                # Poisson noise (photon noise)
                vals = len(np.unique(image))
                vals = 2 ** np.ceil(np.log2(vals))
                noisy = np.random.poisson(image * vals) / float(vals)
                return np.clip(noisy, 0, 255).astype(np.uint8)
            
            # Add scatter to image
            if len(image.shape) == 3:
                scatter = np.stack([scatter] * image.shape[2], axis=2)
            
            scattered = image + scatter
            return np.clip(scattered, 0, 255).astype(np.uint8)
        
        return A.Lambda(image=scatter_transform, p=1.0)
    
    @staticmethod
    def create_beam_hardening_simulation(intensity: float = 0.2) -> A.Compose:
        """
        Simulate beam hardening artifacts common in dental X-rays
        """
        def beam_hardening(image, **kwargs):
            h, w = image.shape[:2]
            
            # Create radial intensity variation
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Beam hardening effect (darker towards edges)
            hardening_factor = 1 - intensity * (dist / max_dist)
            hardening_factor = np.clip(hardening_factor, 1 - intensity, 1)
            
            if len(image.shape) == 3:
                hardening_factor = np.stack([hardening_factor] * image.shape[2], axis=2)
            
            hardened = image * hardening_factor
            return np.clip(hardened, 0, 255).astype(np.uint8)
        
        return A.Lambda(image=beam_hardening, p=1.0)


def update_train_transforms(args):
    """
    Updated training transforms with dental-specific augmentations
    Replace the existing transform creation in train.py _setup_data method
    """
    
    # Create enhanced dental transforms
    train_transform = DentalAugmentations.create_dental_transform(
        image_size=args.image_size,
        is_training=True,
        intensity='medium'  # Can be made configurable via args
    )
    
    val_transform = DentalAugmentations.create_dental_transform(
        image_size=args.image_size,
        is_training=False
    )
    
    return train_transform, val_transform

def get_class_names(class_group: str) -> List[str]:
    """Get class names for simplified class groups"""
    class_groups = {
        'bone-loss': ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"],
        'dental-work': ['bridge', 'filling', 'crown', 'implant'],
        'decay-enamel-pulp': ['decay', 'enamel', 'pulp'],
        'surfaces': ['distal surface', 'occlusal surface', 'mesial surface'],
        'adult-teeth': [f"T{str(i).zfill(2)}" for i in range(1, 33)],
        'primary-teeth': [f"P{str(i).zfill(2)}" for i in range(1, 21)]
    }
    
    if class_group not in class_groups:
        raise ValueError(f"Invalid class_group: {class_group}. Must be one of: {list(class_groups.keys())}")
    
    return class_groups[class_group]


class DentalDataset(Dataset):
    def __init__(self, annotations_file: str, images_dir: str, image_size: int = 224, 
                 is_training: bool = True, augmentation_intensity: str = 'medium',
                 class_group: str = 'bone-loss', xray_type: Optional[str] = None,
                 subset_size: Optional[int] = None):
        self.annotations_file = annotations_file
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.class_group = class_group
        self.xray_type = xray_type
        self.subset_size = subset_size
        
        # Get class names for this group
        self.class_names = get_class_names(class_group)
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"🎯 Loading dataset for class group: {class_group}")
        print(f"📋 Classes: {self.class_names}")
        if xray_type:
            print(f"📷 X-ray type filter: {xray_type}")
        
        # Use dental-specific augmentations
        self.transform = DentalAugmentations.create_dental_transform(
            image_size=image_size,
            is_training=is_training,
            intensity=augmentation_intensity
        )
        
        # Fallback transform for PIL images
        self.torch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and process annotations
        self._load_coco_annotations()
    
    def _load_coco_annotations(self):
        """Load COCO format annotations and filter by class group"""
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        target_classes = set(self.class_names)
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        class_counts = Counter()
        
        for ann in coco_data['annotations']:
            cat_name = self.categories.get(ann['category_id'], 'unknown')
            if cat_name in target_classes:
                image_annotations[ann['image_id']].append({
                    'category_name': cat_name,
                    'category_id': self.class_to_idx[cat_name],
                    'bbox': ann.get('bbox', []),
                    'segmentation': ann.get('segmentation', [])
                })
                class_counts[cat_name] += 1
        
        # Process images
        self.samples = []
        for img_info in coco_data['images']:
            # Apply xray_type filter
            if self.xray_type:
                img_xray_type = img_info.get('xray_type', '').lower()
                allowed_types = [t.strip().lower() for t in self.xray_type.split(',')]
                if img_xray_type not in allowed_types:
                    continue
            
            img_id = img_info['id']
            if img_id in image_annotations:
                # For simplicity, use the first annotation's class as the image label
                # In practice, you might want multi-label classification
                annotations = image_annotations[img_id]
                if annotations:
                    img_path = self.images_dir / img_info['file_name']
                    # Use first annotation's class for single-label classification
                    label = annotations[0]['category_id']
                    self.samples.append((str(img_path), label, annotations))
        
        # Apply subset if specified
        if self.subset_size and len(self.samples) > self.subset_size:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, self.subset_size)
        
        print(f"✅ Loaded {len(self.samples)} samples")
        print(f"📊 Class distribution: {dict(class_counts)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, annotations = self.samples[idx]
        
        if os.path.exists(img_path):
            # Load real image with cv2 for albumentations
            image = cv2.imread(img_path)
            if image is None:
                # Fallback to PIL
                image = Image.open(img_path).convert('RGB')
                return self.torch_transform(image), label
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply albumentations transform
            try:
                augmented = self.transform(image=image)
                image_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1)
            except Exception as e:
                # Fallback to torchvision transform
                image_pil = Image.fromarray(image)
                image_tensor = self.torch_transform(image_pil)
        else:
            # Create dummy image tensor for testing
            print(f"⚠️ Image not found: {img_path}")
            image_tensor = torch.randn(3, self.image_size, self.image_size)
        
        return image_tensor, label


# Rename DummyDataset to DentalDataset for backward compatibility
DummyDataset = DentalDataset


def convert_annotations_for_training(train_annotations: str, val_annotations: str,
                                   images_dir: str, class_group: str, xray_type: Optional[str] = None):
    """Convert COCO annotations to format suitable for training"""
    # For SageMaker environment
    if os.getenv('SM_TRAINING_ENV'):
        base_path = Path('/opt/ml/input/data/train')
        train_ann_path = base_path / train_annotations
        val_ann_path = base_path / val_annotations
        images_path = base_path / 'images'
    else:
        # Local environment
        base_path = Path('.')
        train_ann_path = Path(train_annotations)
        val_ann_path = Path(val_annotations)
        images_path = Path(images_dir)
    
    return train_ann_path, val_ann_path, images_path


def train(args):
    print(f"🧠 Starting MedMamba training for {args.epochs} epochs...")
    print(f"🎯 Class group: {args.class_group}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get paths for annotations and images
    train_ann_path, val_ann_path, images_path = convert_annotations_for_training(
        args.train_annotations, args.val_annotations, args.images_dir, 
        args.class_group, args.xray_type
    )
    
    # Create datasets
    train_dataset = DentalDataset(
        annotations_file=str(train_ann_path),
        images_dir=str(images_path),
        image_size=args.image_size,
        is_training=True,
        augmentation_intensity=args.augmentation_intensity,
        class_group=args.class_group,
        xray_type=args.xray_type,
        subset_size=args.subset_size
    )
    
    val_dataset = None
    if val_ann_path.exists():
        val_dataset = DentalDataset(
            annotations_file=str(val_ann_path),
            images_dir=str(images_path),
            image_size=args.image_size,
            is_training=False,
            augmentation_intensity='light',
            class_group=args.class_group,
            xray_type=args.xray_type,
            subset_size=None  # Use full validation set
        )
    
    # Get number of classes from dataset
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = MedMamba(num_classes=num_classes).to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"📈 Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%")
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            print(f"📊 Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(args.model_dir, exist_ok=True)
                best_model_path = os.path.join(args.model_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'acc': val_acc,
                    'class_group': args.class_group,
                    'class_names': train_dataset.class_names
                }, best_model_path)
                print(f"💾 Saved best model with val loss: {avg_val_loss:.4f}")
        
        scheduler.step()

    # Save final model
    os.makedirs(args.model_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'image_size': args.image_size,
        'class_group': args.class_group,
        'class_names': train_dataset.class_names
    }, final_model_path)
    print(f"✅ Saved final model to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MedMamba on dental X-ray dataset')
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay for optimizer")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train",
                        help="Base data directory (for backward compatibility)")
    parser.add_argument("--images_dir", type=str, default="images",
                        help="Images directory relative to data_dir")
    parser.add_argument("--train_annotations", type=str, default="train.json",
                        help="Training annotations file")
    parser.add_argument("--val_annotations", type=str, default="val.json",
                        help="Validation annotations file")
    
    # Class group selection (simplified)
    parser.add_argument("--class_group", type=str, default="bone-loss",
                        choices=['bone-loss', 'dental-work', 'decay-enamel-pulp', 
                                'surfaces', 'adult-teeth', 'primary-teeth'],
                        help="Simplified class group to train on")
    
    # Dataset filtering
    parser.add_argument("--xray_type", type=str, default=None,
                        help="Filter by X-ray type (e.g., 'bitewing,periapical')")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Use subset of dataset for testing")
    
    # Augmentation parameters
    parser.add_argument("--augmentation_intensity", type=str, default="medium",
                        choices=["light", "medium", "strong"],
                        help="Intensity of data augmentations")
    
    # System parameters
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model",
                        help="Directory to save trained models")
    
    args = parser.parse_args()
    
    train(args)
