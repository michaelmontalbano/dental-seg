#!/usr/bin/env python3
"""
Training script for MedSAM dental landmark segmentation
Interactive prompt-based segmentation for dental landmarks using MedSAM
"""

import subprocess
import sys
import os
import argparse
import torch
import logging
from pathlib import Path
import yaml
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install MedSAM and required packages in SageMaker environment"""
    logger.info("🔧 Installing MedSAM and dependencies...")
    
    try:
        # Install basic requirements first
        requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if os.path.exists(requirements_file):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
            logger.info("✅ Basic requirements installed")
        
        # Install segment-anything
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'])
        logger.info("✅ Segment Anything installed")
        
        # Install additional medical imaging dependencies
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'monai>=1.0.0'])
        logger.info("✅ MONAI installed")
        
        # Install MedSAM (if available as package, otherwise we'll implement from scratch)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medsam'])
            logger.info("✅ MedSAM package installed")
        except:
            logger.info("ℹ️ MedSAM package not available, will use custom implementation")
            
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        logger.info("🔄 Continuing with available packages...")

# Install requirements if in SageMaker environment
if os.getenv('SM_TRAINING_ENV') or os.getenv('SM_MODEL_DIR'):
    install_requirements()

# Import dependencies after installation
try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.utils.transforms import ResizeLongestSide
    logger.info("✅ Successfully imported SAM components")
except ImportError as e:
    logger.error(f"❌ Failed to import SAM: {e}")
    logger.info("Installing segment-anything...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'])
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.utils.transforms import ResizeLongestSide

# Import wandb if available
try:
    import wandb
except ImportError:
    wandb = None

def convert_annotations_to_medsam(train_annotations_file='train.json', val_annotations_file='val.json', class_group='bone-loss', xray_type=None):
    """Convert separate train.json and val.json to MedSAM format with prompts"""
    try:
        logger.info("🔄 Converting train.json and val.json to MedSAM format...")
        
        # Import dependencies
        import json
        import shutil
        from pathlib import Path
        from PIL import Image
        from collections import Counter
        import numpy as np
        
        # SageMaker paths
        input_path = Path('/opt/ml/input/data/train')
        train_annotations_path = input_path / train_annotations_file
        val_annotations_path = input_path / val_annotations_file
        images_dir = input_path / 'images'
        output_dir = Path('/tmp/medsam_dataset')
        
        # Class configuration based on class_group
        if class_group == 'bone-loss':
            class_names = ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        elif class_group == 'teeth':
            class_names = [f"T{str(i).zfill(2)}" for i in range(1, 33)] + [f"P{str(i).zfill(2)}" for i in range(1, 21)]
        elif class_group == 'conditions':
            class_names = ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                          'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant']
        elif class_group == 'surfaces':
            class_names = ['distal surface', 'occlusal surface', 'mesial surface']
        elif class_group == 'all':
            class_names = (["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"] +
                          [f"T{str(i).zfill(2)}" for i in range(1, 33)] + 
                          [f"P{str(i).zfill(2)}" for i in range(1, 21)] +
                          ['bridge', 'margin', 'enamel', 'coronal aspect', 'pulp', 'root aspect',
                           'filling', 'crown', 'impaction', 'decay', 'rct', 'parl', 'missing', 'implant'] +
                          ['distal surface', 'occlusal surface', 'mesial surface'])
        else:
            class_names = ["cej mesial", "cej distal", "ac mesial", "ac distal", "apex"]
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        logger.info(f"📁 Input: {input_path}")
        logger.info(f"📄 Train annotations: {train_annotations_path}")
        logger.info(f"📄 Val annotations: {val_annotations_path}")
        logger.info(f"🖼️ Images: {images_dir}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"🎯 Class group: {class_group}")
        logger.info(f"🎯 Classes: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
        if xray_type:
            logger.info(f"📷 X-ray type filter: {xray_type}")
        
        # Validate paths
        if not train_annotations_path.exists():
            raise FileNotFoundError(f"Train annotations not found: {train_annotations_path}")
        if not val_annotations_path.exists():
            raise FileNotFoundError(f"Val annotations not found: {val_annotations_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images not found: {images_dir}")
        
        # Load both annotation files
        with open(train_annotations_path, 'r') as f:
            train_annotations = json.load(f)
        with open(val_annotations_path, 'r') as f:
            val_annotations = json.load(f)
        
        logger.info(f"✅ Loaded train annotations: {type(train_annotations)}")
        logger.info(f"✅ Loaded val annotations: {type(val_annotations)}")
        
        # Process both train and val annotations
        def process_annotations(annotations, split_name):
            """Process COCO format annotations for MedSAM"""
            if isinstance(annotations, dict) and 'images' in annotations:
                categories = {cat['id']: cat['name'] for cat in annotations['categories']}
                target_classes = set(class_names)
                
                logger.info(f"📊 {split_name} - Found categories: {len(categories)}")
                logger.info(f"🎯 {split_name} - Target categories: {[k for k,v in categories.items() if v in target_classes]}")
                
                # Group annotations by image
                image_annotations = {}
                target_ann_count = 0
                
                for ann in annotations['annotations']:
                    image_id = ann['image_id']
                    category_name = categories.get(ann['category_id'], 'unknown')
                    
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    
                    # For MedSAM, we need bbox, segmentation, and prompt information
                    custom_ann = {
                        'bbox': ann.get('bbox', []),
                        'segmentation': ann.get('segmentation', []),
                        'class_name': category_name,
                        'category_id': ann['category_id']
                    }
                    image_annotations[image_id].append(custom_ann)
                    
                    if category_name in target_classes:
                        target_ann_count += 1
                
                logger.info(f"✅ {split_name} - Found {target_ann_count} target annotations")
                
                # Create processed data with xray_type filtering
                processed_data = []
                
                for img in annotations['images']:
                    img_id = img['id']
                    
                    # Apply xray_type filtering
                    if xray_type:
                        img_xray_type = img.get('xray_type', '').lower()
                        allowed_types = [t.strip().lower() for t in xray_type.split(',')]
                        if img_xray_type not in allowed_types:
                            continue
                    
                    if img_id in image_annotations:
                        target_anns = [ann for ann in image_annotations[img_id] 
                                     if ann['class_name'] in target_classes]
                        
                        if target_anns:
                            processed_item = {
                                'file_name': img['file_name'],
                                'width': img['width'],
                                'height': img['height'],
                                'id': img_id,
                                'target_annotations': target_anns
                            }
                            processed_data.append(processed_item)
                
                logger.info(f"✅ {split_name} - Processed {len(processed_data)} images with targets")
                if xray_type:
                    logger.info(f"📷 {split_name} - After X-ray type filtering: {len(processed_data)} images")
                
                return processed_data
            else:
                raise ValueError(f"Unsupported annotation format for {split_name}: {type(annotations)}")
        
        # Process train and validation data separately
        train_processed_data = process_annotations(train_annotations, "TRAIN")
        val_processed_data = process_annotations(val_annotations, "VAL")
        
        # Create MedSAM dataset structure
        for split in ['train', 'val']:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_processed_data,
            'val': val_processed_data
        }
        
        # Save processed data for MedSAM training
        total_samples = 0
        
        for split_name, split_data in splits.items():
            logger.info(f"📝 Processing {split_name}: {len(split_data)} images")
            
            split_samples = []
            for item in split_data:
                file_name = item.get('file_name')
                if not file_name:
                    continue
                
                # Copy image
                src_path = images_dir / file_name
                if not src_path.exists():
                    logger.warning(f"⚠️ Image not found: {src_path}")
                    continue
                
                dst_path = output_dir / split_name / file_name
                shutil.copy2(src_path, dst_path)
                
                # Create MedSAM sample with prompts
                img_width = item.get('width', 0)
                img_height = item.get('height', 0)
                
                if img_width == 0 or img_height == 0:
                    with Image.open(src_path) as img:
                        img_width, img_height = img.size
                
                for ann in item.get('target_annotations', []):
                    class_name = ann['class_name']
                    if class_name not in class_to_idx:
                        continue
                    
                    class_id = class_to_idx[class_name]
                    bbox = ann.get('bbox', [])
                    segmentation = ann.get('segmentation', [])
                    
                    if len(bbox) == 4:
                        # Create MedSAM sample
                        sample = {
                            'image_path': str(dst_path),
                            'class_name': class_name,
                            'class_id': class_id,
                            'bbox': bbox,
                            'segmentation': segmentation,
                            'image_width': img_width,
                            'image_height': img_height
                        }
                        split_samples.append(sample)
                        total_samples += 1
            
            # Save split data
            split_file = output_dir / f'{split_name}_samples.json'
            with open(split_file, 'w') as f:
                json.dump(split_samples, f, indent=2)
            
            logger.info(f"✅ {split_name}: {len(split_samples)} MedSAM samples created")
        
        # Create dataset config
        config = {
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'num_classes': len(class_names),
            'image_size': 1024,  # MedSAM standard size
            'class_group': class_group
        }
        
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ MedSAM conversion completed!")
        logger.info(f"📊 Total: {total_samples} MedSAM samples")
        logger.info(f"📄 Config created: {config_path}")
        
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise

class MedSAMDataset(Dataset):
    """Dataset for MedSAM training with prompt generation"""
    
    def __init__(self, samples_file, config_file, prompt_mode='auto', num_prompts=5, 
                 prompt_noise=0.1, negative_prompt_ratio=0.2, image_size=1024):
        self.samples_file = samples_file
        self.prompt_mode = prompt_mode
        self.num_prompts = num_prompts
        self.prompt_noise = prompt_noise
        self.negative_prompt_ratio = negative_prompt_ratio
        self.image_size = image_size
        
        # Load samples and config
        with open(samples_file, 'r') as f:
            self.samples = json.load(f)
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.transform = ResizeLongestSide(image_size)
        
        logger.info(f"📊 Loaded {len(self.samples)} samples for MedSAM training")
        logger.info(f"🎯 Prompt mode: {prompt_mode}, num_prompts: {num_prompts}")
    
    def __len__(self):
        return len(self.samples)
    
    def generate_prompts(self, bbox, segmentation, img_width, img_height):
        """Generate prompts for MedSAM training"""
        prompts = []
        
        if self.prompt_mode in ['auto', 'mixed']:
            # Generate automatic prompts from bbox/segmentation
            x, y, w, h = bbox
            
            # Point prompts (positive)
            for _ in range(self.num_prompts):
                # Sample points within the bbox with noise
                px = x + np.random.uniform(0, w)
                py = y + np.random.uniform(0, h)
                
                # Add noise
                px += np.random.normal(0, self.prompt_noise * w)
                py += np.random.normal(0, self.prompt_noise * h)
                
                # Clamp to image bounds
                px = np.clip(px, 0, img_width - 1)
                py = np.clip(py, 0, img_height - 1)
                
                prompts.append({
                    'type': 'point',
                    'coordinates': [px, py],
                    'label': 1  # positive
                })
            
            # Negative prompts
            num_negative = int(self.num_prompts * self.negative_prompt_ratio)
            for _ in range(num_negative):
                # Sample points outside the bbox
                if np.random.random() < 0.5:
                    # Left or right of bbox
                    px = np.random.uniform(0, img_width)
                    py = np.random.uniform(0, img_height)
                    if px > x and px < x + w and py > y and py < y + h:
                        # If inside bbox, move outside
                        px = x - np.random.uniform(10, 50) if px < x + w/2 else x + w + np.random.uniform(10, 50)
                else:
                    # Above or below bbox
                    px = np.random.uniform(0, img_width)
                    py = y - np.random.uniform(10, 50) if np.random.random() < 0.5 else y + h + np.random.uniform(10, 50)
                
                px = np.clip(px, 0, img_width - 1)
                py = np.clip(py, 0, img_height - 1)
                
                prompts.append({
                    'type': 'point',
                    'coordinates': [px, py],
                    'label': 0  # negative
                })
        
        if self.prompt_mode in ['manual', 'mixed']:
            # Add bbox prompt
            prompts.append({
                'type': 'box',
                'coordinates': bbox,
                'label': 1
            })
        
        return prompts
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        
        # Generate prompts
        prompts = self.generate_prompts(
            sample['bbox'], 
            sample.get('segmentation', []),
            sample['image_width'],
            sample['image_height']
        )
        
        # Transform image
        image = self.transform.apply_image(image)
        
        # Create ground truth mask from segmentation or bbox
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        segmentation = sample.get('segmentation', [])
        if segmentation and len(segmentation) > 0:
            # Use segmentation if available
            # This would need proper polygon to mask conversion
            # For now, use bbox as fallback
            pass
        
        # Fallback to bbox mask
        x, y, w, h = sample['bbox']
        # Scale bbox to transformed image size
        scale_x = image.shape[1] / sample['image_width']
        scale_y = image.shape[0] / sample['image_height']
        
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)
        
        mask[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = 1
        
        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'mask': torch.from_numpy(mask).float(),
            'prompts': prompts,
            'class_id': sample['class_id'],
            'class_name': sample['class_name']
        }

class MedSAMTrainer:
    """MedSAM trainer for dental landmark segmentation"""
    
    def __init__(self, sam_model_type='vit_h', device='auto'):
        self.sam_model_type = sam_model_type
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        logger.info(f"🤖 MedSAM Trainer initialized with {sam_model_type} on {self.device}")
        
    def load_model(self, sam_checkpoint=None, medsam_checkpoint=None):
        """Load SAM/MedSAM model"""
        try:
            # Download SAM checkpoint if not provided
            if not sam_checkpoint:
                sam_checkpoint = self.download_sam_checkpoint()
            
            # Load SAM model
            self.model = sam_model_registry[self.sam_model_type](checkpoint=sam_checkpoint)
            self.model.to(self.device)
            
            # Load MedSAM checkpoint if available
            if medsam_checkpoint and os.path.exists(medsam_checkpoint):
                logger.info(f"Loading MedSAM checkpoint: {medsam_checkpoint}")
                checkpoint = torch.load(medsam_checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"✅ Successfully loaded {self.sam_model_type} model")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
        
        return self.model
    
    def download_sam_checkpoint(self):
        """Download SAM checkpoint"""
        import urllib.request
        
        checkpoint_urls = {
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        }
        
        url = checkpoint_urls[self.sam_model_type]
        checkpoint_path = f'/tmp/sam_{self.sam_model_type}.pth'
        
        if not os.path.exists(checkpoint_path):
            logger.info(f"📥 Downloading SAM checkpoint: {url}")
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"✅ Downloaded to: {checkpoint_path}")
        
        return checkpoint_path
    
    def setup_training_config(self, args):
        """Setup training configuration for MedSAM"""
        
        config = {
            # Core parameters
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'image_size': args.image_size,
            'device': self.device,
            'workers': min(args.workers, 4),
            
            # MedSAM-specific parameters
            'prompt_mode': args.prompt_mode,
            'num_prompts': args.num_prompts,
            'prompt_noise': args.prompt_noise,
            'negative_prompt_ratio': args.negative_prompt_ratio,
            'mask_threshold': args.mask_threshold,
            
            # Model configuration
            'sam_model_type': self.sam_model_type,
            'freeze_image_encoder': args.freeze_image_encoder,
            'freeze_prompt_encoder': args.freeze_prompt_encoder,
            
            # Training optimization
            'gradient_checkpointing': args.gradient_checkpointing,
            'mixed_precision': args.mixed_precision,
            'warmup_epochs': args.warmup_epochs,
            'weight_decay': args.weight_decay,
        }
        
        return config
    
    def train(self, dataset_dir, args):
        """Train MedSAM model"""
        logger.info("🚀 Starting MedSAM training for dental landmarks")
        
        if not self.model:
            self.load_model(args.sam_checkpoint, args.medsam_checkpoint)
        
        config = self.setup_training_config(args)
        
        # Create datasets
        train_dataset = MedSAMDataset(
            os.path.join(dataset_dir, 'train_samples.json'),
            os.path.join(dataset_dir, 'config.json'),
            prompt_mode=args.prompt_mode,
            num_prompts=args.num_prompts,
            prompt_noise=args.prompt_noise,
            negative_prompt_ratio=args.negative_prompt_ratio,
            image_size=args.image_size
        )
        
        val_dataset = MedSAMDataset(
            os.path.join(dataset_dir, 'val_samples.json'),
            os.path.join(dataset_dir, 'config.json'),
            prompt_mode=args.prompt_mode,
            num_prompts=args.num_prompts,
            prompt_noise=args.prompt_noise,
            negative_prompt_ratio=args.negative_prompt_ratio,
            image_size=args.image_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers
        )
        
        # Setup optimizer
        if args.freeze_image_encoder:
            # Only train mask decoder
            params = list(self.model.mask_decoder.parameters())
            logger.info("🔒 Freezing image encoder, training mask decoder only")
        else:
            params = list(self.model.parameters())
            logger.info("🔓 Training full model")
        
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Setup loss function
        criterion = nn.BCEWithLogitsLoss()
        
        if args.wandb and wandb:
            wandb.init(project=args.project, name=args.name, config=config)
        
        try:
            logger.info(f"🎯 Starting MedSAM training")
            logger.info(f"📊 Training config: {config}")
            logger.info(f"📊 Train samples: {len(train_dataset)}")
            logger.info(f"📊 Val samples: {len(val_dataset)}")
            
            best_val_loss = float('inf')
            best_model_path = None
            
            for epoch in range(args.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # This is a simplified training loop
                    # In practice, you'd need to implement proper prompt handling
                    # and SAM-specific forward pass
                    
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    # Forward pass (simplified)
                    # In real implementation, you'd use SAM's predictor with prompts
                    outputs = self.model(images)  # This needs proper implementation
                    
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = f'/tmp/medsam_best_epoch_{epoch}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'config': config
                    }, best_model_path)
                    logger.info(f"💾 Saved best model: {best_model_path}")
                
                if args.wandb and wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss
                    })
            
            logger.info("✅ Training completed successfully!")
            return best_model_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(description='Train MedSAM for dental landmark segmentation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=2)  # Smaller for MedSAM
    parser.add_argument('--learning-rate', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-classes', '--num_classes', type=int, default=5)
    parser.add_argument('--image-size', '--image_size', type=int, default=1024)
    parser.add_argument('--subset-size', '--subset_size', type=int, default=None)
    parser.add_argument('--sam-model-type', '--sam_model_type', type=str, default='vit_h')
    parser.add_argument('--project', type=str, default='dental_medsam')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './runs'))
    parser.add_argument('--sam-checkpoint', '--sam_checkpoint', type=str, default=None)
    parser.add_argument('--medsam-checkpoint', '--medsam_checkpoint', type=str, default=None)
    parser.add_argument('--weight-decay', '--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', '--warmup_epochs', type=int, default=3)
    
    # MedSAM-specific parameters
    parser.add_argument('--prompt-mode', '--prompt_mode', type=str, default='auto',
                        choices=['auto', 'manual', 'mixed'],
                        help='Prompt generation mode')
    parser.add_argument('--num-prompts', '--num_prompts', type=int, default=5,
                        help='Number of prompts to generate per sample')
    parser.add_argument('--prompt-noise', '--prompt_noise', type=float, default=0.1,
                        help='Noise level for prompt generation')
    parser.add_argument('--negative-prompt-ratio', '--negative_prompt_ratio', type=float, default=0.2,
                        help='Ratio of negative prompts')
    parser.add_argument('--mask-threshold', '--mask_threshold', type=float, default=0.5,
                        help='Mask prediction threshold')
    parser.add_argument('--freeze-image-encoder', '--freeze_image_encoder', action='store_true',
                        help='Freeze image encoder during training')
    parser.add_argument('--freeze-prompt-encoder', '--freeze_prompt_encoder', action='store_true',
                        help='Freeze prompt encoder during training')
    parser.add_argument('--gradient-checkpointing', '--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--mixed-precision', '--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Dataset filtering parameters
    parser.add_argument('--train-annotations', '--train_annotations', type=str, default='train.json')
    parser.add_argument('--val-annotations', '--val_annotations', type=str, default='val.json')
    parser.add_argument('--class-group', '--class_group', type=str, default='bone-loss',
                        choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth'])
    parser.add_argument('--xray-type', '--xray_type', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("🎯 === MedSAM Dental Landmark Training ===")
    logger.info(f"📊 Arguments: {vars(args)}")
    logger.info(f"🎯 Class group: {args.class_group}")
    if args.xray_type:
        logger.info(f"📷 X-ray type: {args.xray_type}")
    
    # Create/convert dataset
    logger.info("📄 Converting annotations to MedSAM format...")
    dataset_dir = convert_annotations_to_medsam(
        train_annotations_file=args.train_annotations,
        val_annotations_file=args.val_annotations,
        class_group=args.class_group,
        xray_type=args.xray_type
    )
    logger.info(f"✅ Using dataset: {dataset_dir}")
    
    # Set experiment name
    if not args.name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_parts = [f'medsam_{args.sam_model_type}', args.class_group]
        if args.xray_type:
            name_parts.append(args.xray_type)
        name_parts.append(timestamp)
        if args.subset_size:
            name_parts.append(f"subset{args.subset_size}")
        args.name = '_'.join(name_parts)
    
    # Initialize trainer
    trainer = MedSAMTrainer(sam_model_type=args.sam_model_type, device=args.device)
    
    try:
        # Train model
        best_model_path = trainer.train(dataset_dir, args)
        
        # Copy weights to model directory for SageMaker
        if args.model_dir and best_model_path and os.path.exists(best_model_path):
            os.makedirs(args.model_dir, exist_ok=True)
            import shutil
            shutil.copy2(best_model_path, os.path.join(args.model_dir, 'best_medsam.pth'))
            logger.info(f"📦 Best weights copied to {args.model_dir}/best_medsam.pth")
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available. Training will be slow on CPU.")
    else:
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    main()
