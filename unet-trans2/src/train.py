#!/usr/bin/env python3
"""
train.py
Pure TransUNet for CEJ/AC landmark segmentation
Converts bounding boxes to masks for training, masks to boxes for evaluation
Updated with DiceFocalLoss for class imbalance and configurable class groups
FIXED: Dimension mismatch, separate train/val files, X-ray filtering, bone-loss bbox handling
"""
from pycocotools import mask as coco_mask
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import json
from PIL import Image, ImageDraw
import cv2

from dataset import BboxDataset, collate_fn, CLASS_GROUPS
from losses import DiceFocalLoss, get_loss_for_imbalance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_class_names_for_class_group(class_group):
    """Get class names for the specified class group"""
    if class_group == 'all':
        # Use all classes from all groups
        class_names = []
        for group_classes in CLASS_GROUPS.values():
            class_names.extend(group_classes)
        return class_names
    elif class_group in CLASS_GROUPS:
        return CLASS_GROUPS[class_group]
    else:
        raise ValueError(f"Invalid class_group: {class_group}. Must be 'all' or one of: {list(CLASS_GROUPS.keys())}")

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    """Transformer block"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    """Pure TransUNet for landmark segmentation - FIXED dimension mismatch"""
    def __init__(self, num_classes=6, img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN Encoder (ResNet50 backbone)
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels, 1/4 resolution
        self.layer2 = resnet.layer2  # 512 channels, 1/8 resolution
        self.layer3 = resnet.layer3  # 1024 channels, 1/16 resolution
        
        # FIXED: Calculate actual feature size after ResNet layers
        # ResNet reduces by 16x total: conv1+maxpool(4x) + layer2(2x) + layer3(2x) = 16x
        feature_size = img_size // 16
        
        # FIXED: Patch embedding for transformer - ensure consistency
        self.patch_embed = PatchEmbed(
            img_size=feature_size,  # This is the actual feature map size
            patch_size=1,  # 1x1 patches on feature map
            in_chans=1024, 
            embed_dim=embed_dim
        )
        
        # FIXED: Positional embedding - use actual number of patches
        num_patches = feature_size * feature_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # DEBUG: Print dimensions for verification
        logger.info(f"TransUNet init: img_size={img_size}, feature_size={feature_size}, num_patches={num_patches}")
        logger.info(f"Positional embedding shape: (1, {num_patches}, {embed_dim})")
        
        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder (CNN upsampling path)
        self.decoder4 = nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2)  # 1/8
        self.decoder3 = nn.ConvTranspose2d(512 + 512, 256, kernel_size=2, stride=2)  # 1/4
        self.decoder2 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)  # 1/2
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 1/1
        
        # Final segmentation head
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # CNN Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)    # 1/4, 256 channels
        x2 = self.layer2(x1)   # 1/8, 512 channels  
        x3 = self.layer3(x2)   # 1/16, 1024 channels
        
        # Transformer processing
        x_transformer = self.patch_embed(x3)
        
        # FIXED: Now dimensions should match
        x_transformer = x_transformer + self.pos_embed
        
        for transformer_block in self.transformer:
            x_transformer = transformer_block(x_transformer)
        x_transformer = self.norm(x_transformer)
        
        # Reshape back to spatial
        feature_size = self.img_size // 16
        x_transformer = x_transformer.transpose(1, 2).reshape(B, self.embed_dim, feature_size, feature_size)
        
        # CNN Decoder with skip connections
        d4 = self.decoder4(x_transformer)  # 1/8
        d4 = torch.cat([d4, x2], dim=1)    # Skip connection
        
        d3 = self.decoder3(d4)             # 1/4
        d3 = torch.cat([d3, x1], dim=1)    # Skip connection
        
        d2 = self.decoder2(d3)             # 1/2
        d1 = self.decoder1(d2)             # 1/1 (full resolution)
        
        # Final segmentation
        out = self.final_conv(d1)
        
        # Resize to original input size if needed
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return out

class SegmentationDataset(BboxDataset):
    """Extends the working BboxDataset to output segmentation masks instead of boxes"""
        
    def __init__(self, annotations_file, images_dir, transforms=None, img_size=512, class_group='all', xray_type=None):
        super().__init__(annotations_file, images_dir, transforms, class_group, xray_type)
        self.img_size = img_size
        self.class_group = class_group
        self.xray_type = xray_type
        
        # NEW: For bone-loss class group, use bounding boxes instead of segmentation
        self.use_bboxes_for_masks = (class_group == 'bone-loss')
        if self.use_bboxes_for_masks:
            logger.info(f"🎯 Using BOUNDING BOXES for {class_group} class group (no segmentation decoding)")
        
        # Initialize debug tracking - REDUCED VERBOSITY
        self.debug_sample_count = 0
        self.field_debug_stats = {
            'decode_success': 0,
            'decode_failure': 0,
            'bbox_fallback': 0
        }
        
        # Control verbosity
        self.verbose_debug = False  # Set to True for detailed debugging
        
        # Statistics tracking
        self.mask_usage_stats = {
            'using_segmentation': 0,
            'using_bbox_fallback': 0,
            'decode_failures': 0
        }
        
        # Update class mapping to include background
        self.seg_class_names = ["background"] + self.class_names
        self.seg_class_to_idx = {name: idx for idx, name in enumerate(self.seg_class_names)}
        
        filter_desc = f"{class_group}"
        if xray_type:
            filter_desc += f" + {xray_type}"
        logger.info(f"Segmentation classes for {filter_desc}: {self.seg_class_names}")

    def decode_mask(self, mask_data, original_height, original_width):
        """Decode mask data into a numpy array"""
        if isinstance(mask_data, list):
            # Handle COCO polygon format
            if len(mask_data) > 0 and isinstance(mask_data[0], list):
                # Polygon format - convert to mask
                
                # Create RLE from polygon
                rle = coco_mask.frPyObjects(mask_data, original_height, original_width)
                decoded = coco_mask.decode(rle)
                if len(decoded.shape) == 3:
                    decoded = decoded[:, :, 0]  # Take first channel
                return decoded
            else:
                return np.array(mask_data, dtype=np.uint8)
        return mask_data
    
    def __getitem__(self, idx):
        # Get the original image and target from parent class
        image, target = super().__getitem__(idx)
        
        # Get the original item to access mask data
        item = self.filtered_data[idx]
        
        # Convert masks to segmentation mask
        C, H, W = image.shape
        mask = torch.zeros((H, W), dtype=torch.long)
        
        # Get original image dimensions for proper decoding
        original_width = item.get('width', W)
        original_height = item.get('height', H)
        
        # Handle annotations with actual mask data
        annotations = []
        if isinstance(item, dict):
            annotations = item.get('relevant_annotations', [])
        
        # Minimal debug logging
        if self.debug_sample_count < 1:
            logger.info(f"Processing first image: {item.get('file_name', 'unknown')}")
        
        successful_masks = 0
        bbox_fallbacks = 0
        decode_failures = 0
        
        for ann_idx, ann in enumerate(annotations):
            # Get class label
            class_name = ann['normalized_class']
            if class_name not in self.class_to_idx:
                continue
            class_label = self.class_to_idx[class_name]
            
            # 🔍 Check what mask fields are available
            has_mask = 'mask' in ann
            has_masks = 'masks' in ann  
            has_segmentation = 'segmentation' in ann
            
            # 🔍 DEBUG: Log for first few annotations
            if self.debug_sample_count < 3 and ann_idx < 2:
                logger.info(f"   Annotation {ann_idx}: {class_name}")
                logger.info(f"     Has segmentation: {has_segmentation}")
                logger.info(f"     Has mask: {has_mask}")
                logger.info(f"     Has masks: {has_masks}")
                logger.info(f"     Fields: {list(ann.keys())}")
            
            # NEW: For bone-loss, always use bounding boxes
            if self.use_bboxes_for_masks:
                bbox_fallbacks += 1
                self._apply_bbox_mask(ann, mask, class_label, H, W)
                continue
            
            # Check if annotation has mask data
            mask_data = ann.get('segmentation') or ann.get('mask') or ann.get('masks')
            
            if mask_data is not None:
                # ✅ Use actual segmentation mask
                try:
                    # Use decode_mask method
                    decoded_mask = self.decode_mask(mask_data, original_height, original_width)
                    
                    # Resize mask to match current image size if needed
                    if decoded_mask.shape != (H, W):
                        decoded_mask = cv2.resize(decoded_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # Apply the mask with the class label
                    pixels_set = np.sum(decoded_mask > 0) 
                    if pixels_set > 0:
                        mask[decoded_mask > 0] = class_label
                        successful_masks += 1
                        
                        # 🔍 DEBUG: Log successful decodings
                        if successful_masks <= 3 and self.debug_sample_count < 3:
                            logger.info(f"✅ Successfully used segmentation mask for {class_name}")
                            logger.info(f"   Mask shape: {decoded_mask.shape}, Non-zero pixels: {pixels_set}")
                    else:
                        # Silently fall back to bbox if decoded mask is empty
                        bbox_fallbacks += 1
                        self._apply_bbox_mask(ann, mask, class_label, H, W)
                        
                except Exception as e:
                    # Silently fall back to bounding box on decode failure
                    decode_failures += 1
                    bbox_fallbacks += 1
                    self._apply_bbox_mask(ann, mask, class_label, H, W)
            else:
                # Silently fall back to bounding box if no mask data
                bbox_fallbacks += 1
                self._apply_bbox_mask(ann, mask, class_label, H, W)
        
        # Update debug counters
        self.field_debug_stats['decode_success'] += successful_masks
        self.field_debug_stats['decode_failure'] += decode_failures
        self.field_debug_stats['bbox_fallback'] += bbox_fallbacks
        self.debug_sample_count += 1
        
        # Minimal periodic statistics (only every 500 samples)
        if self.debug_sample_count % 500 == 0 and self.debug_sample_count > 0:
            total_processed = sum(self.field_debug_stats.values())
            if total_processed > 0:
                bbox_pct = self.field_debug_stats['bbox_fallback'] / total_processed * 100
                success_pct = self.field_debug_stats['decode_success'] / total_processed * 100
                
                if bbox_pct > 50 and not self.use_bboxes_for_masks:
                    logger.warning(f"WARNING: {bbox_pct:.1f}% using bbox fallback")
        
        return image, mask

    def _apply_bbox_mask(self, ann, mask, class_label, H, W):
        """Helper function to apply bounding box as mask"""
        bbox = ann.get('bbox', ann.get('bounding_box', []))
        if len(bbox) == 4:
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = class_label

def calculate_class_weights_from_annotations(dataset, class_names, background_weight=0.1):
    """
    Calculate class weights based on actual annotation counts in the filtered dataset
    
    Args:
        dataset: The segmentation dataset
        class_names: List of class names being trained on (excluding background)
        background_weight: Weight for background class (typically low)
    """
    
    # Count annotations for each class in the actual training data
    class_counts = {name: 0 for name in class_names}
    total_annotations = 0
    
    # Iterate through dataset to count actual annotations
    for i in range(len(dataset)):
        try:
            # Get the underlying dataset if it's a Subset
            if hasattr(dataset, 'dataset'):
                item = dataset.dataset.filtered_data[dataset.indices[i]]
            else:
                item = dataset.filtered_data[i]
            
            annotations = item.get('relevant_annotations', [])
            for ann in annotations:
                class_name = ann.get('normalized_class', '')
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    total_annotations += 1
        except:
            continue
    
    logger.info(f"📊 Actual annotation counts in training data:")
    for class_name, count in class_counts.items():
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        logger.info(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    # Calculate weights using inverse frequency with smoothing
    weights = []
    
    # Background weight (class 0) - typically low since it's dominant
    weights.append(background_weight)
    
    # Calculate weights for landmark classes
    for class_name in class_names:
        count = class_counts[class_name]
        if count > 0:
            # Inverse frequency with sqrt smoothing for gentler weighting
            weight = np.sqrt(total_annotations / count)
        else:
            weight = 1.0  # Default weight for missing classes
        weights.append(weight)
    
    # Normalize weights (optional - prevents extreme values)
    weights = np.array(weights)
    weights = weights / weights.mean()  # Normalize around 1.0
    
    # Cap extreme weights
    max_weight = 10.0
    weights = np.clip(weights, 0.1, max_weight)
    
    logger.info(f"📈 Calculated class weights:")
    logger.info(f"  Background (class 0): {weights[0]:.3f}")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name} (class {i+1}): {weights[i+1]:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)

def analyze_class_distribution(data_loader, num_classes, max_batches=20):
    """Analyze class distribution in the dataset"""
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    logger.info("Analyzing class distribution...")
    for i, (images, masks) in enumerate(data_loader):
        if i >= max_batches:  # Sample first N batches
            break
            
        for class_idx in range(num_classes):
            class_counts[class_idx] += (masks == class_idx).sum().item()
        total_pixels += masks.numel()
    
    # Calculate percentages
    class_percentages = class_counts / total_pixels * 100
    
    logger.info("Class distribution in training data:")
    
    return class_counts.tolist()

# Keep all the metric calculation functions (unchanged)
def find_landmark_instances(mask, min_area=10):
    """Find individual landmark instances in a mask using connected components"""
    import cv2
    
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    instances = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:  # Filter small noise
            centroid_x = centroids[i, 0]
            centroid_y = centroids[i, 1]
            instances.append((centroid_x, centroid_y, area))
    
    return instances

def match_landmarks(pred_instances, true_instances, max_distance_pixels=50):
    """Match predicted landmarks to ground truth using Hungarian algorithm"""
    if len(pred_instances) == 0 or len(true_instances) == 0:
        return []
    
    # Create distance matrix
    distances = []
    for pred_x, pred_y, _ in pred_instances:
        row = []
        for true_x, true_y, _ in true_instances:
            dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            row.append(dist if dist <= max_distance_pixels else 999.0)
        distances.append(row)
    
    distances = np.array(distances)
    
    # Use simple greedy matching
    matches = []
    used_true = set()
    used_pred = set()
    
    # Sort by distance and greedily match
    pred_indices, true_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
    
    for pred_idx, true_idx in zip(pred_indices, true_indices):
        if pred_idx not in used_pred and true_idx not in used_true:
            if distances[pred_idx, true_idx] < 999.0:  # Valid match
                matches.append((pred_idx, true_idx, distances[pred_idx, true_idx]))
                used_pred.add(pred_idx)
                used_true.add(true_idx)
    
    return matches

def calculate_multi_landmark_metrics(outputs, targets, num_classes, pixel_spacing=0.0187, confidence_threshold=0.4):
    """
    Calculate metrics for multiple landmarks of each type per image
     
    Args:
        pixel_spacing: Calibrated from 31.64mm / 1692 pixels = 0.0187 mm/pixel
    """
    # Get class probabilities and predictions
    probabilities = torch.softmax(outputs, dim=1)  # [B, C, H, W]
    predicted = torch.argmax(outputs, dim=1)       # [B, H, W]
    
    batch_size = outputs.shape[0]
    all_metrics = []
    
    for batch_idx in range(batch_size):
        batch_metrics = {}
        
        for class_idx in range(1, num_classes):  # Skip background
            # Get masks for this class
            pred_mask = (predicted[batch_idx] == class_idx)
            true_mask = (targets[batch_idx] == class_idx)
            
            # Find individual landmark instances
            pred_instances = find_landmark_instances(pred_mask)
            true_instances = find_landmark_instances(true_mask)
            
            # Filter predictions by confidence
            confident_pred_instances = []
            for i, (x, y, area) in enumerate(pred_instances):
                # Get average confidence in this region
                region_mask = torch.zeros_like(pred_mask)
                # Create small region around centroid for confidence check
                x_int, y_int = int(x), int(y)
                radius = 3
                x_min, x_max = max(0, x_int-radius), min(pred_mask.shape[1], x_int+radius)
                y_min, y_max = max(0, y_int-radius), min(pred_mask.shape[0], y_int+radius)
                region_mask[y_min:y_max, x_min:x_max] = 1
                
                if region_mask.sum() > 0:
                    avg_confidence = probabilities[batch_idx, class_idx][region_mask].mean().item()
                    if avg_confidence > confidence_threshold:
                        confident_pred_instances.append((x, y, area, avg_confidence))
            
            # Match landmarks
            if confident_pred_instances and true_instances:
                # Convert confident predictions back to (x, y, area) format for matching
                confident_pred_simple = [(x, y, area) for x, y, area, conf in confident_pred_instances]
                matches = match_landmarks(confident_pred_simple, true_instances)
                
                # Calculate distances for matches
                distances = []
                confidences = []
                accurate_1mm = []  # ✅ Added 1mm threshold for clinical accuracy
                accurate_2mm = []
                accurate_5mm = []
                
                for pred_idx, true_idx, distance_pixels in matches:
                    distance_mm = distance_pixels * pixel_spacing  # ✅ Now uses correct 0.0187 mm/pixel
                    distances.append(distance_mm)
                    confidences.append(confident_pred_instances[pred_idx][3])  # confidence
                    accurate_1mm.append(1.0 if distance_mm <= 1.0 else 0.0)  # ✅ Clinical accuracy
                    accurate_2mm.append(1.0 if distance_mm <= 2.0 else 0.0)
                    accurate_5mm.append(1.0 if distance_mm <= 5.0 else 0.0)
                
                # Store metrics for this class
                if distances:
                    batch_metrics[f'distances_class_{class_idx}'] = distances
                    batch_metrics[f'confidences_class_{class_idx}'] = confidences
                    batch_metrics[f'accurate_1mm_class_{class_idx}'] = accurate_1mm  # ✅ Added
                    batch_metrics[f'accurate_2mm_class_{class_idx}'] = accurate_2mm
                    batch_metrics[f'accurate_5mm_class_{class_idx}'] = accurate_5mm
                    batch_metrics[f'matched_count_class_{class_idx}'] = len(matches)
                else:
                    batch_metrics[f'distances_class_{class_idx}'] = []
                    batch_metrics[f'confidences_class_{class_idx}'] = []
                    batch_metrics[f'accurate_1mm_class_{class_idx}'] = []  # ✅ Added
                    batch_metrics[f'accurate_2mm_class_{class_idx}'] = []
                    batch_metrics[f'accurate_5mm_class_{class_idx}'] = []
                    batch_metrics[f'matched_count_class_{class_idx}'] = 0
            else:
                batch_metrics[f'distances_class_{class_idx}'] = []
                batch_metrics[f'confidences_class_{class_idx}'] = []
                batch_metrics[f'accurate_1mm_class_{class_idx}'] = []  # ✅ Added
                batch_metrics[f'accurate_2mm_class_{class_idx}'] = []
                batch_metrics[f'accurate_5mm_class_{class_idx}'] = []
                batch_metrics[f'matched_count_class_{class_idx}'] = 0
            
            # Store counts
            batch_metrics[f'pred_count_class_{class_idx}'] = len(confident_pred_instances)
            batch_metrics[f'true_count_class_{class_idx}'] = len(true_instances)
        
        all_metrics.append(batch_metrics)
    
    # Aggregate across batch
    aggregated_metrics = {}
    
    for class_idx in range(1, num_classes):
        # Collect all distances and accuracies for this class across batch
        all_distances = []
        all_confidences = []
        all_accurate_1mm = []  # ✅ Added
        all_accurate_2mm = []
        all_accurate_5mm = []
        total_matched = 0
        total_pred = 0
        total_true = 0
        
        for batch_metrics in all_metrics:
            all_distances.extend(batch_metrics.get(f'distances_class_{class_idx}', []))
            all_confidences.extend(batch_metrics.get(f'confidences_class_{class_idx}', []))
            all_accurate_1mm.extend(batch_metrics.get(f'accurate_1mm_class_{class_idx}', []))  # ✅ Added
            all_accurate_2mm.extend(batch_metrics.get(f'accurate_2mm_class_{class_idx}', []))
            all_accurate_5mm.extend(batch_metrics.get(f'accurate_5mm_class_{class_idx}', []))
            total_matched += batch_metrics.get(f'matched_count_class_{class_idx}', 0)
            total_pred += batch_metrics.get(f'pred_count_class_{class_idx}', 0)
            total_true += batch_metrics.get(f'true_count_class_{class_idx}', 0)
        
        # Calculate aggregated metrics
        if all_distances:
            aggregated_metrics[f'mean_distance_mm_class_{class_idx}'] = np.mean(all_distances)
            aggregated_metrics[f'std_distance_mm_class_{class_idx}'] = np.std(all_distances)
            aggregated_metrics[f'mean_confidence_class_{class_idx}'] = np.mean(all_confidences)
            aggregated_metrics[f'accuracy_1mm_class_{class_idx}'] = np.mean(all_accurate_1mm)  # ✅ Added
            aggregated_metrics[f'accuracy_2mm_class_{class_idx}'] = np.mean(all_accurate_2mm)
            aggregated_metrics[f'accuracy_5mm_class_{class_idx}'] = np.mean(all_accurate_5mm)
        else:
            aggregated_metrics[f'mean_distance_mm_class_{class_idx}'] = 999.0
            aggregated_metrics[f'std_distance_mm_class_{class_idx}'] = 0.0
            aggregated_metrics[f'mean_confidence_class_{class_idx}'] = 0.0
            aggregated_metrics[f'accuracy_1mm_class_{class_idx}'] = 0.0  # ✅ Added
            aggregated_metrics[f'accuracy_2mm_class_{class_idx}'] = 0.0
            aggregated_metrics[f'accuracy_5mm_class_{class_idx}'] = 0.0
        
        # Detection metrics
        aggregated_metrics[f'detection_rate_class_{class_idx}'] = total_matched / total_true if total_true > 0 else 0.0
        aggregated_metrics[f'precision_class_{class_idx}'] = total_matched / total_pred if total_pred > 0 else 0.0
        aggregated_metrics[f'total_landmarks_class_{class_idx}'] = total_true
        aggregated_metrics[f'detected_landmarks_class_{class_idx}'] = total_matched
    
    # Overall metrics
    all_class_distances = []
    all_class_1mm = []  # ✅ Added
    all_class_2mm = []
    all_class_5mm = []
    
    for class_idx in range(1, num_classes):
        if aggregated_metrics[f'mean_distance_mm_class_{class_idx}'] < 999.0:
            all_class_distances.append(aggregated_metrics[f'mean_distance_mm_class_{class_idx}'])
            all_class_1mm.append(aggregated_metrics[f'accuracy_1mm_class_{class_idx}'])  # ✅ Added
            all_class_2mm.append(aggregated_metrics[f'accuracy_2mm_class_{class_idx}'])
            all_class_5mm.append(aggregated_metrics[f'accuracy_5mm_class_{class_idx}'])
    
    if all_class_distances:
        aggregated_metrics['overall_mean_distance_mm'] = np.mean(all_class_distances)
        aggregated_metrics['overall_accuracy_1mm'] = np.mean(all_class_1mm)  # ✅ Added
        aggregated_metrics['overall_accuracy_2mm'] = np.mean(all_class_2mm)
        aggregated_metrics['overall_accuracy_5mm'] = np.mean(all_class_5mm)
    else:
        aggregated_metrics['overall_mean_distance_mm'] = 999.0
        aggregated_metrics['overall_accuracy_1mm'] = 0.0  # ✅ Added
        aggregated_metrics['overall_accuracy_2mm'] = 0.0
        aggregated_metrics['overall_accuracy_5mm'] = 0.0
    
    return aggregated_metrics

def calculate_comprehensive_metrics(outputs, targets, num_classes, pixel_spacing=0.0187, confidence_threshold=0.4):
    """
    Calculate both segmentation metrics (all predictions) and distance metrics (confident only)
    
    Args:
        pixel_spacing: Calibrated from 31.64mm / 1692 pixels = 0.0187 mm/pixel
    """
    predicted = torch.argmax(outputs, dim=1)
    
    metrics = {}
    
    # 1. Segmentation metrics (ALL predictions - no confidence filtering)
    ious = []
    dices = []
    
    for class_idx in range(num_classes):
        pred_mask = (predicted == class_idx)
        true_mask = (targets == class_idx)
        
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        else:
            iou = torch.tensor(0.0)
            ious.append(0.0)
        
        if pred_mask.sum() + true_mask.sum() > 0:
            dice = (2 * intersection) / (pred_mask.sum() + true_mask.sum())
            dices.append(dice.item())
        else:
            dice = torch.tensor(0.0)
            dices.append(0.0)
        
        metrics[f'IoU_class_{class_idx}'] = iou.item()
        metrics[f'Dice_class_{class_idx}'] = dice.item()
    
    # Mean segmentation metrics (excluding background)
    metrics['mean_IoU'] = np.mean(ious[1:]) if len(ious) > 1 else 0.0
    metrics['mean_Dice'] = np.mean(dices[1:]) if len(dices) > 1 else 0.0
    
    # 2. Multi-landmark distance metrics (CONFIDENT predictions only) - ✅ Now uses correct pixel spacing
    distance_metrics = calculate_multi_landmark_metrics(outputs, targets, num_classes, pixel_spacing, confidence_threshold)
    metrics.update(distance_metrics)
    
    return metrics

def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, num_classes):
    """Train for one epoch with comprehensive debugging"""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    # Track predictions per class
    prediction_distribution = torch.zeros(num_classes, device=device)
    target_distribution = torch.zeros(num_classes, device=device)
    
    for i, (images, masks) in enumerate(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # ===== DEBUGGING: Check if loss makes sense =====
        if i == 0:  # First batch
            print(f"DEBUG: Raw loss value: {loss.item()}")
            print(f"DEBUG: Output shape: {outputs.shape}")
            print(f"DEBUG: Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"DEBUG: Mask shape: {masks.shape}")
            print(f"DEBUG: Mask unique values: {torch.unique(masks).cpu().tolist()}")
        
        # Calculate pixel accuracy and track distributions
        predicted = torch.argmax(outputs, dim=1)
        correct_pixels += (predicted == masks).sum().item()
        total_pixels += masks.numel()
        
        # Track class distributions
        for class_id in range(num_classes):
            pred_count = (predicted == class_id).sum()
            target_count = (masks == class_id).sum()
            prediction_distribution[class_id] += pred_count
            target_distribution[class_id] += target_count
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Minimal logging every 50 steps
        if i % 50 == 0:
            current_acc = correct_pixels / total_pixels * 100
            logger.info(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], "
                       f"Loss: {loss.item():.4f}, Pixel Acc: {current_acc:.2f}%")
    
    epoch_acc = correct_pixels / total_pixels * 100
    avg_loss = total_loss / len(data_loader)
    
    # Final epoch statistics
    final_pred_dist = prediction_distribution / prediction_distribution.sum() * 100
    final_target_dist = target_distribution / target_distribution.sum() * 100
    
    logger.info(f"EPOCH {epoch} SUMMARY:")
    logger.info(f"  Final prediction distribution: {final_pred_dist.cpu().tolist()}")
    logger.info(f"  Final target distribution: {final_target_dist.cpu().tolist()}")
    logger.info(f"  Background pred %: {final_pred_dist[0].item():.1f}%")
    logger.info(f"  Background target %: {final_target_dist[0].item():.1f}%")
    
    # Log epoch-level metrics for SageMaker
    print(f"[{epoch}] train_loss: {avg_loss:.4f}")
    print(f"[{epoch}] train_pixel_accuracy: {epoch_acc:.4f}")
    print(f"[{epoch}] background_prediction_pct: {final_pred_dist[0].item():.4f}")
    
    return avg_loss, epoch_acc

def evaluate(model, data_loader, device, num_classes, criterion):
    """Evaluate the model with CORRECTED pixel spacing"""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate pixel accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()
            
            # ✅ FIXED: Use correct pixel spacing
            batch_metrics = calculate_comprehensive_metrics(outputs, masks, num_classes, pixel_spacing=0.0187)
            all_metrics.append(batch_metrics)
    
    # Average metrics across all batches
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_loss = total_loss / len(data_loader)
    pixel_acc = correct_pixels / total_pixels * 100
    
    return avg_loss, pixel_acc, avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Train Pure TransUNet for landmark segmentation')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--class_group', type=str, default='all',
                        choices=['all', 'conditions', 'surfaces', 'bone-loss', 'teeth'],
                        help='Subset of classes to train on')
    # X-ray type filtering - supports comma-separated values
    parser.add_argument('--xray_type', type=str, default=None,
                        help='X-ray type filtering (comma-separated for multiple: bitewing,periapical,panoramic)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=1024)
    # Transformer architecture parameters
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--embed_dim', type=int, default=768, help='Transformer embedding dimension')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--max_images', type=int, default=None, help='Limit dataset size for testing')
    parser.add_argument('--train_annotations', type=str, default='train.json', 
                       help='Name of the training annotations JSON file')
    parser.add_argument('--val_annotations', type=str, default='val.json', 
                       help='Name of the validation annotations JSON file')
    # Loss function configuration
    parser.add_argument('--loss_function', type=str, default='dicefocal',
                        choices=['crossentropy', 'dicefocal', 'focal'],
                        help='Loss function to use: crossentropy (simple), dicefocal (recommended), or focal')
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to use class weights for loss functions')
    parser.add_argument('--background_weight', type=float, default=0.1,
                        help='Weight for background class (used with DiceFocalLoss)')
    parser.add_argument('--dice_weight', type=float, default=0.6,
                        help='Weight for Dice loss component in DiceFocalLoss')
    parser.add_argument('--focal_weight', type=float, default=0.4,
                        help='Weight for Focal loss component in DiceFocalLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal loss')
    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Training on {args.class_group} class group")
    if args.xray_type:
        logger.info(f"X-ray type filter: {args.xray_type}")
    
    # Get class names for the specified class group
    class_names = get_class_names_for_class_group(args.class_group)
    num_landmark_classes = len(class_names)
    
    logger.info(f"Number of landmark classes for {args.class_group}: {num_landmark_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Create model - add 1 for background class
    model = TransUNet(
        num_classes=num_landmark_classes + 1,  # Add background class
        img_size=args.image_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Load separate train and validation datasets
    train_annotations_file = os.path.join(args.train, args.train_annotations)
    val_annotations_file = os.path.join(args.train, args.val_annotations)
    images_dir = os.path.join(args.train, 'images')
    
    logger.info(f"📄 Loading train annotations: {train_annotations_file}")
    logger.info(f"📄 Loading val annotations: {val_annotations_file}")
    logger.info(f"🖼️ Images directory: {images_dir}")
    
    # Enhanced transforms with research-backed augmentation for dental imaging (TRAINING ONLY)
    train_transforms = T.Compose([
        T.Resize((1024, 1024)),  # Research-recommended image size
        T.ToTensor(),
        # Conservative augmentation for dental imaging
        T.ColorJitter(
            brightness=0.2,      # ±20% brightness variation
            contrast=0.2,        # ±20% contrast variation  
            saturation=0.35,     # Conservative saturation (0.3-0.4 range)
            hue=0.025           # Conservative hue (0.02-0.03 range)
        ),
        # Random rotation within anatomically safe range
        T.RandomRotation(degrees=10, fill=0),  # 5-15° rotation range
        # Conservative geometric transforms
        T.RandomAffine(
            degrees=0,           # Rotation handled above
            translate=(0.075, 0.075),  # 0.05-0.1 translation range
            scale=(0.9, 1.1),    # 0.8-1.2 scaling range (conservative)
            fill=0
        ),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (NO augmentation)
    val_transforms = T.Compose([
        T.Resize((1024, 1024)),  # Research-recommended image size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create separate train and validation datasets
    train_dataset = SegmentationDataset(
        train_annotations_file, 
        images_dir, 
        train_transforms, 
        args.image_size,
        class_group=args.class_group,
        xray_type=args.xray_type
    )
    
    val_dataset = SegmentationDataset(
        val_annotations_file, 
        images_dir, 
        val_transforms, 
        args.image_size,
        class_group=args.class_group,
        xray_type=args.xray_type
    )
    
    # Limit dataset size for testing if specified
    if args.max_images is not None:
        if args.max_images < len(train_dataset):
            logger.info(f"🧪 TESTING MODE: Limiting train dataset to {args.max_images} images (original: {len(train_dataset)})")
            train_indices = list(range(len(train_dataset)))
            limited_train_indices = train_indices[:args.max_images]
            train_dataset = torch.utils.data.Subset(train_dataset, limited_train_indices)
            logger.info(f"Train dataset size after limiting: {len(train_dataset)}")
        
        # Also limit validation dataset proportionally
        val_limit = max(1, args.max_images // 4)  # Use 1/4 of max_images for validation
        if val_limit < len(val_dataset):
            logger.info(f"🧪 TESTING MODE: Limiting val dataset to {val_limit} images (original: {len(val_dataset)})")
            val_indices = list(range(len(val_dataset)))
            limited_val_indices = val_indices[:val_limit]
            val_dataset = torch.utils.data.Subset(val_dataset, limited_val_indices)
            logger.info(f"Val dataset size after limiting: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Configure loss function based on command line arguments
    logger.info(f"🔧 Configuring {args.loss_function} loss function...")
    
    if args.loss_function == 'dicefocal':
        # DiceFocalLoss with annotation-based class weights (recommended)
        logger.info("🔍 Calculating class weights from actual training annotations...")
        class_weights = calculate_class_weights_from_annotations(
            train_dataset, 
            class_names, 
            background_weight=args.background_weight
        )
        class_weights = class_weights.to(device)
        
        from losses import DiceFocalLoss
        criterion = DiceFocalLoss(
            alpha=class_weights if args.use_class_weights else None,
            gamma=args.focal_gamma,
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            ignore_index=-100
        )
        logger.info(f"✅ Using DiceFocalLoss (dice_weight={args.dice_weight}, focal_weight={args.focal_weight}, gamma={args.focal_gamma})")
        
    elif args.loss_function == 'focal':
        # Focal Loss only
        if args.use_class_weights:
            logger.info("🔍 Calculating class weights from actual training annotations...")
            class_weights = calculate_class_weights_from_annotations(
                train_dataset, 
                class_names, 
                background_weight=args.background_weight
            )
            class_weights = class_weights.to(device)
        else:
            class_weights = None
            
        from losses import FocalLoss
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            ignore_index=-100
        )
        logger.info(f"✅ Using FocalLoss (gamma={args.focal_gamma})")
        
    elif args.loss_function == 'crossentropy':
        # Simple CrossEntropyLoss (baseline)
        if args.use_class_weights:
            logger.info("🔍 Analyzing class distribution for simple weighting...")
            class_distribution = analyze_class_distribution(train_loader, num_landmark_classes + 1)
            
            # Simple inverse frequency weighting
            total_samples = sum(class_distribution)
            class_weights = []
            for count in class_distribution:
                if count > 0:
                    # Gentle weighting with square root for stability
                    weight = np.sqrt(total_samples / count)
                else:
                    weight = 1.0
                class_weights.append(weight)
            
            # Cap maximum weight to prevent extreme values
            max_weight = max(class_weights)
            if max_weight > 10.0:
                scaling_factor = 10.0 / max_weight
                class_weights = [w * scaling_factor for w in class_weights]
            
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            logger.info(f"Class weights: {class_weights.cpu().tolist()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            logger.info("Using unweighted CrossEntropyLoss")
        
        logger.info(f"✅ Using CrossEntropyLoss (weighted={args.use_class_weights})")
    
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    criterion = criterion.to(device)
    
    if args.use_class_weights:
        logger.info(f"✅ Using class weights with {args.loss_function} loss")
    else:
        logger.info(f"✅ Using unweighted {args.loss_function} loss")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        # Train with improved loss
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch, criterion, num_landmark_classes + 1)
        
        # Validate with detailed metrics
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, device, num_landmark_classes + 1, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Log comprehensive metrics
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"            Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log class-specific metrics with multiple landmarks
        if val_metrics:
            logger.info(f"            Mean IoU: {val_metrics.get('mean_IoU', 0):.4f}, Mean Dice: {val_metrics.get('mean_Dice', 0):.4f}")
            logger.info(f"            Overall Distance: {val_metrics.get('overall_mean_distance_mm', 0):.2f}mm, 2mm Acc: {val_metrics.get('overall_accuracy_2mm', 0):.3f}")
            
            # Log per-class metrics for landmark classes (skip background)
            seg_class_names = ["background"] + class_names
            for i in range(1, len(seg_class_names)):  # Skip background
                if i < len(seg_class_names):
                    iou = val_metrics.get(f'IoU_class_{i}', 0)
                    dice = val_metrics.get(f'Dice_class_{i}', 0)
                    dist_mm = val_metrics.get(f'mean_distance_mm_class_{i}', 0)
                    dist_std = val_metrics.get(f'std_distance_mm_class_{i}', 0)
                    confidence = val_metrics.get(f'mean_confidence_class_{i}', 0)
                    acc_2mm = val_metrics.get(f'accuracy_2mm_class_{i}', 0)
                    detection_rate = val_metrics.get(f'detection_rate_class_{i}', 0)
                    total_landmarks = val_metrics.get(f'total_landmarks_class_{i}', 0)
                    detected = val_metrics.get(f'detected_landmarks_class_{i}', 0)
                    
                    logger.info(f"            {seg_class_names[i]}: IoU={iou:.3f}, Dist={dist_mm:.1f}±{dist_std:.1f}mm, DetRate={detection_rate:.2f} ({detected}/{total_landmarks}), 2mmAcc={acc_2mm:.2f}")
        
        # Log for SageMaker monitoring
        print(f"[{epoch}] validation_loss: {val_loss:.4f}")
        print(f"[{epoch}] validation_pixel_accuracy: {val_acc:.4f}")
        print(f"[{epoch}] mean_iou: {val_metrics.get('mean_IoU', 0):.4f}")
        print(f"[{epoch}] mean_dice: {val_metrics.get('mean_Dice', 0):.4f}")
        print(f"[{epoch}] overall_mean_distance_mm: {val_metrics.get('overall_mean_distance_mm', 0):.4f}")
        print(f"[{epoch}] overall_accuracy_2mm: {val_metrics.get('overall_accuracy_2mm', 0):.4f}")
        print(f"[{epoch}] overall_accuracy_5mm: {val_metrics.get('overall_accuracy_5mm', 0):.4f}")
        
        # Log individual class metrics for SageMaker
        for i in range(1, num_landmark_classes + 1):
            iou = val_metrics.get(f'IoU_class_{i}', 0)
            dice = val_metrics.get(f'Dice_class_{i}', 0)
            dist_mm = val_metrics.get(f'mean_distance_mm_class_{i}', 0)
            confidence = val_metrics.get(f'mean_confidence_class_{i}', 0)
            acc_2mm = val_metrics.get(f'accuracy_2mm_class_{i}', 0)
            acc_5mm = val_metrics.get(f'accuracy_5mm_class_{i}', 0)
            detection_rate = val_metrics.get(f'detection_rate_class_{i}', 0)
            precision = val_metrics.get(f'precision_class_{i}', 0)
            
            print(f"[{epoch}] class_{i}_iou: {iou:.4f}")
            print(f"[{epoch}] class_{i}_dice: {dice:.4f}")
            print(f"[{epoch}] class_{i}_mean_distance_mm: {dist_mm:.4f}")
            print(f"[{epoch}] class_{i}_mean_confidence: {confidence:.4f}")
            print(f"[{epoch}] class_{i}_accuracy_2mm: {acc_2mm:.4f}")
            print(f"[{epoch}] class_{i}_accuracy_5mm: {acc_5mm:.4f}")
            print(f"[{epoch}] class_{i}_detection_rate: {detection_rate:.4f}")
            print(f"[{epoch}] class_{i}_precision: {precision:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_metrics': val_metrics,
                'class_names': class_names,
                'class_group': args.class_group,
            }, os.path.join(args.model_dir, 'best_model.pth'))
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()