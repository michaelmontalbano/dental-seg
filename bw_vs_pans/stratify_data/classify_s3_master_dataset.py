#!/usr/bin/env python3
"""
Classify all images in s3://codentist-general/datasets/master
and update annotations with X-ray type classification
"""

import argparse
import boto3
import json
import os
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DentalClassifier(torch.nn.Module):
    """Recreate the model architecture for inference"""
    
    def __init__(self, model_name='resnet50', num_classes=3):
        super(DentalClassifier, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(num_features, num_classes)
        elif model_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=False)
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = torch.nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)

class S3DatasetClassifier:
    def __init__(self, model_path, device='cuda', batch_size=16):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.s3_client = boto3.client('s3')
        
        # Load model
        self.model, self.metadata = self._load_model(model_path)
        self.classes = self.metadata['class_names']
        self.image_size = self.metadata.get('image_size', 224)
        
        # Setup transforms
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Loaded model: {self.metadata['model_name']}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if model_path.startswith('s3://'):
            # Download from S3
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                s3_parts = model_path.replace('s3://', '').split('/')
                bucket = s3_parts[0]
                key = '/'.join(s3_parts[1:])
                self.s3_client.download_file(bucket, key, tmp.name)
                model_path = tmp.name
        
        # Load with weights_only=False to handle newer PyTorch models
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=False: {e}")
            logger.info("Trying with weights_only=True...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        model = DentalClassifier(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint
    
    def _download_and_classify_image(self, bucket, key):
        """Download image from S3 and classify it"""
        try:
            # Download image to memory
            with tempfile.NamedTemporaryFile() as tmp:
                self.s3_client.download_file(bucket, key, tmp.name)
                
                # Load and preprocess image
                image = Image.open(tmp.name).convert('RGB')
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Classify
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    prediction = torch.argmax(outputs, dim=1)
                
                predicted_class = self.classes[prediction.item()]
                confidence = probabilities[0][prediction].item()
                
                # Get all probabilities
                all_probs = {
                    class_name: probabilities[0][i].item() 
                    for i, class_name in enumerate(self.classes)
                }
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': all_probs
                }
                
        except Exception as e:
            logger.warning(f"Failed to classify {key}: {e}")
            return None
    
    def get_s3_images(self, bucket, prefix):
        """Get all image files from S3 bucket/prefix"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = []
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        # Skip if it's in a subdirectory we don't want
                        if not any(skip in key for skip in ['thumbs', 'cache', '.DS_Store']):
                            images.append(key)
        
        logger.info(f"Found {len(images)} images in s3://{bucket}/{prefix}")
        return images
    
    def load_existing_annotations(self, bucket, annotations_key):
        """Load existing annotations from S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=annotations_key)
            annotations = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded existing annotations: {len(annotations)} entries")
            return annotations
        except self.s3_client.exceptions.NoSuchKey:
            logger.info("No existing annotations found, starting fresh")
            return {}
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return {}
    
    def classify_s3_dataset(self, bucket, prefix, annotations_file='annotations_image_mode.json', 
                           update_existing=False, existing_annotations_file='annotations.json',
                           test_mode=False, test_output_file='annotations_test.json'):
        """Classify all images in S3 dataset and update annotations"""
        
        # Get all images
        image_keys = self.get_s3_images(bucket, prefix)
        if not image_keys:
            logger.error("No images found!")
            return
        
        # Load existing annotations if updating
        if update_existing:
            existing_annotations = self.load_existing_annotations(bucket, f"{prefix}/{existing_annotations_file}")
            base_annotations = existing_annotations.copy()
        else:
            base_annotations = {}
        
        # Classification results
        classification_results = {}
        stats = defaultdict(int)
        
        # Process images
        logger.info(f"Starting classification of {len(image_keys)} images...")
        
        for image_key in tqdm(image_keys, desc="Classifying images"):
            filename = Path(image_key).name
            
            # Classify image
            result = self._download_and_classify_image(bucket, image_key)
            
            if result:
                classification_results[filename] = {
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'classification_date': datetime.now().isoformat(),
                    's3_key': image_key
                }
                
                stats[result['predicted_class']] += 1
                stats['total'] += 1
            else:
                stats['failed'] += 1
        
        # Update or create annotations
        if update_existing:
            # Add classification info to existing annotations
            for filename, classification in classification_results.items():
                if filename in base_annotations:
                    base_annotations[filename].update({
                        'xray_type': classification['predicted_class'],
                        'xray_confidence': classification['confidence'],
                        'xray_probabilities': classification['probabilities'],
                        'classification_date': classification['classification_date']
                    })
                else:
                    # Create new entry
                    base_annotations[filename] = {
                        'xray_type': classification['predicted_class'],
                        'xray_confidence': classification['confidence'],
                        'xray_probabilities': classification['probabilities'],
                        'classification_date': classification['classification_date'],
                        's3_key': classification['s3_key']
                    }
            
            final_annotations = base_annotations
            
            # Choose output file based on test mode
            if test_mode:
                output_file = test_output_file
                logger.info(f"🧪 TEST MODE: Saving to {test_output_file} instead of {existing_annotations_file}")
            else:
                output_file = existing_annotations_file
        else:
            # Create separate classification file
            final_annotations = classification_results
            output_file = annotations_file
        
        # Save annotations back to S3
        annotations_key = f"{prefix}/{output_file}"
        annotations_json = json.dumps(final_annotations, indent=2)
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=annotations_key,
            Body=annotations_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        # Save summary stats with test mode indication
        summary = {
            'classification_date': datetime.now().isoformat(),
            'test_mode': test_mode,
            'output_file': output_file,
            'original_preserved': test_mode,
            'total_images': len(image_keys),
            'successfully_classified': stats['total'],
            'failed_classifications': stats['failed'],
            'class_distribution': {
                'bitewing': stats['bitewing'],
                'periapical': stats['periapical'],
                'panoramic': stats['panoramic']
            },
            'model_info': {
                'model_name': self.metadata['model_name'],
                'accuracy': self.metadata.get('final_accuracy', 'unknown')
            }
        }
        
        summary_suffix = '_test' if test_mode else ''
        summary_key = f"{prefix}/classification_summary{summary_suffix}.json"
        self.s3_client.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        # Print results
        self._print_results(stats, annotations_key, summary_key, test_mode)
        
        return final_annotations
    
    def _print_results(self, stats, annotations_key, summary_key, test_mode=False):
        """Print classification results"""
        mode_text = "🧪 TEST MODE" if test_mode else "🦷 MASTER DATASET CLASSIFICATION COMPLETE"
        logger.info("=" * 80)
        logger.info(mode_text)
        logger.info("=" * 80)
        
        if test_mode:
            logger.info("✅ ORIGINAL ANNOTATIONS.JSON PRESERVED")
            logger.info("🔍 Review test results before running full classification")
            logger.info("")
        
        logger.info(f"Total images processed: {stats['total'] + stats['failed']}")
        logger.info(f"Successfully classified: {stats['total']}")
        logger.info(f"Failed classifications: {stats['failed']}")
        logger.info("")
        logger.info("Class Distribution:")
        for class_name in ['bitewing', 'periapical', 'panoramic']:
            count = stats[class_name]
            percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"  {class_name.capitalize()}: {count} ({percentage:.1f}%)")
        logger.info("")
        logger.info(f"📝 Annotations saved to: s3://codentist-general/{annotations_key}")
        logger.info(f"📊 Summary saved to: s3://codentist-general/{summary_key}")
        
        if test_mode:
            logger.info("")
            logger.info("🔄 To apply to original annotations.json, run without --test-mode")
        
        logger.info("=" * 80)

def download_model_from_sagemaker(job_name):
    """Download trained model from SageMaker job"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        job_details = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        model_s3_path = job_details['ModelArtifacts']['S3ModelArtifacts']
        
        # Download and extract
        s3_client = boto3.client('s3')
        s3_parts = model_s3_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        key = '/'.join(s3_parts[1:])
        
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            
            # Extract tar.gz
            import tarfile
            extract_dir = tempfile.mkdtemp()
            with tarfile.open(tmp.name, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            model_path = os.path.join(extract_dir, 'model.pth')
            if os.path.exists(model_path):
                logger.info(f"Downloaded model from SageMaker job: {job_name}")
                return model_path
            else:
                raise FileNotFoundError("model.pth not found in SageMaker artifacts")
                
    except Exception as e:
        logger.error(f"Failed to download model from SageMaker: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Classify master dataset and update annotations')
    
    # Model source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sagemaker-job', help='SageMaker training job name')
    group.add_argument('--model-path', help='Path to model file (local or s3://)')
    
    # S3 configuration
    parser.add_argument('--bucket', default='codentist-general', help='S3 bucket')
    parser.add_argument('--prefix', default='datasets/master', help='S3 prefix')
    
    # Annotation options
    parser.add_argument('--update-existing', action='store_true',
                       help='Update existing annotations.json with classification info')
    parser.add_argument('--annotations-file', default='annotations_image_mode.json',
                       help='Name for new annotations file')
    parser.add_argument('--existing-annotations', default='annotations.json',
                       help='Name of existing annotations file to update')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    # Test mode options
    parser.add_argument('--test-mode', action='store_true',
                       help='TEST MODE: Save to test file instead of modifying original')
    parser.add_argument('--test-output-file', default='annotations_test.json',
                       help='Test output filename (only used with --test-mode)')
    
    args = parser.parse_args()
    
    # Get model path
    if args.sagemaker_job:
        model_path = download_model_from_sagemaker(args.sagemaker_job)
    else:
        model_path = args.model_path
    
    # Create classifier
    classifier = S3DatasetClassifier(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run classification
    results = classifier.classify_s3_dataset(
        bucket=args.bucket,
        prefix=args.prefix,
        annotations_file=args.annotations_file,
        update_existing=args.update_existing,
        existing_annotations_file=args.existing_annotations,
        test_mode=args.test_mode,
        test_output_file=args.test_output_file
    )
    
    print(f"\n🎉 Master dataset classification complete!")
    print(f"📊 Classified {len(results)} images")
    
    if args.test_mode:
        print(f"🧪 TEST MODE: Results saved to {args.test_output_file}")
        print(f"✅ Original {args.existing_annotations} was NOT modified")
        print(f"🔍 Review the test results, then run without --test-mode to apply")
    elif args.update_existing:
        print(f"✅ Updated existing annotations in {args.existing_annotations}")
    else:
        print(f"📝 Created new annotations file: {args.annotations_file}")

if __name__ == '__main__':
    main()
