import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import boto3
import tempfile

# Import model architectures
from train_hierarchical_model import HierarchicalClassifier, AttentionModule

class CompanyClassifier(nn.Module):
    """Standalone company classifier"""
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class ModelClassifier(nn.Module):
    """Standalone model classifier"""
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class ImplantInference:
    def __init__(self, model_type='hierarchical', device='cuda'):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.models = {}
        self.vocabs = {}
        
    def download_from_s3(self, s3_path: str, local_path: str):
        """Download file from S3"""
        s3 = boto3.client('s3')
        bucket_name = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        s3.download_file(bucket_name, key, local_path)
        print(f"📥 Downloaded {s3_path} to {local_path}")
        
    def load_model(self, checkpoint_path: str, model_type: str = 'hierarchical'):
        """Load model from checkpoint"""
        # Download if S3 path
        if checkpoint_path.startswith('s3://'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            self.download_from_s3(checkpoint_path, temp_file.name)
            checkpoint_path = temp_file.name
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if model_type == 'hierarchical':
            # Load hierarchical model
            company_vocab = checkpoint['company_vocab']
            model_vocab = checkpoint['model_vocab']
            args = checkpoint.get('args', {})
            
            model = HierarchicalClassifier(
                args.get('base_model_name', 'vit_base_patch16_224'),
                len(company_vocab),
                len(model_vocab)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models['hierarchical'] = model
            self.vocabs['company'] = {v: k for k, v in company_vocab.items()}
            self.vocabs['model'] = {v: k for k, v in model_vocab.items()}
            
        elif model_type == 'company':
            # Load company classifier
            vocab = checkpoint['vocab']
            args = checkpoint.get('args', {})
            
            model_size = args.get('model_size', 'base')
            model_map = {
                'tiny': 'vit_tiny_patch16_224',
                'small': 'vit_small_patch16_224',
                'base': 'vit_base_patch16_224',
                'large': 'vit_large_patch16_224'
            }
            
            model = CompanyClassifier(model_map[model_size], len(vocab))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models['company'] = model
            self.vocabs['company'] = {v: k for k, v in vocab.items()}
            
        elif model_type == 'model':
            # Load model classifier
            vocab = checkpoint['vocab']
            args = checkpoint.get('args', {})
            
            model_size = args.get('model_size', 'base')
            model_map = {
                'tiny': 'vit_tiny_patch16_224',
                'small': 'vit_small_patch16_224',
                'base': 'vit_base_patch16_224',
                'large': 'vit_large_patch16_224'
            }
            
            model = ModelClassifier(model_map[model_size], len(vocab))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models['model'] = model
            self.vocabs['model'] = {v: k for k, v in vocab.items()}
            
        print(f"✅ Loaded {model_type} model successfully")
        
    def predict_single(self, image_path: str, top_k: int = 5) -> Dict:
        """Predict company and model for a single image"""
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {'error': f'Failed to load image: {str(e)}'}
        
        with torch.no_grad():
            if self.model_type == 'hierarchical' and 'hierarchical' in self.models:
                # Hierarchical prediction
                model = self.models['hierarchical']
                company_logits, model_logits, company_confidence = model(image_tensor)
                
                # Get company predictions
                company_probs = F.softmax(company_logits, dim=1)
                company_top_k = torch.topk(company_probs, min(top_k, company_probs.size(1)), dim=1)
                
                company_predictions = []
                for i in range(company_top_k.values.size(1)):
                    idx = company_top_k.indices[0, i].item()
                    prob = company_top_k.values[0, i].item()
                    company_predictions.append({
                        'company': self.vocabs['company'][idx],
                        'confidence': prob
                    })
                
                # Get model predictions
                model_probs = F.softmax(model_logits, dim=1)
                model_top_k = torch.topk(model_probs, min(top_k, model_probs.size(1)), dim=1)
                
                model_predictions = []
                for i in range(model_top_k.values.size(1)):
                    idx = model_top_k.indices[0, i].item()
                    prob = model_top_k.values[0, i].item()
                    model_predictions.append({
                        'model': self.vocabs['model'][idx],
                        'confidence': prob
                    })
                
                return {
                    'image': image_path,
                    'company_predictions': company_predictions,
                    'model_predictions': model_predictions,
                    'company_confidence': company_confidence.item()
                }
                
            else:
                # Separate predictions
                results = {'image': image_path}
                
                if 'company' in self.models:
                    model = self.models['company']
                    company_logits = model(image_tensor)
                    company_probs = F.softmax(company_logits, dim=1)
                    company_top_k = torch.topk(company_probs, min(top_k, company_probs.size(1)), dim=1)
                    
                    company_predictions = []
                    for i in range(company_top_k.values.size(1)):
                        idx = company_top_k.indices[0, i].item()
                        prob = company_top_k.values[0, i].item()
                        company_predictions.append({
                            'company': self.vocabs['company'][idx],
                            'confidence': prob
                        })
                    results['company_predictions'] = company_predictions
                
                if 'model' in self.models:
                    model = self.models['model']
                    model_logits = model(image_tensor)
                    model_probs = F.softmax(model_logits, dim=1)
                    model_top_k = torch.topk(model_probs, min(top_k, model_probs.size(1)), dim=1)
                    
                    model_predictions = []
                    for i in range(model_top_k.values.size(1)):
                        idx = model_top_k.indices[0, i].item()
                        prob = model_top_k.values[0, i].item()
                        model_predictions.append({
                            'model': self.vocabs['model'][idx],
                            'confidence': prob
                        })
                    results['model_predictions'] = model_predictions
                
                return results
    
    def predict_batch(self, input_source: str, output_path: str, top_k: int = 5):
        """Predict on batch of images from folder or JSON file"""
        results = []
        
        if os.path.isdir(input_source):
            # Process all images in folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            
            for root, dirs, files in os.walk(input_source):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            print(f"📸 Found {len(image_files)} images to process")
            
            for image_path in tqdm(image_files, desc="Processing images"):
                result = self.predict_single(image_path, top_k)
                results.append(result)
                
        elif input_source.endswith('.json'):
            # Process images from JSON file
            with open(input_source) as f:
                entries = json.load(f)
            
            print(f"📋 Processing {len(entries)} entries from JSON")
            
            for entry in tqdm(entries, desc="Processing images"):
                image_path = entry.get('image', entry.get('path'))
                if image_path:
                    result = self.predict_single(image_path, top_k)
                    # Add any additional metadata from JSON
                    result['metadata'] = {k: v for k, v in entry.items() if k not in ['image', 'path']}
                    results.append(result)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved results to {output_path}")
        
        # Print summary statistics
        if results:
            company_correct = model_correct = 0
            company_total = model_total = 0
            
            for result in results:
                if 'metadata' in result:
                    # Check company accuracy if ground truth available
                    if 'company' in result['metadata'] and 'company_predictions' in result:
                        company_total += 1
                        if result['company_predictions'][0]['company'] == result['metadata']['company']:
                            company_correct += 1
                    
                    # Check model accuracy if ground truth available
                    if 'model' in result['metadata'] and 'model_predictions' in result:
                        model_total += 1
                        if result['model_predictions'][0]['model'] == result['metadata']['model']:
                            model_correct += 1
            
            print(f"\n📊 Summary:")
            print(f"   Total images: {len(results)}")
            if company_total > 0:
                print(f"   Company accuracy: {company_correct/company_total:.2%} ({company_correct}/{company_total})")
            if model_total > 0:
                print(f"   Model accuracy: {model_correct/model_total:.2%} ({model_correct}/{model_total})")

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Company and Model Classification')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], required=True,
                        help='Inference mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Image path (single mode) or folder/json path (batch mode)')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output JSON file path (batch mode)')
    parser.add_argument('--model_type', type=str, default='hierarchical',
                        choices=['hierarchical', 'separate'],
                        help='Model type to use')
    parser.add_argument('--hierarchical_checkpoint', type=str,
                        help='Path to hierarchical model checkpoint')
    parser.add_argument('--company_checkpoint', type=str,
                        help='Path to company classifier checkpoint')
    parser.add_argument('--model_checkpoint', type=str,
                        help='Path to model classifier checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🔧 Implant Company & Model Inference")
    print(f"📊 Mode: {args.mode}")
    print(f"🧠 Model type: {args.model_type}")
    print(f"🖥️  Device: {args.device}")
    
    # Initialize inference engine
    inference = ImplantInference(model_type=args.model_type, device=args.device)
    
    # Load models
    if args.model_type == 'hierarchical':
        if not args.hierarchical_checkpoint:
            print("❌ Error: --hierarchical_checkpoint required for hierarchical model")
            return
        inference.load_model(args.hierarchical_checkpoint, 'hierarchical')
    else:
        if args.company_checkpoint:
            inference.load_model(args.company_checkpoint, 'company')
        if args.model_checkpoint:
            inference.load_model(args.model_checkpoint, 'model')
        
        if not args.company_checkpoint and not args.model_checkpoint:
            print("❌ Error: At least one of --company_checkpoint or --model_checkpoint required")
            return
    
    # Run inference
    if args.mode == 'single':
        result = inference.predict_single(args.input, args.top_k)
        
        print(f"\n📸 Results for: {args.input}")
        
        if 'company_predictions' in result:
            print(f"\n🏢 Company Predictions:")
            for i, pred in enumerate(result['company_predictions']):
                print(f"   {i+1}. {pred['company']} ({pred['confidence']:.2%})")
        
        if 'model_predictions' in result:
            print(f"\n🔧 Model Predictions:")
            for i, pred in enumerate(result['model_predictions']):
                print(f"   {i+1}. {pred['model']} ({pred['confidence']:.2%})")
        
        if 'company_confidence' in result:
            print(f"\n📊 Company confidence for model prediction: {result['company_confidence']:.2%}")
            
    else:  # batch mode
        inference.predict_batch(args.input, args.output, args.top_k)

if __name__ == '__main__':
    main()
