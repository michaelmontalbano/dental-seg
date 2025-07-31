#!/usr/bin/env python3
"""
Complete pseudo-labeling pipeline for dental implants
1. Find unlabeled images
2. Generate predictions with trained model
3. Convert to Label Studio format
"""

import subprocess
import argparse
import os
from pathlib import Path
import json
import boto3
from ultralytics import YOLO
import shutil

def download_trained_model(s3_model_path, local_model_path='./best.pt'):
    """Download trained model from S3"""
    print(f"📥 Downloading trained model from {s3_model_path}")
    
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    from urllib.parse import urlparse
    parsed = urlparse(s3_model_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    # Download model.tar.gz and extract
    s3_client.download_file(bucket, key, 'model.tar.gz')
    
    # Extract tar.gz
    import tarfile
    with tarfile.open('model.tar.gz', 'r:gz') as tar:
        tar.extractall('.')
    
    # Find best.pt in extracted files
    if os.path.exists('best.pt'):
        shutil.move('best.pt', local_model_path)
    else:
        # Sometimes it's in a subdirectory
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file == 'best.pt':
                    shutil.move(os.path.join(root, file), local_model_path)
                    break
    
    print(f"✅ Model downloaded to {local_model_path}")
    return local_model_path

def run_yolo_predictions(model_path, images_dir, output_dir, confidence=0.6):
    """Run YOLO predictions on unlabeled images"""
    print(f"🔮 Running YOLO predictions...")
    print(f"📁 Images: {images_dir}")
    print(f"🎯 Confidence: {confidence}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run predictions
    results = model.predict(
        source=images_dir,
        conf=confidence,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='predictions',
        exist_ok=True
    )
    
    predictions_dir = Path(output_dir) / 'predictions'
    labels_dir = predictions_dir / 'labels'
    
    print(f"✅ Predictions saved to {labels_dir}")
    
    # Count predictions
    if labels_dir.exists():
        label_files = list(labels_dir.glob('*.txt'))
        total_predictions = 0
        for label_file in label_files:
            with open(label_file, 'r') as f:
                total_predictions += len(f.readlines())
        
        print(f"📊 Generated {total_predictions} predictions across {len(label_files)} images")
    
    return str(labels_dir)

def convert_to_label_studio(predictions_dir, images_dir, classes_file, output_json='pseudo_predictions.json'):
    """Convert YOLO predictions to Label Studio format"""
    print(f"🔄 Converting predictions to Label Studio format...")
    
    # Use label-studio-converter
    cmd = [
        'label-studio-converter', 'import', 'yolo',
        '-i', str(Path(predictions_dir).parent),  # Parent dir containing labels/
        '-o', output_json,
        '--image-root-url', '/data/local-files/?d=unlabeled_images',
        '--image-dir', str(images_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Label Studio format saved to {output_json}")
        return output_json
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return None

def create_manual_label_studio_format(predictions_dir, images_dir, classes_file, output_json='pseudo_predictions.json'):
    """Manually create Label Studio format if converter fails"""
    print(f"🔧 Creating Label Studio format manually...")
    
    # Read classes
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    predictions_dir = Path(predictions_dir)
    images_dir = Path(images_dir)
    
    label_studio_tasks = []
    
    # Process each prediction file
    for label_file in predictions_dir.glob('*.txt'):
        image_name = f"{label_file.stem}.jpg"  # Assume .jpg
        
        # Check if corresponding image exists
        image_path = images_dir / image_name
        if not image_path.exists():
            image_name = f"{label_file.stem}.png"
            image_path = images_dir / image_name
            if not image_path.exists():
                continue
        
        # Read predictions
        predictions = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:  # class_id, x, y, w, h, confidence
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    confidence = float(parts[5])
                    
                    # Convert to Label Studio format (percentage)
                    prediction = {
                        "from_name": "label",
                        "to_name": "image", 
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x - w/2) * 100,  # Convert to top-left corner
                            "y": (y - h/2) * 100,
                            "width": w * 100,
                            "height": h * 100,
                            "rectanglelabels": [class_names[class_id]]
                        },
                        "score": confidence
                    }
                    predictions.append(prediction)
        
        if predictions:
            task = {
                "data": {"image": f"/data/local-files/?d=unlabeled_images/{image_name}"},
                "predictions": [{
                    "result": predictions,
                    "score": sum(p["score"] for p in predictions) / len(predictions)
                }]
            }
            label_studio_tasks.append(task)
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(label_studio_tasks, f, indent=2)
    
    print(f"✅ Created {len(label_studio_tasks)} Label Studio tasks in {output_json}")
    return output_json

def main():
    parser = argparse.ArgumentParser(description='Complete pseudo-labeling pipeline')
    
    # S3 paths
    parser.add_argument('--all-images-s3', type=str,
                        default='s3://codentist-general/datasets/implant_labeling/all_images/',
                        help='S3 path to all images')
    parser.add_argument('--labeled-images-s3', type=str,
                        default='s3://codentist-general/datasets/implant_labeling/images/',
                        help='S3 path to labeled images')
    parser.add_argument('--model-s3', type=str, required=True,
                        help='S3 path to trained model (model.tar.gz)')
    parser.add_argument('--classes-file', type=str, default='classes.txt',
                        help='Path to classes.txt file')
    
    # Local paths
    parser.add_argument('--unlabeled-dir', type=str, default='./unlabeled_images',
                        help='Local directory for unlabeled images')
    parser.add_argument('--predictions-dir', type=str, default='./predictions',
                        help='Directory for YOLO predictions')
    parser.add_argument('--output-json', type=str, default='pseudo_predictions.json',
                        help='Output Label Studio JSON file')
    
    # Parameters
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Confidence threshold for predictions')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading unlabeled images (if already downloaded)')
    
    args = parser.parse_args()
    
    print("🚀 === Pseudo-Labeling Pipeline ===")
    
    # Step 1: Find and download unlabeled images
    if not args.skip_download:
        print("\n🔍 Step 1: Finding unlabeled images...")
        from find_unlabeled_s3 import find_unlabeled_images_s3
        
        find_unlabeled_images_s3(
            all_images_s3_path=args.all_images_s3,
            labeled_images_s3_path=args.labeled_images_s3,
            output_method='download',
            local_output_dir=args.unlabeled_dir
        )
    else:
        print(f"⏭️ Skipping download, using existing images in {args.unlabeled_dir}")
    
    # Step 2: Download trained model
    print("\n📥 Step 2: Downloading trained model...")
    model_path = download_trained_model(args.model_s3)
    
    # Step 3: Run predictions
    print("\n🔮 Step 3: Running YOLO predictions...")
    labels_dir = run_yolo_predictions(
        model_path=model_path,
        images_dir=args.unlabeled_dir,
        output_dir=args.predictions_dir,
        confidence=args.confidence
    )
    
    # Step 4: Convert to Label Studio format
    print("\n🔄 Step 4: Converting to Label Studio format...")
    
    # Try automatic conversion first
    output_json = convert_to_label_studio(
        predictions_dir=labels_dir,
        images_dir=args.unlabeled_dir,
        classes_file=args.classes_file,
        output_json=args.output_json
    )
    
    # If that fails, create manually
    if not output_json or not os.path.exists(output_json):
        output_json = create_manual_label_studio_format(
            predictions_dir=labels_dir,
            images_dir=args.unlabeled_dir,
            classes_file=args.classes_file,
            output_json=args.output_json
        )
    
    print(f"\n🎉 === Pipeline Complete ===")
    print(f"📁 Unlabeled images: {args.unlabeled_dir}")
    print(f"📁 Predictions: {labels_dir}")
    print(f"📄 Label Studio JSON: {output_json}")
    print(f"\n💡 Next steps:")
    print(f"1. Import {output_json} into Label Studio")
    print(f"2. Review and correct predictions")
    print(f"3. Export corrected data for final training")

if __name__ == '__main__':
    main()
