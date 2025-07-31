import argparse
from ultralytics import YOLO
import os
import torch
import logger
import os
from ultralytics.utils import SETTINGS

print("\n🔍 Ultralytics SETTINGS before override:")
print(SETTINGS)

# Override dataset dir to match SageMaker's actual mount point
SETTINGS['datasets_dir'] = '/opt/ml/input/data/train'

print("\n📌 Overriding datasets_dir to:")
print(SETTINGS['datasets_dir'])

# Confirm expected directories
paths_to_check = [
    '/opt/ml/input/data/train',
    '/opt/ml/input/data/train/images',
    '/opt/ml/input/data/train/images/train',
    '/opt/ml/input/data/train/labels',
    '/opt/ml/input/data/train/labels/train',
]

for path in paths_to_check:
    print(f"\n🔎 Checking path: {path}")
    if os.path.exists(path):
        print(f"✅ Exists: {path}")
        print(f"📁 Contents of {path[:60]}: {os.listdir(path)}")
    else:
        print(f"❌ MISSING: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to data.yaml")
    parser.add_argument('--model', type=str, default='yolov8n.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolo_custom')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    config = {
        'data': args.data_yaml,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': min(args.workers, 4),
        'project': args.project,
        'name': args.name,
        'lr0': args.learning_rate,

        # Add your full config here ↓
        'hsv_h': 0.025,
        'hsv_s': 0.35,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.075,
        'scale': 0.1,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'val': True,
        'save': True,
        'save_period': 10,
        'deterministic': True,
        'amp': True,
        'patience': 100,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'verbose': True,
    }

    model.train(**config)

    best_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    os.makedirs(args.model_dir, exist_ok=True)
    os.system(f"cp {best_model_path} {args.model_dir}/")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Training will be slow on CPU.")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    main()