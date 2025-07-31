import json
import os
from urllib.parse import urlparse
import boto3

def convert_label_studio_to_yolo(json_path, image_dir, output_dir, classes):
    if json_path.startswith("s3://"):
        s3 = boto3.client('s3')
        parsed = urlparse(json_path)
        obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
        data = json.load(obj['Body'])
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)

    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)

    for item in data:
        img_uri = item['data']['image']
        img_name = os.path.basename(urlparse(img_uri).path)
        label_path = os.path.join(output_dir, 'labels/train', img_name.replace('.jpg', '.txt'))

        label_lines = []
        for ann in item['annotations']:
            for result in ann['result']:
                if 'value' in result and 'rectanglelabels' in result['value']:
                    label = result['value']['rectanglelabels'][0]
                    bbox = result['value']
                    cls_id = classes.index(label)

                    # YOLO requires normalized cx, cy, w, h
                    x = bbox['x'] / 100
                    y = bbox['y'] / 100
                    w = bbox['width'] / 100
                    h = bbox['height'] / 100
                    cx = x + w / 2
                    cy = y + h / 2
                    label_lines.append(f"{cls_id} {cx} {cy} {w} {h}")

        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))

