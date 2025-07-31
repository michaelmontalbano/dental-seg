import os
import json
import boto3
from urllib.parse import urlparse
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ImplantDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.entries = self._load_json(json_path)
        self.image_root = image_root
        self.transform = transform or T.ToTensor()
        self.samples = self._parse_entries()

    def _load_json(self, path):
        if path.startswith("s3://"):
            s3 = boto3.client("s3")
            parsed = urlparse(path)
            obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            return json.load(obj["Body"])
        else:
            with open(path, "r") as f:
                return json.load(f)

    def _parse_entries(self):
        samples = []
        for entry in self.entries:
            image_path = entry["data"]["image"]
            annotations = entry.get("annotations", [])
            labels = []

            for ann in annotations:
                for result in ann.get("result", []):
                    if "value" in result and "rectanglelabels" in result["value"]:
                        labels.extend(result["value"]["rectanglelabels"])

            if labels:
                samples.append((image_path, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_uri, labels = self.samples[idx]
        image_path = self._resolve_path(image_uri)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), labels

    def _resolve_path(self, s3_path):
        if s3_path.startswith("s3://"):
            parsed = urlparse(s3_path)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            local_file = f"/tmp/{os.path.basename(key)}"
            if not os.path.exists(local_file):
                boto3.client("s3").download_file(bucket, key, local_file)
            return local_file
        elif self.image_root.startswith("s3://"):
            # relative filename assumed
            filename = os.path.basename(s3_path)
            parsed = urlparse(self.image_root)
            bucket = parsed.netloc
            prefix = parsed.path.strip("/") + "/" + filename
            local_file = f"/tmp/{filename}"
            if not os.path.exists(local_file):
                boto3.client("s3").download_file(bucket, prefix, local_file)
            return local_file
        else:
            return os.path.join(self.image_root, os.path.basename(s3_path))

