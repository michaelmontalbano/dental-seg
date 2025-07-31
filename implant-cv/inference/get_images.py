import os
import shutil
import yaml

# --- Config ---
DATA_YAML = "pipeline/imp11-12/data.yaml"
IMAGE_DIRS = [
    "pipeline/imp11-12/train/images",
    "pipeline/imp11-12/valid/images"
]
LABEL_DIRS = [
    "pipeline/imp11-12/train/labels",
    "pipeline/imp11-12/valid/labels"
]
DEST_DIR = "demo_images"
TARGET_COMPANY = "Straumann"

os.makedirs(DEST_DIR, exist_ok=True)

# --- Load class map from YAML ---
with open(DATA_YAML, "r") as f:
    data_yaml = yaml.safe_load(f)

class_map = data_yaml["names"]
if isinstance(class_map, dict):
    class_map = {v: int(k) for k, v in class_map.items()}
else:
    class_map = {name: idx for idx, name in enumerate(class_map)}

target_id = class_map.get(TARGET_COMPANY)
if target_id is None:
    raise ValueError(f"Company '{TARGET_COMPANY}' not found in data.yaml")

# --- Scan label files ---
copied = []
for label_dir, image_dir in zip(LABEL_DIRS, IMAGE_DIRS):
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, file)
        with open(label_path, "r") as f:
            content = f.read()
        if content.strip().startswith(str(target_id)):
            base_name = file.replace(".txt", "")
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(image_dir, base_name + ext)
                if os.path.exists(candidate):
                    shutil.copy(candidate, os.path.join(DEST_DIR, os.path.basename(candidate)))
                    copied.append(os.path.basename(candidate))
                    break

print(f"✅ Copied {len(copied)} Straumann images to {DEST_DIR}")

