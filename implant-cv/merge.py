import os
import shutil
import yaml
from pathlib import Path

# Input dataset directories
imp11_12_path = Path("~/work/repos/cej-ac-detection/dir/inference/pipeline/imp11-12").expanduser()
combined_path = Path("~/work/repos/data/combined").expanduser()
merged_path = Path("~/work/repos/data/merged_implants").expanduser()

# Make folder structure: train/{images,labels} and val/{images,labels}
for split in ["train/images", "train/labels", "val/images", "val/labels"]:
    (merged_path / split).mkdir(parents=True, exist_ok=True)

# Helper to copy from imp11-12 and combined
def copy_data(src_base, split):
    for kind in ["images", "labels"]:
        src_dir = src_base / split / kind
        dst_dir = merged_path / split / kind
        if not src_dir.exists():
            print(f"Skipping missing {src_dir}")
            continue
        for f in src_dir.glob("*"):
            shutil.copy(f, dst_dir / f.name)

# Copy both datasets
for split in ["train", "val"]:
    copy_data(imp11_12_path, split)
    copy_data(combined_path, split)

# Merge and deduplicate class names
with open(imp11_12_path / "data.yaml") as f1, open(combined_path / "data.yaml") as f2:
    d1 = yaml.safe_load(f1)
    d2 = yaml.safe_load(f2)

merged_names = list(dict.fromkeys(list(d1["names"].values()) + list(d2["names"].values())))

# Write the merged data.yaml
merged_yaml = {
    "train": "train/images",
    "val": "val/images",
    "names": {i: name for i, name in enumerate(merged_names)}
}
with open(merged_path / "data.yaml", "w") as f:
    yaml.dump(merged_yaml, f, sort_keys=False)

print("✅ Local merge complete. Ready for upload to:")
print("s3://codentist-general/datasets/merged_implants/")

