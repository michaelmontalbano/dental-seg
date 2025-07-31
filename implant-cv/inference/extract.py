import json
import os
import shutil

ANNOTATION_FILE = "train.json"
TARGET_COMPANY = "Straumann ITI"

def extract_straumann_images(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect image suffixes like 'blx-2_1.png'
    matching_suffixes = [
        os.path.basename(entry["image"])
        for entry in data
        if entry.get("company", "").strip().lower() == TARGET_COMPANY.lower()
    ]

    print(f"✅ Found {len(matching_suffixes)} Straumann image targets:")
    for name in matching_suffixes:
        print(f" - {name}")

    return matching_suffixes

if __name__ == "__main__":
    SUFFIXES = extract_straumann_images(ANNOTATION_FILE)

    SOURCE_DIR = os.path.expanduser("~/work/repos/data/preannotations_0722/images")
    DEST_DIR = os.path.expanduser("~/work/repos/cej-ac-detection/dir/inference/demo_images")
    os.makedirs(DEST_DIR, exist_ok=True)

    # List all files in source dir
    all_files = os.listdir(SOURCE_DIR)
    copied = 0

    for suffix in SUFFIXES:
        matches = [f for f in all_files if f.endswith(suffix)]
        for match in matches:
            src = os.path.join(SOURCE_DIR, match)
            dst = os.path.join(DEST_DIR, match)
            shutil.copy2(src, dst)
            copied += 1
            print(f"✅ Copied: {match}")

    print(f"\n📦 Copied {copied} total images to {DEST_DIR}")
