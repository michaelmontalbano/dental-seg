import json

# Path to the annotation JSON
ANNOTATION_FILE = "train.json"
TARGET_COMPANY = "Straumann"

def extract_straumann_images(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    matching = [
        os.path.basename(entry["image"])
        for entry in data
        if entry.get("company", "").strip().lower() == TARGET_COMPANY.lower()
    ]

    print(f"✅ Found {len(matching)} Straumann images.")
    for img in matching:
        print(img)

    return matching

if __name__ == "__main__":
    import os
    images = extract_straumann_images(ANNOTATION_FILE)

    # Optionally, copy to a local demo folder
    SOURCE_DIR = "/path/to/full/images"
    DEST_DIR = "./demo_images"
    os.makedirs(DEST_DIR, exist_ok=True)

    for img_name in images:
        src = os.path.join(SOURCE_DIR, img_name)
        dst = os.path.join(DEST_DIR, img_name)
        if os.path.exists(src):
            os.system(f"cp '{src}' '{dst}'")

