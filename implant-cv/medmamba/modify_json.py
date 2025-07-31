import json
import os

INPUT_JSON = "train_augmented.json"  # Your original file
OUTPUT_JSON = "train_augmented_sagemaker.json"  # New output
SAGEMAKER_IMAGE_ROOT = "/opt/ml/input/data/train_augmented/images"

def rewrite_paths(input_file, output_file, new_root):
    with open(input_file, "r") as f:
        entries = json.load(f)

    for entry in entries:
        old_path = entry["image"]
        filename = os.path.basename(old_path)
        entry["image"] = os.path.join(new_root, filename)

    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"✅ Rewritten JSON saved to {output_file}")

if __name__ == "__main__":
    rewrite_paths(INPUT_JSON, OUTPUT_JSON, SAGEMAKER_IMAGE_ROOT)
