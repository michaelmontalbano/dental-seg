import json
import random
import os

# Path to your demo images
demo_dir = "/Users/michaelmontalbano/work/repos/cej-ac-detection/dir/inference/demo_images"
output_json = "/Users/michaelmontalbano/work/repos/cej-ac-detection/dir/inference/simulate_predictions.json"

# Define correct prediction label (Straumann ITI)
STRAUMANN_ID = 33
# Define 5 plausible incorrect predictions
INCORRECT_IDS = [24, 0, 19, 21, 34]  # MIS, 3M ESPE, Keystone Dental, Implant Direct, Titan Implants

# Gather all image paths
all_images = sorted([
    os.path.join("demo_images", f) for f in os.listdir(demo_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Shuffle and insert 5 incorrect ones
random.seed(42)
incorrect_indices = sorted(random.sample(range(len(all_images)), 5))
predictions = []

for i, img_path in enumerate(all_images):
    if i in incorrect_indices:
        pred_class = random.choice(INCORRECT_IDS)
    else:
        pred_class = STRAUMANN_ID

    predictions.append({
        "image": img_path,
        "predicted_class": pred_class
    })

# Save to JSON
with open(output_json, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"✅ Simulated predictions saved to: {output_json}")

